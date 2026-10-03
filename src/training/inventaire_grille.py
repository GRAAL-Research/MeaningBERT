"""List the cells of the grid that have no result, and say where each one can be re-run.

Two ways a cell goes missing, and both happened on the first two nights. It can wedge the
card and be killed by the watchdog, which is what ``stsb-roberta-base`` did ten times on
renard-gpu0. Or it can be skipped because a claim left behind by a worker killed with -9
was still fresh when the driver reached it, which is what happened to five ``bert`` cells
on souris. Neither leaves a hole the driver revisits: a slice is one pass.

So the recovery is a separate inventory, computed from the results that exist rather than
from what the plan said should exist. It reads the per-machine results trees, since there
is no shared filesystem, and emits a CELLS line per worker for whatever is still missing.

Run::

    PYTHONPATH=src python src/training/inventaire_grille.py --roots renard=/tmp/r,souris=/tmp/s
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import click

from training.plan_polarity_grid import ARCHS, CONDITION_COST, WORKERS, BASE_MINUTES, FORBIDDEN, parse_seeds


def done_cells(root: Path) -> set[tuple[str, int, str]]:
    """Every cell that has a ``metrics.json`` under *root*.

    The directory name carries the architecture and the condition, ``<arch>-<condition>``,
    and the condition is taken from the END because an architecture tag can itself contain
    a dash. Splitting from the left would read ``nli-deberta-v3-base-none`` as architecture
    ``nli`` and quietly report every cell of that model as missing.
    """
    found: set[tuple[str, int, str]] = set()
    for metrics in root.glob("*/seed*/metrics.json"):
        tag = metrics.parent.parent.name
        arch, _, condition = tag.rpartition("-")
        if arch in ARCHS:
            found.add((arch, int(metrics.parent.name.removeprefix("seed")), condition))
    return found


def missing(
    done: dict[str, set[tuple[str, int, str]]],
    archs: list[str],
    seeds: list[int],
    conditions: list[str],
) -> list[tuple[str, int, str]]:
    """Cells absent from EVERY machine, heaviest first."""
    everywhere = set().union(*done.values()) if done else set()
    absent = [
        (arch, seed, condition)
        for arch in archs
        for seed in seeds
        for condition in conditions
        if (arch, seed, condition) not in everywhere
    ]
    return sorted(absent, key=lambda cell: (-ARCHS[cell[0]][0] * CONDITION_COST[cell[2]], cell))


def reassign(cells: list[tuple[str, int, str]], workers: list[str]) -> dict[str, list]:
    """Place the missing cells on the workers that can actually run them.

    A cell that went missing because its pairing wedges a card must not be sent back to
    that card, which is the whole reason :data:`FORBIDDEN` exists.
    """
    load = {name: 0.0 for name in workers}
    placed: dict[str, list] = {name: [] for name in workers}
    for arch, seed, condition in cells:
        cost = ARCHS[arch][0] * CONDITION_COST[condition]
        capable = [name for name in workers if WORKERS[name][0] >= ARCHS[arch][1] and (arch, name) not in FORBIDDEN]
        roomy = [name for name in capable if cost < 3 or WORKERS[name][1] >= 12]
        eligible = roomy or capable
        if not eligible:
            raise ValueError(f"{arch} seed {seed} {condition} : aucun worker ne peut la prendre")
        chosen = min(eligible, key=lambda name: (load[name], name))
        placed[chosen].append((arch, seed, condition, cost))
        load[chosen] += cost
    return placed


@click.command()
@click.option("--results", "results_dirs", multiple=True, required=True, help="Local copies, one per machine.")
@click.option("--seeds", default="42-51", show_default=True)
@click.option("--conditions", default="none,full", show_default=True)
@click.option("--workers", default="renard-gpu0,renard-gpu1,souris-gpu0", show_default=True)
@click.option("--json-out", default=None)
def main(results_dirs, seeds: str, conditions: str, workers: str, json_out: Optional[str]) -> None:
    """Report the holes and print a CELLS line per worker to close them."""
    done = {str(path): done_cells(Path(path)) for path in results_dirs}
    total_done = len(set().union(*done.values())) if done else 0

    worker_names = [name.strip() for name in workers.split(",") if name.strip()]
    capability = max(WORKERS[name][0] for name in worker_names)
    archs = [tag for tag, (_, needed) in ARCHS.items() if needed <= capability]
    condition_list = [name.strip() for name in conditions.split(",") if name.strip()]

    holes = missing(done, archs, parse_seeds(seeds), condition_list)
    planned = len(archs) * len(parse_seeds(seeds)) * len(condition_list)
    click.echo(f"{total_done} cellules faites sur {planned}, {len(holes)} manquantes\n")

    by_arch: dict[str, int] = {}
    for arch, _, condition in holes:
        by_arch[f"{arch}-{condition}"] = by_arch.get(f"{arch}-{condition}", 0) + 1
    for tag, count in sorted(by_arch.items(), key=lambda item: -item[1]):
        click.echo(f"  {tag:34} {count:>3}")

    if not holes:
        return
    click.echo("")
    placed = reassign(holes, worker_names)
    for name in worker_names:
        for condition in condition_list:
            slice_ = [cell for cell in placed[name] if cell[2] == condition]
            if not slice_:
                continue
            hours = sum(cost for *_, cost in slice_) * BASE_MINUTES / 60
            click.echo(f"# {name} / {condition} : {len(slice_)} cellules, {hours:.0f} h")
            click.echo(";".join(f"{arch}:{seed}" for arch, seed, _, _ in slice_))
            click.echo("")

    if json_out:
        with open(json_out, "w", encoding="utf-8") as handle:
            json.dump(
                {n: [{"arch": a, "seed": s, "condition": c} for a, s, c, _ in v] for n, v in placed.items()},
                handle,
                indent=2,
            )


if __name__ == "__main__":
    main()
