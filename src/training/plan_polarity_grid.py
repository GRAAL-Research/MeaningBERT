"""Partition the experiment 3 grid across the GPUs, statically.

Static and not dynamic, for a reason the v2 campaign paid for: there is no shared
filesystem, so each worker decides what to skip by looking at its OWN results tree. A cell
finished on souris does not exist on renard, and a worker told to run it would recompute
it. That happened twenty times. A slice must therefore name only cells whose results will
live on that machine, which is exactly what this emits.

Assignment is longest-processing-time first: the most expensive cells are placed while
there is still room to balance them. Cost is measured, not guessed, from the runs of
2026-09-25: a base model takes about 72 minutes on the ``_none`` condition, ``_full`` is
1.45 times longer because it carries 1.5 times the rows, and a large model is 4.5 times a
base one. That last figure was 3.0 until it was measured: deberta-v3-large runs at 1.93
seconds per optimiser step against 2.5 steps per second for a base model, and planning on
the guess left one card with 8 days of work and another idle after 4.

Run::

    PYTHONPATH=src python src/training/plan_polarity_grid.py --seeds 42-51
"""

from __future__ import annotations

import json
from typing import Final, Optional

import click

#: tag -> (relative cost, minimum compute capability x10). Cost is in units of one base
#: model on the ``_none`` condition, which is about 72 minutes on a GTX 1080 Ti.
ARCHS: Final[dict[str, tuple[float, int]]] = {
    "bert": (1.0, 0),
    "deberta-v3-base": (1.0, 0),
    "nli-deberta-v3-base": (1.0, 0),
    "stsb-roberta-base": (1.0, 0),
    "deberta-v3-large": (4.5, 0),
    "nli-deberta-v3-large": (4.5, 0),
    "roberta-large-mnli": (4.5, 0),
    "modernbert-base": (1.0, 80),
    "modernbert-large": (4.5, 80),
}

#: The augmented condition carries 1.5 times the training rows, and the evaluation splits
#: are identical, so it costs a little less than 1.5 times a ``_none`` run.
CONDITION_COST: Final[dict[str, float]] = {"none": 1.0, "full": 1.45}

#: One base unit, in minutes, measured on a GTX 1080 Ti at micro-batch 8, length 256, fp32.
BASE_MINUTES: Final[float] = 72.0

#: name -> (compute capability x10, VRAM in GB). A large model at micro-batch 4 needs about
#: 10 GB, which the 1080 Ti can only just hold, so large cells are steered away from it;
#: the runner's out-of-memory fallback is a safety net, not a plan.
WORKERS: Final[dict[str, tuple[int, int]]] = {
    "renard-gpu0": (61, 11),
    "renard-gpu1": (61, 16),
    "souris-gpu0": (61, 12),
    "caribou-gpu0": (89, 24),
}


def parse_seeds(text: str) -> list[int]:
    """Parse ``42-51`` or ``42,43,44`` into a list of seeds."""
    if "-" in text:
        low, high = text.split("-", 1)
        return list(range(int(low), int(high) + 1))
    return [int(part) for part in text.split(",") if part.strip()]


def cells(archs: list[str], seeds: list[int], conditions: list[str]) -> list[tuple[str, int, str, float]]:
    """Every cell of the grid with its cost, heaviest first."""
    built = [
        (arch, seed, condition, ARCHS[arch][0] * CONDITION_COST[condition])
        for arch in archs
        for seed in seeds
        for condition in conditions
    ]
    return sorted(built, key=lambda cell: (-cell[3], cell[0], cell[1], cell[2]))


def assign(
    grid: list[tuple[str, int, str, float]],
    workers: list[str],
) -> dict[str, list[tuple[str, int, str, float]]]:
    """Place each cell on the least loaded worker that can run it.

    Args:
        grid: Output of :func:`cells`, heaviest first.
        workers: Worker names, keys of :data:`WORKERS`.

    Returns:
        Worker name to its cells, in the order it should run them.

    Raises:
        ValueError: If a cell fits no worker. Silently dropping it would produce a grid
            with holes that only show up in the analysis, weeks later.
    """
    load = {name: 0.0 for name in workers}
    placed: dict[str, list] = {name: [] for name in workers}

    for arch, seed, condition, cost in grid:
        needed = ARCHS[arch][1]
        # A large cell wants more than 11 GB to be comfortable. Steering, not a hard rule:
        # the runner halves the micro-batch on out-of-memory, so a misplacement costs speed
        # and not a result.
        roomy = [name for name in workers if WORKERS[name][0] >= needed and (cost < 3 or WORKERS[name][1] >= 12)]
        eligible = roomy or [name for name in workers if WORKERS[name][0] >= needed]
        if not eligible:
            raise ValueError(
                f"{arch} seed {seed} {condition} needs compute {needed} and no worker has it; "
                "a dropped cell becomes a hole nobody notices until the analysis"
            )
        chosen = min(eligible, key=lambda name: (load[name], name))
        placed[chosen].append((arch, seed, condition, cost))
        load[chosen] += cost

    return placed


@click.command()
@click.option("--seeds", default="42-51", show_default=True)
@click.option("--archs", default="", help="Comma separated; default is every Pascal-capable arch.")
@click.option("--conditions", default="none,full", show_default=True)
@click.option("--workers", default="renard-gpu0,renard-gpu1,souris-gpu0", show_default=True)
@click.option("--json-out", default=None)
def main(seeds: str, archs: str, conditions: str, workers: str, json_out: Optional[str]) -> None:
    """Print one CELLS line per worker, ready to paste into run_polarity.sh."""
    worker_names = [name.strip() for name in workers.split(",") if name.strip()]
    capability = max(WORKERS[name][0] for name in worker_names)
    chosen = (
        [tag.strip() for tag in archs.split(",") if tag.strip()]
        if archs
        else [tag for tag, (_, needed) in ARCHS.items() if needed <= capability]
    )
    condition_list = [name.strip() for name in conditions.split(",") if name.strip()]

    placed = assign(cells(chosen, parse_seeds(seeds), condition_list), worker_names)

    total = sum(cost for cells_ in placed.values() for *_, cost in cells_)
    click.echo(f"{sum(len(v) for v in placed.values())} cellules, {total * BASE_MINUTES / 60:.0f} h-GPU au total\n")
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
                {name: [{"arch": a, "seed": s, "condition": c} for a, s, c, _ in v] for name, v in placed.items()},
                handle,
                indent=2,
            )
        click.echo(f"plan : {json_out}")


if __name__ == "__main__":
    main()
