"""Measure what the metric costs to run, in pairs per second.

Why. A metric is used, not just published. v1 ships a 110 M encoder; the v2 candidate is a
435 M one, and the symmetry guarantee doubles the forward passes on top. Three to four
times the parameters and twice the passes is not a detail for someone scoring a corpus of
a hundred thousand simplifications, and the article has to state the price next to the
gain rather than leave the reader to guess it.

CPU is measured as well as GPU, and it is not an afterthought: an evaluation metric is
often run inside a training loop or on a laptop, where no GPU is free.

Run::

    PYTHONPATH=src python src/diagnostics/benchmark_inference.py \\
        --checkpoint davebulaval/MeaningBERT --checkpoint <other> --device cuda --device cpu
"""

from __future__ import annotations

import json
import time
from typing import Optional

import click

try:  # PYTHONPATH=src.
    from meaningbert.scorer import MeaningBERTScorer
except ImportError:  # pragma: no cover
    from src.meaningbert.scorer import MeaningBERTScorer  # type: ignore


#: A sentence pair of representative length. Timing on real corpus text rather than on a
#: toy string: attention is quadratic in length, so a five-word pair would flatter the
#: larger model by hiding the cost it only pays on real input.
DOCUMENT = (
    "In other developments, both Iceland and Greenland accepted the overlordship of Norway, "
    "but Scotland was able to repulse a Norse invasion and broker a favorable peace settlement."
)
SIMPLIFICATION = (
    "Iceland and Greenland accepted Norway as their ruler, but Scotland pushed back the Norse "
    "invasion and negotiated a good peace deal."
)


def footprint(scorer: MeaningBERTScorer, device: str) -> dict:
    """What the model occupies: parameters, bytes of weights, peak VRAM while scoring.

    Peak and not resident: what decides whether the metric fits alongside the model being
    evaluated is the high-water mark during a batch, not the weights at rest.
    """
    import torch

    params = sum(p.numel() for p in scorer._model.parameters())  # noqa: SLF001
    weights_bytes = sum(p.numel() * p.element_size() for p in scorer._model.parameters())  # noqa: SLF001
    out = {"parameters": params, "weights_mb": weights_bytes / 1e6, "peak_vram_mb": float("nan")}
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        scorer.score([DOCUMENT] * 32, [SIMPLIFICATION] * 32)
        torch.cuda.synchronize()
        out["peak_vram_mb"] = torch.cuda.max_memory_allocated() / 1e6
    return out


def measure(scorer: MeaningBERTScorer, pairs: int, repeats: int) -> dict:
    """Pairs per second, taking the best of *repeats* runs.

    The best and not the mean: the slow runs measure what else the machine was doing, and
    the question here is what the metric costs, not what the machine was busy with.
    """
    documents = [DOCUMENT] * pairs
    simplifications = [SIMPLIFICATION] * pairs
    scorer.score(documents[:8], simplifications[:8])  # Warm up kernels and allocator.

    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        scorer.score(documents, simplifications)
        timings.append(time.perf_counter() - start)
    best = min(timings)
    return {"seconds": best, "pairs_per_second": pairs / best, "all_timings": timings}


@click.command()
@click.option("--checkpoint", "checkpoints", multiple=True, required=True, help="Repeatable.")
@click.option("--device", "devices", multiple=True, default=("cuda", "cpu"), show_default=True)
@click.option("--pairs", default=256, show_default=True, help="Pairs scored per measurement.")
@click.option("--repeats", default=3, show_default=True)
@click.option("--batch-size", default=32, show_default=True)
@click.option("--json-out", default=None, help="Where the measurements are written.")
def main(checkpoints, devices, pairs: int, repeats: int, batch_size: int, json_out: Optional[str]) -> None:
    """Time every checkpoint on every device, symmetric and not."""
    import torch

    rows = []
    for checkpoint in checkpoints:
        for device in devices:
            if device == "cuda" and not torch.cuda.is_available():
                print(f"  {device} indisponible, saute")
                continue
            scorer = MeaningBERTScorer(checkpoint, device=device, batch_size=batch_size)
            size = footprint(scorer, device)
            params = size["parameters"]
            print(f"{checkpoint:58} {device:5} {params / 1e6:6.1f} M parametres, "
                  f"{size['weights_mb']:7.0f} Mo de poids, pic VRAM {size['peak_vram_mb']:7.0f} Mo")
            got = measure(scorer, pairs, repeats)
            rows.append({"checkpoint": checkpoint, "device": device, **size,
                         **{k: v for k, v in got.items() if k != "all_timings"}})
            print(f"{checkpoint:58} {device:5} {got['pairs_per_second']:8.1f} paires/s   "
                  f"{pairs / got['pairs_per_second']:6.2f} s pour {pairs} paires")
            del scorer
            if device == "cuda":
                torch.cuda.empty_cache()

    if json_out:
        with open(json_out, "w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2)
        print(f"\necrit dans {json_out}")

    base = next((r for r in rows if "bert-base-uncased" in r["checkpoint"]), None)
    if base:
        print("\nrapport a bert-base-uncased, a materiel egal :")
        for r in rows:
            if r is base:
                continue
            ref = next((b for b in rows if b["device"] == r["device"] and "bert-base-uncased" in b["checkpoint"]), None)
            if ref:
                print(f"  {r['checkpoint'][:44]:44} {r['device']:5} "
                      f"{ref['pairs_per_second'] / r['pairs_per_second']:5.1f} fois plus lent")


if __name__ == "__main__":
    main()
