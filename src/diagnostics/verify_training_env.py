"""Prove the training environment works, rather than assuming it does.

``torch.cuda.is_available()`` returning True proves almost nothing. It stays True when the
installed wheel has no kernels for the device, and the failure only surfaces on the first
real kernel launch, often minutes into a run. This script launches one.

Checks, in order, stopping at the first hard failure:

1. torch imports, and reports its CUDA build;
2. a CUDA device is visible;
3. **the wheel contains kernels for this device's architecture**;
4. a real forward and backward pass runs on it;
5. the precision the code would pick matches what the device supports;
6. throughput, recorded so two environments can be compared after an upgrade.

Run on the training host::

    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python src/diagnostics/verify_training_env.py --gpu 0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

from diagnostics.hardware import (
    assert_arch_compiled,
    compatible_archs,
    blocked_features,
    describe,
    detect,
    recommended_precision,
)


def _check(label: str, passed: bool, detail: str = "") -> bool:
    """Print one check line and return *passed*."""
    mark = "OK  " if passed else "FAIL"
    print(f"[{mark}] {label}" + (f" - {detail}" if detail else ""))
    return passed


def measure_throughput(index: int, hidden: int = 1024, batch: int = 16, seq: int = 128, steps: int = 20) -> float:
    """Time a forward and backward pass on a transformer-shaped workload.

    Not a benchmark of the real model, a stable yardstick: the same shapes measured before
    and after a server upgrade say whether anything regressed.

    Returns:
        Steps per second.
    """
    import torch

    device = torch.device(f"cuda:{index}")
    torch.backends.cudnn.benchmark = True
    layer = torch.nn.TransformerEncoderLayer(
        d_model=hidden, nhead=16, dim_feedforward=4 * hidden, batch_first=True
    ).to(device)
    optimiser = torch.optim.AdamW(layer.parameters(), lr=1e-5)
    data = torch.randn(batch, seq, hidden, device=device)

    for _ in range(3):  # warm-up: cudnn autotuning and allocator growth
        layer(data).mean().backward()
        optimiser.step()
        optimiser.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)

    start = time.perf_counter()
    for _ in range(steps):
        layer(data).mean().backward()
        optimiser.step()
        optimiser.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)
    return steps / (time.perf_counter() - start)


def verify(index: int, skip_throughput: bool = False) -> tuple[bool, dict[str, Any]]:
    """Run every check against CUDA device *index*."""
    report: dict[str, Any] = {}

    try:
        import torch
    except ImportError as error:
        _check("torch imports", False, str(error))
        return False, {"error": "torch not installed"}

    report["torch_version"] = torch.__version__
    report["torch_cuda"] = torch.version.cuda
    _check("torch imports", True, f"{torch.__version__}, CUDA build {torch.version.cuda}")

    if not _check("a CUDA device is visible", torch.cuda.is_available()):
        print("\n  A driver/library mismatch after an upgrade is the usual cause.")
        print("  Compare /proc/driver/nvidia/version with `dpkg -l | grep nvidia-driver`.")
        print("  If they differ, reboot.")
        return False, report

    device = detect(index)
    arch_list = torch.cuda.get_arch_list()
    report["device"] = device.name
    report["arch"] = device.arch
    report["capability"] = device.capability
    report["memory_gb"] = round(device.total_memory_gb, 1)
    report["arch_list"] = arch_list
    _check("device detected", True, f"{device.name}, {device.arch}, {device.total_memory_gb:.1f} GB")

    label = f"the wheel has kernels that run on {device.arch}"
    try:
        assert_arch_compiled(device, arch_list)
        usable = compatible_archs(device, arch_list)
        exact = " (exact match)" if device.arch in usable else " (via CUDA's upward binary compatibility)"
        report["compatible_archs"] = usable
        _check(label, True, f"{', '.join(usable)}{exact}")
    except RuntimeError as error:
        _check(label, False, str(error))
        return False, report

    try:
        probe = torch.nn.Linear(64, 64).to(f"cuda:{index}")
        out = probe(torch.randn(8, 64, device=f"cuda:{index}")).sum()
        out.backward()
        assert probe.weight.grad is not None
        _check("a real forward and backward pass runs", True)
        report["fwd_bwd"] = True
    except Exception as error:  # noqa: BLE001 - any failure here is fatal and worth printing
        _check("a real forward and backward pass runs", False, f"{type(error).__name__}: {error}")
        report["fwd_bwd"] = False
        return False, report

    precision = recommended_precision(device)
    report["recommended_precision"] = precision
    report["blocked"] = blocked_features(device)
    _check("precision derived from the device", True, precision)

    if not skip_throughput:
        rate = measure_throughput(index)
        report["steps_per_second"] = round(rate, 2)
        _check("throughput measured", True, f"{rate:.2f} steps/s on the reference workload")

    print()
    print(describe(device))
    return True, report


def main() -> None:
    """Verify the training environment and exit non-zero if it is not usable."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device index, after CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--skip-throughput", action="store_true", help="Skip the timing step.")
    parser.add_argument("--json-out", default=None, help="Where to write the machine-readable report.")
    args = parser.parse_args()

    ok, report = verify(args.gpu, args.skip_throughput)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(f"\nreport written to {args.json_out}")

    print("\n" + ("Environment usable for training." if ok else "Environment NOT usable for training."))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
