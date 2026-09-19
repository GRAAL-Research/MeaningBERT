"""Derive training settings from the GPU actually present.

The v1 sweep scripts hard-code ``--bf16``. On the v2 training host that flag is a lie:
Pascal has no bfloat16, so the run either dies or silently falls back, and the
configuration recorded in wandb no longer describes what ran.

Everything here is a pure function of the compute capability, so it is testable without a
GPU and without torch. Only :func:`detect` touches torch.

Thresholds, and what each one gates:

============ ======= ==========================================================
Capability   Since   What it unlocks
============ ======= ==========================================================
7.0          Volta   Triton, therefore ``torch.compile`` with the inductor backend
7.0          Volta   fp16 tensor cores, therefore fp16 that is actually faster
8.0          Ampere  bfloat16, and FlashAttention
============ ======= ==========================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

TRITON_MIN_CAPABILITY: float = 7.0
FP16_TENSOR_CORE_MIN_CAPABILITY: float = 7.0
BF16_MIN_CAPABILITY: float = 8.0
FLASH_ATTENTION_MIN_CAPABILITY: float = 8.0


@dataclass(frozen=True)
class DeviceCapability:
    """What one CUDA device can do."""

    name: str
    major: int
    minor: int
    total_memory_gb: float

    @property
    def capability(self) -> float:
        """Compute capability as a comparable number, e.g. 6.1."""
        return self.major + self.minor / 10.0

    @property
    def arch(self) -> str:
        """The ``sm_XY`` tag torch reports in :func:`torch.cuda.get_arch_list`."""
        return f"sm_{self.major}{self.minor}"

    @property
    def supports_bf16(self) -> bool:
        """Native bfloat16."""
        return self.capability >= BF16_MIN_CAPABILITY

    @property
    def has_fp16_tensor_cores(self) -> bool:
        """Whether fp16 buys throughput, rather than only halving memory."""
        return self.capability >= FP16_TENSOR_CORE_MIN_CAPABILITY

    @property
    def supports_triton(self) -> bool:
        """Whether ``torch.compile`` can use the inductor backend."""
        return self.capability >= TRITON_MIN_CAPABILITY

    @property
    def supports_flash_attention(self) -> bool:
        """Whether the FlashAttention SDPA backend is usable."""
        return self.capability >= FLASH_ATTENTION_MIN_CAPABILITY


def recommended_precision(device: DeviceCapability) -> str:
    """Pick the precision to train in.

    ``fp16`` is deliberately never recommended. Where it would be fast the device also has
    ``bf16``, which has the same speed and a far wider exponent range; where ``bf16`` is
    missing, fp16 arithmetic runs at a fraction of fp32 throughput and only saves memory.
    Memory is a reason to reach for fp16, and it is a decision to take against a measured
    out-of-memory error, not in advance.
    """
    return "bf16" if device.supports_bf16 else "fp32"


def recommended_compile_backend(device: DeviceCapability) -> Optional[str]:
    """Pick a ``torch.compile`` backend, or ``None`` to skip compilation."""
    return "inductor" if device.supports_triton else None


def recommended_attention_backends(device: DeviceCapability) -> list[str]:
    """SDPA backends usable on this device, best first."""
    backends = ["flash", "mem_efficient", "math"] if device.supports_flash_attention else ["mem_efficient", "math"]
    return backends


def blocked_features(device: DeviceCapability) -> dict[str, str]:
    """Features this device cannot run, and the threshold each one needs.

    Returned rather than logged so callers can fail loudly on a flag the user passed
    explicitly, instead of downgrading in silence.
    """
    blocked: dict[str, str] = {}
    if not device.supports_bf16:
        blocked["bf16"] = f"needs compute capability >= {BF16_MIN_CAPABILITY}, device is {device.capability}"
    if not device.supports_triton:
        blocked["torch.compile"] = (
            f"Triton needs compute capability >= {TRITON_MIN_CAPABILITY}, device is {device.capability}"
        )
    if not device.supports_flash_attention:
        blocked["flash_attention"] = (
            f"needs compute capability >= {FLASH_ATTENTION_MIN_CAPABILITY}, device is {device.capability}"
        )
    if not device.has_fp16_tensor_cores:
        blocked["fp16_speedup"] = (
            f"fp16 tensor cores need compute capability >= {FP16_TENSOR_CORE_MIN_CAPABILITY}, "
            f"device is {device.capability}; fp16 would save memory and cost speed"
        )
    return blocked


def describe(device: DeviceCapability) -> str:
    """One-screen summary of what this device allows."""
    lines = [
        f"{device.name} ({device.arch}, capability {device.capability}, {device.total_memory_gb:.1f} GB)",
        f"  precision      : {recommended_precision(device)}",
        f"  compile backend: {recommended_compile_backend(device) or 'none (Triton unavailable)'}",
        f"  attention      : {', '.join(recommended_attention_backends(device))}",
    ]
    blocked = blocked_features(device)
    if blocked:
        lines.append("  blocked        :")
        lines.extend(f"    - {feature}: {reason}" for feature, reason in sorted(blocked.items()))
    return "\n".join(lines)


def detect(index: int = 0) -> DeviceCapability:
    """Read the capability of CUDA device *index*.

    Raises:
        RuntimeError: If torch cannot see a CUDA device, which on this host usually means
            the loaded kernel module does not match the installed userspace driver.
    """
    import torch  # local import so the module stays usable without torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "torch reports no CUDA device. If the host has GPUs, compare "
            "/proc/driver/nvidia/version with the installed nvidia-driver package: a "
            "mismatch after an upgrade needs a reboot."
        )
    major, minor = torch.cuda.get_device_capability(index)
    properties = torch.cuda.get_device_properties(index)
    return DeviceCapability(
        name=properties.name,
        major=major,
        minor=minor,
        total_memory_gb=properties.total_memory / 1024**3,
    )


def assert_arch_compiled(device: DeviceCapability, arch_list: list[str]) -> None:
    """Fail if the installed torch has no kernels for this device.

    This is the failure that wastes the most time: torch imports, ``cuda.is_available()``
    returns True, and the first real kernel launch dies with "no kernel image is available
    for execution on the device". PyTorch drops old architectures from its wheels over
    time, and Pascal is on the way out, so the check belongs in every environment probe.

    Args:
        device: Detected device.
        arch_list: Output of :func:`torch.cuda.get_arch_list`.

    Raises:
        RuntimeError: If *device*'s architecture is absent from *arch_list*.
    """
    if device.arch not in arch_list:
        raise RuntimeError(
            f"this torch build has no {device.arch} kernels for {device.name}; "
            f"it was compiled for {', '.join(arch_list)}. Install a build that still ships "
            f"{device.arch}: the cu126 channel keeps Pascal, cu128 dropped part of it, and "
            f"PyTorch 2.15 removes it entirely."
        )
