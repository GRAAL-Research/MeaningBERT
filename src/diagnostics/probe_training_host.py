"""Probe a training host and derive what it can actually train.

``docs/serveur-entrainement-renard.md`` was written by hand from a one-off inspection.
Driver, CUDA and Python versions on a training box change, and a hand-written capability
sheet rots silently, which is the worst way for it to be wrong. This module re-derives the
sheet from the host itself.

Usage::

    python src/diagnostics/probe_training_host.py --host renard --gpu 1 \
        --json-out results/host-renard.json --markdown-out docs/serveur-releve-renard.md

The interesting part is not the collection, it is :func:`derive`: compute capability
decides bf16, and parameter counts plus optimiser state decide which checkpoints fit. Both
are mechanical, so neither should ever be asserted from memory again.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Optional

#: bfloat16 needs Ampere or newer.
BF16_MIN_COMPUTE_CAPABILITY: float = 8.0
#: Tensor cores for fp16 appear with Volta.
FP16_TENSOR_CORE_MIN_COMPUTE_CAPABILITY: float = 7.0

#: Bytes per parameter for AdamW in fp32: weights, gradients, and two optimiser moments.
BYTES_PER_PARAM_ADAMW_FP32: int = 16

#: Total parameter counts, embeddings included. DeBERTa's 128k-token vocabulary makes the
#: embedding matrix a large share of the total, so backbone-only counts mislead here.
CHECKPOINT_PARAMS: dict[str, int] = {
    "microsoft/deberta-v3-small": 142_000_000,
    "microsoft/deberta-v3-base": 184_000_000,
    "microsoft/deberta-v3-large": 434_000_000,
    "microsoft/deberta-v2-xlarge": 900_000_000,
    "answerdotai/ModernBERT-large": 395_000_000,
    "bert-base-uncased": 110_000_000,
}

#: Share of GPU memory left for activations, fragmentation and the CUDA context.
ACTIVATION_HEADROOM: float = 0.35


@dataclass
class Gpu:
    """One GPU as reported by nvidia-smi."""

    index: int
    name: str
    memory_mib: int
    compute_capability: float

    @property
    def supports_bf16(self) -> bool:
        """Whether this GPU has native bfloat16."""
        return self.compute_capability >= BF16_MIN_COMPUTE_CAPABILITY

    @property
    def has_fp16_tensor_cores(self) -> bool:
        """Whether fp16 buys real throughput here rather than just halving memory."""
        return self.compute_capability >= FP16_TENSOR_CORE_MIN_COMPUTE_CAPABILITY


@dataclass
class HostProbe:
    """Everything collected from a training host in one pass."""

    host: str
    hostname: str
    gpus: list[Gpu] = field(default_factory=list)
    driver: str = ""
    cuda_toolkit: str = ""
    python_version: str = ""
    torch_version: str = ""
    torch_cuda: str = ""
    cpu_count: str = ""
    memory_gb: str = ""
    disk_free: str = ""
    errors: list[str] = field(default_factory=list)


def trainable(gpu: Gpu, params: int, bytes_per_param: int = BYTES_PER_PARAM_ADAMW_FP32) -> tuple[bool, float]:
    """Whether a model of *params* parameters can be fine-tuned on *gpu*.

    Counts weights, gradients and optimiser moments, then reserves
    :data:`ACTIVATION_HEADROOM` of the card for activations and the CUDA context. This is a
    screening estimate, not a guarantee: a model that clears it can still hit an OOM at a
    large batch size, and one that fails it will not be rescued by a small one.

    Args:
        gpu: Target GPU.
        params: Total parameter count, embeddings included.
        bytes_per_param: Bytes of persistent state per parameter.

    Returns:
        Whether it fits, and the gigabytes of persistent state required.
    """
    required_gb = params * bytes_per_param / 1024**3
    available_gb = (gpu.memory_mib / 1024) * (1 - ACTIVATION_HEADROOM)
    return required_gb <= available_gb, required_gb


def _run(host: Optional[str], command: str, timeout: int = 60) -> str:
    """Run *command* on *host* over SSH, or locally when *host* is None."""
    argv = ["bash", "-lc", command] if host is None else ["ssh", "-o", "BatchMode=yes", host, command]
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
    except (subprocess.TimeoutExpired, OSError) as error:
        return f"__ERROR__ {error}"
    return result.stdout.strip() if result.returncode == 0 else f"__ERROR__ {result.stderr.strip()[:200]}"


def _parse_gpus(raw: str) -> list[Gpu]:
    """Parse ``nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv,noheader``."""
    gpus: list[Gpu] = []
    for line in raw.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            gpus.append(
                Gpu(
                    index=int(parts[0]),
                    name=parts[1],
                    memory_mib=int(parts[2].split()[0]),
                    compute_capability=float(parts[3]),
                )
            )
        except ValueError:
            continue
    return gpus


def probe(host: Optional[str]) -> HostProbe:
    """Collect the capability of *host*, or of the local machine when *host* is None."""
    label = host or "localhost"
    result = HostProbe(host=label, hostname=_run(host, "hostname"))

    gpu_raw = _run(host, "nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv,noheader")
    if gpu_raw.startswith("__ERROR__"):
        result.errors.append(f"nvidia-smi unavailable: {gpu_raw[10:]}")
    else:
        result.gpus = _parse_gpus(gpu_raw)

    probes = {
        "driver": "nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1",
        "cuda_toolkit": "nvcc --version 2>/dev/null | grep -oP 'release \\K[0-9.]+' | head -1",
        "python_version": "python3 -V 2>&1 | awk '{print $2}'",
        "torch_version": "python3 -c 'import torch;print(torch.__version__)' 2>/dev/null",
        "torch_cuda": "python3 -c 'import torch;print(torch.version.cuda)' 2>/dev/null",
        "cpu_count": "nproc",
        "memory_gb": "free -g | awk 'NR==2{print $2}'",
        "disk_free": "df -h ~ | awk 'NR==2{print $4}'",
    }
    for attribute, command in probes.items():
        value = _run(host, command)
        setattr(result, attribute, "" if value.startswith("__ERROR__") else value)

    return result


def derive(result: HostProbe, gpu_index: Optional[int] = None) -> dict:
    """Turn a raw probe into the decisions that depend on it.

    Args:
        result: Output of :func:`probe`.
        gpu_index: Restrict the verdict to one GPU, as the v2 plan does for renard.

    Returns:
        Precision support, the selected GPUs, and the fit verdict per checkpoint.
    """
    selected = [g for g in result.gpus if gpu_index is None or g.index == gpu_index]
    fits = {
        name: {
            gpu.index: {"fits": trainable(gpu, params)[0], "state_gb": round(trainable(gpu, params)[1], 1)}
            for gpu in selected
        }
        for name, params in CHECKPOINT_PARAMS.items()
    }
    return {
        "selected_gpu_indices": [gpu.index for gpu in selected],
        "bf16": all(gpu.supports_bf16 for gpu in selected) if selected else False,
        "fp16_tensor_cores": all(gpu.has_fp16_tensor_cores for gpu in selected) if selected else False,
        "recommended_precision": "bf16" if selected and all(g.supports_bf16 for g in selected) else "fp32",
        "checkpoint_fit": fits,
    }


def render_markdown(result: HostProbe, verdict: dict) -> str:
    """Render the capability sheet that replaces the hand-written one."""
    lines = [
        f"# Serveur d'entrainement : {result.host}",
        "",
        "**Fichier genere.** Ne pas editer a la main. Regenerer avec :",
        "",
        "```",
        f"PYTHONPATH=src python src/diagnostics/probe_training_host.py --host {result.host} "
        f"--gpu {verdict['selected_gpu_indices'][0] if verdict['selected_gpu_indices'] else 0} \\",
        f"    --json-out results/host-{result.host}.json --markdown-out docs/serveur-releve-{result.host}.md",
        "```",
        "",
        "## Releve",
        "",
        "| | |",
        "|---|---|",
        f"| Hote | {result.hostname or result.host} |",
        f"| Pilote | {result.driver or 'inconnu'} |",
        f"| Toolkit CUDA | {result.cuda_toolkit or 'absent'} |",
        f"| Python | {result.python_version or 'inconnu'} |",
        f"| torch | {result.torch_version or 'absent'} |",
        f"| torch CUDA | {result.torch_cuda or 'n/a'} |",
        f"| CPU | {result.cpu_count or '?'} coeurs |",
        f"| RAM | {result.memory_gb or '?'} Go |",
        f"| Disque libre | {result.disk_free or '?'} |",
        "",
        "## GPU",
        "",
        "| Index | Nom | Memoire | Capacite | bf16 | Coeurs tensoriels fp16 |",
        "|---|---|---|---|---|---|",
    ]
    for gpu in result.gpus:
        selected = " **(retenue)**" if gpu.index in verdict["selected_gpu_indices"] else ""
        lines.append(
            f"| {gpu.index}{selected} | {gpu.name} | {gpu.memory_mib} MiB | {gpu.compute_capability} | "
            f"{'oui' if gpu.supports_bf16 else 'non'} | {'oui' if gpu.has_fp16_tensor_cores else 'non'} |"
        )

    indices = verdict["selected_gpu_indices"]
    lines += [
        "",
        "## Precision",
        "",
        f"- bf16 disponible sur les GPU retenues : **{'oui' if verdict['bf16'] else 'non'}**"
        f" (exige une capacite de calcul >= {BF16_MIN_COMPUTE_CAPABILITY}).",
        f"- Coeurs tensoriels fp16 : **{'oui' if verdict['fp16_tensor_cores'] else 'non'}**.",
        f"- Precision retenue par defaut : **{verdict['recommended_precision']}**.",
        "",
    ]
    if not verdict["bf16"]:
        lines += [
            "> Les scripts de sweep existants passent `--bf16`. Ils echoueront ou tomberont",
            "> silencieusement en fp32 sur cette machine. A corriger avant tout entrainement.",
            "",
        ]
    if indices:
        lines += [
            f"## GPU a utiliser : index {', '.join(str(i) for i in indices)}",
            "",
            "```",
            f"CUDA_VISIBLE_DEVICES={','.join(str(i) for i in indices)}",
            "```",
            "",
        ]
    lines += [
        "## Ce qui rentre en fine-tuning",
        "",
        f"Estimation AdamW fp32, {BYTES_PER_PARAM_ADAMW_FP32} octets par parametre pour poids,",
        f"gradients et moments, avec {int(ACTIVATION_HEADROOM * 100)} % de la carte reservee aux",
        "activations et au contexte CUDA. C'est un tri, pas une garantie.",
        "",
        "| Checkpoint | Parametres | Etats | Verdict |",
        "|---|---|---|---|",
    ]
    for name, params in CHECKPOINT_PARAMS.items():
        per_gpu = verdict["checkpoint_fit"][name]
        if not per_gpu:
            continue
        state = next(iter(per_gpu.values()))["state_gb"]
        ok = all(entry["fits"] for entry in per_gpu.values())
        lines.append(f"| {name} | {params / 1e6:.0f} M | {state} Go | {'passe' if ok else '**ne passe pas**'} |")
    lines += [
        "",
        "## Rappel",
        "",
        "Le backbone n'est pas le goulot de la v2. `docs/H1-diagnostic-calibration.md` etablit",
        "que les quatre checkpoints du sweep convergent vers le meme plancher de RMSE apres",
        "recalibrage affine, 22,1 a 23,1, et que le Pearson est plat a 0,78-0,80. Perdre un gros",
        "checkpoint faute de memoire ne coute donc rien de scientifique.",
    ]
    if result.errors:
        lines += ["", "## Erreurs de sonde", ""] + [f"- {error}" for error in result.errors]
    return "\n".join(lines) + "\n"


def main() -> None:
    """Probe a training host and write the capability sheet."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default=None, help="SSH host. Omit to probe the local machine.")
    parser.add_argument("--gpu", type=int, default=None, help="Restrict the verdict to this GPU index.")
    parser.add_argument("--json-out", default=None, help="Where to write the raw probe.")
    parser.add_argument("--markdown-out", default=None, help="Where to write the capability sheet.")
    args = parser.parse_args()

    result = probe(args.host)
    verdict = derive(result, args.gpu)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump({"probe": asdict(result), "verdict": verdict}, handle, indent=2)
        print(f"probe written to {args.json_out}")

    markdown = render_markdown(result, verdict)
    if args.markdown_out:
        with open(args.markdown_out, "w", encoding="utf-8") as handle:
            handle.write(markdown)
        print(f"capability sheet written to {args.markdown_out}")
    else:
        print(markdown)

    for error in result.errors:
        print(f"warning: {error}")


if __name__ == "__main__":
    main()
