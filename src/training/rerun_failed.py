"""Relaunch failed wandb runs by parsing their config and re-submitting.

Run with --gpu to target a specific GPU. Launch 3 instances in parallel:

    nohup python rerun_failed.py --gpu 0 > rerun_gpu0.log 2>&1 &
    nohup python rerun_failed.py --gpu 1 > rerun_gpu1.log 2>&1 &
    nohup python rerun_failed.py --gpu 2 > rerun_gpu2.log 2>&1 &
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

import wandb

# phi-2 failed in fp32 due to backward dtype mismatch -> use fp16
PHI2_CHECKPOINT = "microsoft/phi-2"

COMMON = ["--data_dir=./data", "--early_stopping_patience=50", "--dataloader_num_workers=4"]

# Checkpoint -> GPU assignment matching the original sweep layout.
# GPU 0: deberta-v3-small
# GPU 1: deberta-v3-base, deberta-v3-large
# GPU 2: deberta-v2-xlarge, phi-2
CHECKPOINT_GPU: dict[str, int] = {
    "microsoft/deberta-v3-small": 0,
    "microsoft/deberta-v3-base": 1,
    "microsoft/deberta-v3-large": 1,
    "microsoft/deberta-v2-xlarge": 2,
    "microsoft/phi-2": 2,
}


def gpu_for_checkpoint(checkpoint: str) -> int:
    for key, gpu in CHECKPOINT_GPU.items():
        if key in checkpoint:
            return gpu
    return 0


def build_command(run: wandb.apis.public.Run) -> list[str]:
    cfg = run.config
    checkpoint: str = cfg.get("checkpoint") or _parse_checkpoint_from_name(run.name)
    fold: int = cfg.get("fold") if cfg.get("fold") is not None else _parse_fold_from_name(run.name)
    aug: str = cfg.get("data_augmentation") or _parse_aug_from_name(run.name)
    seed: int = cfg.get("seed", 42)
    lr: float = cfg.get("learning_rate", 1e-5)
    freeze: int = cfg.get("freeze_layers", 0)
    bs: int = cfg.get("per_device_train_batch_size", 32)

    return [
        sys.executable,
        "few_shot_training.py",
        f"--checkpoint={checkpoint}",
        f"--fold={fold}",
        f"--seed={seed}",
        f"--data_augmentation={aug}",
        f"--learning_rate={lr}",
        f"--freeze_layers={freeze}",
        f"--per_device_train_batch_size={bs}",
        *_precision_flags(checkpoint),
        *COMMON,
    ]


def _precision_flags(checkpoint: str) -> list[str]:
    if PHI2_CHECKPOINT in checkpoint:
        return ["--no-bf16", "--fp16"]
    if "gemma" in checkpoint.lower():
        return ["--no-bf16"]
    return ["--bf16"]


def _parse_checkpoint_from_name(name: str) -> str:
    base = name.split("_seed")[0]
    parts = base.split("_", 1)
    return "/".join(parts) if len(parts) == 2 else base


def _parse_fold_from_name(name: str) -> int:
    for part in name.split("_"):
        if part.startswith("fold"):
            return int(part[4:])
    raise ValueError(f"Cannot parse fold from run name: {name}")


def _parse_aug_from_name(name: str) -> str:
    return "back_translation" if "back_translation" in name else "swap"


def main() -> None:
    parser = argparse.ArgumentParser(description="Relaunch failed wandb runs on a specific GPU.")
    parser.add_argument("--project", default="davebulaval/meaningbert-checkpoint-sweep")
    parser.add_argument(
        "--gpu",
        type=int,
        choices=[0, 1, 2],
        required=True,
        help="GPU index to use (sets CUDA_VISIBLE_DEVICES). Run 3 instances in parallel.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    api = wandb.Api()
    all_failed = list(api.runs(args.project, filters={"state": "failed"}))

    # Keep only runs assigned to this GPU
    my_runs = []
    for run in all_failed:
        checkpoint = run.config.get("checkpoint") or _parse_checkpoint_from_name(run.name)
        if gpu_for_checkpoint(checkpoint) == args.gpu:
            my_runs.append(run)

    total = len(all_failed)
    mine = len(my_runs)
    print(f"GPU {args.gpu}: {mine} runs to relaunch (out of {total} total failed)")

    for i, run in enumerate(my_runs, 1):
        cmd = build_command(run)
        checkpoint = run.config.get("checkpoint") or _parse_checkpoint_from_name(run.name)
        print(f"\n=== [GPU {args.gpu}] [{i}/{mine}] {run.name} ===")
        print("  " + " ".join(cmd))
        if not args.dry_run:
            result = subprocess.run(cmd, check=False)
            if result.returncode != 0:
                print(f"  ERROR: exit code {result.returncode}")

    print(f"\n[GPU {args.gpu}] Done: {mine} runs {'(dry-run)' if args.dry_run else 'completed'}.")


if __name__ == "__main__":
    main()
