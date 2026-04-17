"""Relaunch failed wandb runs by parsing their config and re-submitting."""
from __future__ import annotations

import argparse
import subprocess
import sys

import wandb

# phi-2 failed in fp32 due to backward dtype mismatch -> use fp16
PHI2_CHECKPOINT = "microsoft/phi-2"

COMMON = ["--data_dir=./data", "--early_stopping_patience=50", "--dataloader_num_workers=4"]


def build_command(run: wandb.apis.public.Run) -> list[str]:
    cfg = run.config
    checkpoint: str = cfg.get("checkpoint") or _parse_checkpoint_from_name(run.name)
    fold: int = cfg.get("fold") if cfg.get("fold") is not None else _parse_fold_from_name(run.name)
    aug: str = cfg.get("data_augmentation") or _parse_aug_from_name(run.name)
    seed: int = cfg.get("seed", 42)
    lr: float = cfg.get("learning_rate", 1e-5)
    freeze: int = cfg.get("freeze_layers", 0)
    bs: int = cfg.get("per_device_train_batch_size", 32)

    precision_flags = _precision_flags(checkpoint)

    return [
        sys.executable, "few_shot_training.py",
        f"--checkpoint={checkpoint}",
        f"--fold={fold}",
        f"--seed={seed}",
        f"--data_augmentation={aug}",
        f"--learning_rate={lr}",
        f"--freeze_layers={freeze}",
        f"--per_device_train_batch_size={bs}",
        *precision_flags,
        *COMMON,
    ]


def _precision_flags(checkpoint: str) -> list[str]:
    if checkpoint == PHI2_CHECKPOINT:
        return ["--no-bf16", "--fp16"]
    if "gemma" in checkpoint.lower():
        return ["--no-bf16"]
    return ["--bf16"]


def _parse_checkpoint_from_name(name: str) -> str:
    # e.g. microsoft_deberta-v3-small_seed42_... -> microsoft/deberta-v3-small
    base = name.split("_seed")[0]
    # first underscore that separates org from model name
    parts = base.split("_", 1)
    return "/".join(parts) if len(parts) == 2 else base


def _parse_fold_from_name(name: str) -> int:
    for part in name.split("_"):
        if part.startswith("fold"):
            return int(part[4:])
    raise ValueError(f"Cannot parse fold from run name: {name}")


def _parse_aug_from_name(name: str) -> str:
    if "back_translation" in name:
        return "back_translation"
    return "swap"


def main() -> None:
    parser = argparse.ArgumentParser(description="Relaunch failed wandb runs.")
    parser.add_argument("--project", default="davebulaval/meaningbert-checkpoint-sweep")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    parser.add_argument("--filter-checkpoint", default=None, help="Only relaunch runs matching this substring.")
    args = parser.parse_args()

    api = wandb.Api()
    failed_runs = list(api.runs(args.project, filters={"state": "failed"}))
    print(f"Found {len(failed_runs)} failed runs in {args.project}")

    if args.filter_checkpoint:
        failed_runs = [r for r in failed_runs if args.filter_checkpoint in r.name]
        print(f"Filtered to {len(failed_runs)} runs matching '{args.filter_checkpoint}'")

    for i, run in enumerate(failed_runs, 1):
        cmd = build_command(run)
        label = f"[{i}/{len(failed_runs)}] {run.name}"
        print(f"\n=== {label} ===")
        print("  " + " ".join(cmd))
        if not args.dry_run:
            result = subprocess.run(cmd)
            if result.returncode != 0:
                print(f"  ERROR: run failed with exit code {result.returncode}")

    print(f"\nDone: {len(failed_runs)} runs {'(dry-run)' if args.dry_run else 'completed'}.")


if __name__ == "__main__":
    main()
