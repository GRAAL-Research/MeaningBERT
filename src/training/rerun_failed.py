"""Relaunch failed wandb runs.

Two modes:
  --hardcoded   Use the predefined list of 100 known-failed runs (use when
                wandb runs were already deleted).
  (default)     Query wandb for currently failed runs.

Launch 3 instances in parallel:

    nohup python rerun_failed.py --gpu 0 --hardcoded > rerun_gpu0.log 2>&1 &
    nohup python rerun_failed.py --gpu 1 --hardcoded > rerun_gpu1.log 2>&1 &
    nohup python rerun_failed.py --gpu 2 --hardcoded > rerun_gpu2.log 2>&1 &
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

import wandb

PHI2_CHECKPOINT = "microsoft/phi-2"

COMMON = ["--data_dir=./data", "--early_stopping_patience=50", "--dataloader_num_workers=4"]

FOLDS = list(range(10))
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
AUGMENTATIONS = ["swap", "back_translation"]

# Checkpoint -> GPU assignment matching the original sweep layout.
CHECKPOINT_GPU: dict[str, int] = {
    "microsoft/deberta-v3-small": 0,
    "microsoft/deberta-v3-base": 1,
    "microsoft/deberta-v3-large": 1,
    "microsoft/deberta-v2-xlarge": 2,
    "microsoft/phi-2": 2,
}

# Full config for each failed checkpoint family.
# (checkpoint, lr, freeze_layers, batch_size)
HARDCODED_CONFIGS: list[tuple[str, str, int, int]] = [
    ("microsoft/deberta-v3-small", "2e-5", 0, 128),
    ("microsoft/deberta-v3-base", "2e-5", 0, 128),
    ("microsoft/deberta-v3-large", "2e-5", 6, 64),
    ("microsoft/deberta-v2-xlarge", "1e-5", 12, 32),
    ("microsoft/phi-2", "1e-5", 10, 4),
]


def gpu_for_checkpoint(checkpoint: str) -> int:
    for key, gpu in CHECKPOINT_GPU.items():
        if key in checkpoint:
            return gpu
    return 0


def _precision_flags(checkpoint: str) -> list[str]:
    if PHI2_CHECKPOINT in checkpoint:
        return ["--no-bf16", "--fp16"]
    if "gemma" in checkpoint.lower():
        return ["--no-bf16"]
    return ["--bf16"]


def build_command_from_config(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    checkpoint: str,
    fold: int,
    seed: int,
    aug: str,
    lr: str,
    freeze: int,
    bs: int,
) -> list[str]:
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


def build_command_from_run(run: wandb.apis.public.Run) -> list[str]:
    cfg = run.config
    checkpoint: str = cfg.get("checkpoint") or _parse_checkpoint_from_name(run.name)
    fold: int = cfg.get("fold") if cfg.get("fold") is not None else _parse_fold_from_name(run.name)
    aug: str = cfg.get("data_augmentation") or _parse_aug_from_name(run.name)
    seed: int = cfg.get("seed", 42)
    lr: str = str(cfg.get("learning_rate", 1e-5))
    freeze: int = cfg.get("freeze_layers", 0)
    bs: int = cfg.get("per_device_train_batch_size", 32)
    return build_command_from_config(checkpoint, fold, seed, aug, lr, freeze, bs)


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


def get_hardcoded_runs(gpu: int, finished_names: set[str]) -> list[tuple[list[str], str]]:
    """Return (command, label) pairs for the given GPU, skipping already-finished runs."""
    runs = []
    for checkpoint, lr, freeze, bs in HARDCODED_CONFIGS:
        if gpu_for_checkpoint(checkpoint) != gpu:
            continue
        checkpoint_short = checkpoint.replace("/", "_")
        for aug in AUGMENTATIONS:
            for fold, seed in zip(FOLDS, SEEDS):
                # float(lr) ensures "2e-5" formats as "2e-05", matching few_shot_training.py run names.
                run_name = (
                    f"{checkpoint_short}_seed{seed}_lr{float(lr)}_bs{bs}"
                    f"_freeze{freeze}_aug{aug}_fold{fold}"
                )
                if run_name in finished_names:
                    print(f"  SKIP (already finished): {run_name}")
                    continue
                cmd = build_command_from_config(checkpoint, fold, seed, aug, lr, freeze, bs)
                label = f"{checkpoint} fold={fold} aug={aug}"
                runs.append((cmd, label))
    return runs


def get_finished_run_names(project: str) -> set[str]:
    """Return the set of run names that already finished successfully."""
    api = wandb.Api()
    return {run.name for run in api.runs(project, filters={"state": "finished"})}


def get_wandb_runs(gpu: int, project: str) -> list[tuple[list[str], str]]:
    """Return (command, label) pairs for failed/crashed runs assigned to this GPU."""
    api = wandb.Api()
    all_failed = [r for r in api.runs(project) if r.state in ("failed", "crashed")]
    runs = []
    for run in all_failed:
        checkpoint = run.config.get("checkpoint") or _parse_checkpoint_from_name(run.name)
        if gpu_for_checkpoint(checkpoint) == gpu:
            runs.append((build_command_from_run(run), run.name))
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description="Relaunch failed wandb runs on a specific GPU.")
    parser.add_argument("--project", default="davebulaval/meaningbert-checkpoint-sweep")
    parser.add_argument(
        "--gpu",
        type=int,
        choices=[0, 1, 2],
        required=True,
        help="GPU index (sets CUDA_VISIBLE_DEVICES). Run 3 instances in parallel.",
    )
    parser.add_argument(
        "--hardcoded",
        action="store_true",
        help="Use predefined run list instead of querying wandb (use when runs were deleted).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if args.hardcoded:
        finished = get_finished_run_names(args.project)
        print(f"GPU {args.gpu}: {len(finished)} runs already finished (will be skipped)")
        my_runs = get_hardcoded_runs(args.gpu, finished)
        print(f"GPU {args.gpu}: {len(my_runs)} runs to relaunch (hardcoded list)")
    else:
        my_runs = get_wandb_runs(args.gpu, args.project)
        print(f"GPU {args.gpu}: {len(my_runs)} runs to relaunch (from wandb failed/crashed)")

    for i, (cmd, label) in enumerate(my_runs, 1):
        print(f"\n=== [GPU {args.gpu}] [{i}/{len(my_runs)}] {label} ===")
        print("  " + " ".join(cmd))
        if not args.dry_run:
            result = subprocess.run(cmd, check=False)
            if result.returncode != 0:
                print(f"  ERROR: exit code {result.returncode}")

    print(f"\n[GPU {args.gpu}] Done: {len(my_runs)} runs {'(dry-run)' if args.dry_run else 'completed'}.")


if __name__ == "__main__":
    main()
