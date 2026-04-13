"""Clean up failed wandb runs and intermediate training checkpoints.

Usage::

    # Dry run (show what would be deleted)
    python cleanup.py

    # Actually delete
    python cleanup.py --delete

    # Also clean failed/crashed wandb runs
    python cleanup.py --delete --clean-wandb
"""
from __future__ import annotations

import argparse
import os
import shutil
from glob import glob
from pathlib import Path


TRAINING_DIR = Path(__file__).parent


def find_intermediate_checkpoints() -> list[Path]:
    """Find checkpoint-* directories inside training output dirs."""
    results: list[Path] = []
    for output_dir in sorted(TRAINING_DIR.glob("meaning_bert_train_*")):
        if not output_dir.is_dir():
            continue
        for checkpoint_dir in sorted(output_dir.glob("checkpoint-*")):
            if checkpoint_dir.is_dir():
                results.append(checkpoint_dir)
    return results


def find_wandb_failed_runs() -> list[Path]:
    """Find wandb run directories that crashed (no summary.json or short duration)."""
    wandb_dir = TRAINING_DIR / "wandb"
    if not wandb_dir.exists():
        return []

    results: list[Path] = []
    for run_dir in sorted(wandb_dir.glob("run-*")):
        if not run_dir.is_dir():
            continue
        # A run without a wandb-summary.json likely crashed
        summary = run_dir / "files" / "wandb-summary.json"
        if not summary.exists():
            results.append(run_dir)
    return results


def get_dir_size(path: Path) -> int:
    """Get total size of a directory in bytes."""
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total


def format_size(size_bytes: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def clean_wandb_failed(delete: bool) -> None:
    """Delete failed wandb runs via the API."""
    try:
        import wandb
    except ImportError:
        print("  wandb not installed, skipping API cleanup")
        return

    api = wandb.Api()
    project = "davebulaval/meaningbert-checkpoint-sweep"
    try:
        runs = api.runs(project, filters={"state": "crashed"})
    except Exception as e:
        print(f"  Could not fetch wandb runs: {e}")
        return

    crashed = list(runs)
    if not crashed:
        print("  No crashed wandb runs found")
        return

    for run in crashed:
        if delete:
            run.delete()
            print(f"  Deleted wandb run: {run.name} ({run.id})")
        else:
            print(f"  Would delete wandb run: {run.name} ({run.id})")
    print(f"  Total: {len(crashed)} crashed runs")


def main() -> None:
    """Find and optionally delete intermediate checkpoints and failed runs."""
    parser = argparse.ArgumentParser(description="Clean up training artifacts.")
    parser.add_argument("--delete", action="store_true", help="Actually delete files (default: dry run).")
    parser.add_argument("--clean-wandb", action="store_true", help="Also delete crashed wandb runs via API.")
    args = parser.parse_args()

    # 1. Intermediate checkpoints
    print("=== Intermediate checkpoints ===")
    checkpoints = find_intermediate_checkpoints()
    total_size = 0
    for cp in checkpoints:
        size = get_dir_size(cp)
        total_size += size
        if args.delete:
            shutil.rmtree(cp)
            print(f"  Deleted: {cp} ({format_size(size)})")
        else:
            print(f"  Would delete: {cp} ({format_size(size)})")
    print(f"  Total: {len(checkpoints)} checkpoints, {format_size(total_size)}")

    # 2. Local wandb failed run directories
    print("\n=== Failed local wandb runs ===")
    failed_runs = find_wandb_failed_runs()
    total_size = 0
    for run_dir in failed_runs:
        size = get_dir_size(run_dir)
        total_size += size
        if args.delete:
            shutil.rmtree(run_dir)
            print(f"  Deleted: {run_dir.name} ({format_size(size)})")
        else:
            print(f"  Would delete: {run_dir.name} ({format_size(size)})")
    print(f"  Total: {len(failed_runs)} failed runs, {format_size(total_size)}")

    # 3. Remote wandb crashed runs
    if args.clean_wandb:
        print("\n=== Crashed wandb runs (remote) ===")
        clean_wandb_failed(args.delete)

    if not args.delete:
        print("\nDry run. Use --delete to actually remove files.")


if __name__ == "__main__":
    main()
