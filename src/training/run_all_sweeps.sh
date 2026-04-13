#!/bin/bash
# Run all MeaningBERT sweeps sequentially
set -e

cd "$(dirname "$0")"

echo "=== MeaningBERT Full Sweep ==="
echo "Seeds 42-51 | With and without data augmentation"
echo ""

# With data augmentation (seeds 42-51)
echo "--- With data augmentation ---"
for seed in 42 43 44 45 46 47 48 49 50 51; do
    echo "[$(date '+%H:%M:%S')] Running seed=$seed with data_augmentation=true"
    python few_shot_training.py --seed=$seed --data_augmentation="true"
    echo "[$(date '+%H:%M:%S')] Seed $seed (augmented) done"
    echo ""
done

# Without data augmentation (seeds 42-51)
echo "--- Without data augmentation ---"
for seed in 42 43 44 45 46 47 48 49 50 51; do
    echo "[$(date '+%H:%M:%S')] Running seed=$seed without data_augmentation"
    python few_shot_training.py --seed=$seed
    echo "[$(date '+%H:%M:%S')] Seed $seed (no augmentation) done"
    echo ""
done

echo "=== All sweeps complete ==="
echo "Run figures_generator/generates_results_tables.py to generate LaTeX tables."
