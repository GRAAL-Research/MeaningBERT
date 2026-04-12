#!/bin/bash
# GPU 1 - 5 models x 10 seeds = 50 runs
# Reads pre-generated datasets from ./data (run prepare_datasets.py first)
export CUDA_VISIBLE_DEVICES=1
export WANDB_PROJECT="meaningbert-checkpoint-sweep"

SEEDS=(42 43 44 45 46 47 48 49 50 51)
COMMON="--data_dir=./data --data_augmentation=swap --gradient_accumulation_steps=2 --early_stopping_patience=50"

# deberta-v3-base (86M, encoder)
for seed in "${SEEDS[@]}"; do
    echo "=== [GPU 1] deberta-v3-base seed=${seed} ==="
    python few_shot_training.py --seed="${seed}" --checkpoint="microsoft/deberta-v3-base" --learning_rate=2e-5 ${COMMON}
done

# deberta-v3-large (304M, encoder, freeze 6 layers)
for seed in "${SEEDS[@]}"; do
    echo "=== [GPU 1] deberta-v3-large seed=${seed} ==="
    python few_shot_training.py --seed="${seed}" --checkpoint="microsoft/deberta-v3-large" --learning_rate=2e-5 --freeze_layers=6 ${COMMON}
done

# electra-large (335M, encoder, freeze 6 layers)
for seed in "${SEEDS[@]}"; do
    echo "=== [GPU 1] electra-large seed=${seed} ==="
    python few_shot_training.py --seed="${seed}" --checkpoint="google/electra-large-discriminator" --learning_rate=3e-5 --freeze_layers=6 ${COMMON}
done

# ModernBERT-large (395M, encoder, freeze 6 layers)
for seed in "${SEEDS[@]}"; do
    echo "=== [GPU 1] ModernBERT-large seed=${seed} ==="
    python few_shot_training.py --seed="${seed}" --checkpoint="answerdotai/ModernBERT-large" --learning_rate=5e-5 --freeze_layers=6 ${COMMON}
done

# Qwen2.5-0.5B (500M, decoder)
for seed in "${SEEDS[@]}"; do
    echo "=== [GPU 1] Qwen2.5-0.5B seed=${seed} ==="
    python few_shot_training.py --seed="${seed}" --checkpoint="Qwen/Qwen2.5-0.5B" --learning_rate=1e-5 ${COMMON}
done
