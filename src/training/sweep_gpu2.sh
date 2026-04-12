#!/bin/bash
# GPU 2 - 4 models x 10 seeds = 40 runs
# Large models with layer freezing
# Reads pre-generated datasets from ./data (run prepare_datasets.py first)
export CUDA_VISIBLE_DEVICES=2
export WANDB_PROJECT="meaningbert-checkpoint-sweep"

SEEDS=(42 43 44 45 46 47 48 49 50 51)
COMMON="--data_dir=./data --data_augmentation=swap --gradient_accumulation_steps=2 --early_stopping_patience=50"

# deberta-v2-xlarge (900M, encoder, freeze 12 layers)
for seed in "${SEEDS[@]}"; do
    echo "=== [GPU 2] deberta-v2-xlarge seed=${seed} ==="
    python few_shot_training.py --seed="${seed}" --checkpoint="microsoft/deberta-v2-xlarge" --learning_rate=1e-5 --freeze_layers=12 ${COMMON}
done

# Qwen2.5-1.5B (1.5B, decoder, freeze 8 layers)
for seed in "${SEEDS[@]}"; do
    echo "=== [GPU 2] Qwen2.5-1.5B seed=${seed} ==="
    python few_shot_training.py --seed="${seed}" --checkpoint="Qwen/Qwen2.5-1.5B" --learning_rate=1e-5 --freeze_layers=8 ${COMMON}
done

# gemma-2-2b (2.6B, decoder, freeze 10 layers)
for seed in "${SEEDS[@]}"; do
    echo "=== [GPU 2] gemma-2-2b seed=${seed} ==="
    python few_shot_training.py --seed="${seed}" --checkpoint="google/gemma-2-2b" --learning_rate=1e-5 --freeze_layers=10 ${COMMON}
done

# phi-2 (2.7B, decoder, freeze 10 layers)
for seed in "${SEEDS[@]}"; do
    echo "=== [GPU 2] phi-2 seed=${seed} ==="
    python few_shot_training.py --seed="${seed}" --checkpoint="microsoft/phi-2" --learning_rate=1e-5 --freeze_layers=10 ${COMMON}
done
