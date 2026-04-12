#!/bin/bash
# GPU 0 - 5 models x 10 seeds = 50 runs
# Reads pre-generated datasets from ./data (run prepare_datasets.py first)
export CUDA_VISIBLE_DEVICES=0
export WANDB_PROJECT="meaningbert-checkpoint-sweep"

SEEDS=(42 43 44 45 46 47 48 49 50 51)
COMMON="--data_dir=./data --data_augmentation=swap --gradient_accumulation_steps=2 --early_stopping_patience=50"

# bert-base-uncased (110M, encoder)
for seed in "${SEEDS[@]}"; do
    echo "=== [GPU 0] bert-base-uncased seed=${seed} ==="
    python few_shot_training.py --seed="${seed}" --checkpoint="bert-base-uncased" --learning_rate=5e-5 ${COMMON}
done

# deberta-v3-small (44M, encoder)
for seed in "${SEEDS[@]}"; do
    echo "=== [GPU 0] deberta-v3-small seed=${seed} ==="
    python few_shot_training.py --seed="${seed}" --checkpoint="microsoft/deberta-v3-small" --learning_rate=2e-5 ${COMMON}
done

# gpt2 (124M, decoder)
for seed in "${SEEDS[@]}"; do
    echo "=== [GPU 0] gpt2 seed=${seed} ==="
    python few_shot_training.py --seed="${seed}" --checkpoint="openai-community/gpt2" --learning_rate=2e-5 ${COMMON}
done

# SmolLM2-135M (135M, decoder)
for seed in "${SEEDS[@]}"; do
    echo "=== [GPU 0] SmolLM2-135M seed=${seed} ==="
    python few_shot_training.py --seed="${seed}" --checkpoint="HuggingFaceTB/SmolLM2-135M" --learning_rate=2e-5 ${COMMON}
done

# SmolLM2-360M (360M, decoder)
for seed in "${SEEDS[@]}"; do
    echo "=== [GPU 0] SmolLM2-360M seed=${seed} ==="
    python few_shot_training.py --seed="${seed}" --checkpoint="HuggingFaceTB/SmolLM2-360M" --learning_rate=2e-5 ${COMMON}
done
