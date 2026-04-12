#!/bin/bash
# GPU 1 - 5 models x 10 seeds = 50 runs
# Reads pre-generated datasets from ./data (run prepare_datasets.py first)
export CUDA_VISIBLE_DEVICES=1
export WANDB_PROJECT="meaningbert-checkpoint-sweep"

SEEDS=(42 43 44 45 46 47 48 49 50 51)
COMMON="--data_dir=./data --data_augmentation=swap --gradient_accumulation_steps=2 --early_stopping_patience=50"

TOTAL=50
RUN=0

declare -A MODELS
MODELS=(
    ["microsoft/deberta-v3-base"]="2e-5 0"
    ["microsoft/deberta-v3-large"]="2e-5 6"
    ["google/electra-large-discriminator"]="3e-5 6"
    ["answerdotai/ModernBERT-large"]="5e-5 6"
    ["Qwen/Qwen2.5-0.5B"]="1e-5 0"
)

for checkpoint in "${!MODELS[@]}"; do
    read -r lr freeze <<< "${MODELS[$checkpoint]}"
    for seed in "${SEEDS[@]}"; do
        RUN=$((RUN + 1))
        echo "=== [GPU 1] [${RUN}/${TOTAL}] ${checkpoint} seed=${seed} lr=${lr} ==="
        python few_shot_training.py --seed="${seed}" --checkpoint="${checkpoint}" \
            --learning_rate="${lr}" --freeze_layers="${freeze}" ${COMMON}
    done
done
echo "=== [GPU 1] Done: ${TOTAL}/${TOTAL} runs ==="
