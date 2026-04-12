#!/bin/bash
# GPU 0 - 5 models x 10 seeds = 50 runs
# Reads pre-generated datasets from ./data (run prepare_datasets.py first)
export CUDA_VISIBLE_DEVICES=0
export WANDB_PROJECT="meaningbert-checkpoint-sweep"

SEEDS=(42 43 44 45 46 47 48 49 50 51)
COMMON="--data_dir=./data --data_augmentation=swap --gradient_accumulation_steps=2 --early_stopping_patience=50"

TOTAL=50
RUN=0

declare -A MODELS
MODELS=(
    ["bert-base-uncased"]="5e-5 0"
    ["microsoft/deberta-v3-small"]="2e-5 0"
    ["openai-community/gpt2"]="2e-5 0"
    ["HuggingFaceTB/SmolLM2-135M"]="2e-5 0"
    ["HuggingFaceTB/SmolLM2-360M"]="2e-5 0"
)

for checkpoint in "${!MODELS[@]}"; do
    read -r lr freeze <<< "${MODELS[$checkpoint]}"
    for seed in "${SEEDS[@]}"; do
        RUN=$((RUN + 1))
        echo "=== [GPU 0] [${RUN}/${TOTAL}] ${checkpoint} seed=${seed} lr=${lr} ==="
        python few_shot_training.py --seed="${seed}" --checkpoint="${checkpoint}" \
            --learning_rate="${lr}" --freeze_layers="${freeze}" ${COMMON}
    done
done
echo "=== [GPU 0] Done: ${TOTAL}/${TOTAL} runs ==="
