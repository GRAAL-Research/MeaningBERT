#!/bin/bash
# GPU 1 - 5 models x 10 folds x 2 augmentations = 100 runs
# Reads pre-generated datasets from ./data (run prepare_datasets.py first)
export CUDA_VISIBLE_DEVICES=1
export WANDB_PROJECT="meaningbert-checkpoint-sweep"

FOLDS=(0 1 2 3 4 5 6 7 8 9)
SEEDS=(42 43 44 45 46 47 48 49 50 51)
AUGMENTATIONS=("swap" "back_translation")
COMMON="--data_dir=./data --gradient_accumulation_steps=2 --early_stopping_patience=50"

TOTAL=100
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
    for aug in "${AUGMENTATIONS[@]}"; do
        for i in "${!FOLDS[@]}"; do
            fold=${FOLDS[$i]}
            seed=${SEEDS[$i]}
            RUN=$((RUN + 1))
            echo "=== [GPU 1] [${RUN}/${TOTAL}] ${checkpoint} fold=${fold} aug=${aug} ==="
            python few_shot_training.py --seed="${seed}" --fold="${fold}" \
                --checkpoint="${checkpoint}" --learning_rate="${lr}" \
                --freeze_layers="${freeze}" --data_augmentation="${aug}" ${COMMON}
        done
    done
done
echo "=== [GPU 1] Done: ${TOTAL}/${TOTAL} runs ==="
