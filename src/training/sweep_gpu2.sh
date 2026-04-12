#!/bin/bash
# GPU 2 - 4 models x 10 folds x 2 augmentations = 80 runs
# Large models with layer freezing
# Reads pre-generated datasets from ./data (run prepare_datasets.py first)
export CUDA_VISIBLE_DEVICES=2
export WANDB_PROJECT="meaningbert-checkpoint-sweep"

FOLDS=(0 1 2 3 4 5 6 7 8 9)
SEEDS=(42 43 44 45 46 47 48 49 50 51)
AUGMENTATIONS=("swap" "back_translation")
COMMON="--data_dir=./data --gradient_accumulation_steps=2 --early_stopping_patience=50"

TOTAL=80
RUN=0

declare -A MODELS
MODELS=(
    ["microsoft/deberta-v2-xlarge"]="1e-5 12"
    ["Qwen/Qwen2.5-1.5B"]="1e-5 8"
    ["google/gemma-2-2b"]="1e-5 10"
    ["microsoft/phi-2"]="1e-5 10"
)

for checkpoint in "${!MODELS[@]}"; do
    read -r lr freeze <<< "${MODELS[$checkpoint]}"
    for aug in "${AUGMENTATIONS[@]}"; do
        for i in "${!FOLDS[@]}"; do
            fold=${FOLDS[$i]}
            seed=${SEEDS[$i]}
            RUN=$((RUN + 1))
            echo "=== [GPU 2] [${RUN}/${TOTAL}] ${checkpoint} fold=${fold} aug=${aug} ==="
            python few_shot_training.py --seed="${seed}" --fold="${fold}" \
                --checkpoint="${checkpoint}" --learning_rate="${lr}" \
                --freeze_layers="${freeze}" --data_augmentation="${aug}" ${COMMON}
        done
    done
done
echo "=== [GPU 2] Done: ${TOTAL}/${TOTAL} runs ==="
