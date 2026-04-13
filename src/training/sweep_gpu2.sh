#!/bin/bash
# GPU 2 - 4 models x 10 folds x 2 augmentations = 80 runs
# RTX 6000 Ada 49GB - bf16, large models with layer freezing
export CUDA_VISIBLE_DEVICES=2
export WANDB_PROJECT="meaningbert-checkpoint-sweep"

FOLDS=(0 1 2 3 4 5 6 7 8 9)
SEEDS=(42 43 44 45 46 47 48 49 50 51)
AUGMENTATIONS=("swap" "back_translation")
COMMON="--data_dir=./data --early_stopping_patience=50 --bf16 --dataloader_num_workers=4"

TOTAL=80
RUN=0

# checkpoint -> "learning_rate freeze_layers batch_size"
declare -A MODELS
MODELS=(
    ["microsoft/deberta-v2-xlarge"]="1e-5 12 32"
    ["Qwen/Qwen2.5-1.5B"]="1e-5 8 32"
    ["google/gemma-2-2b"]="1e-5 10 4"
    ["microsoft/phi-2"]="1e-5 10 4"
)

for checkpoint in "${!MODELS[@]}"; do
    read -r lr freeze bs <<< "${MODELS[$checkpoint]}"
    for aug in "${AUGMENTATIONS[@]}"; do
        for i in "${!FOLDS[@]}"; do
            fold=${FOLDS[$i]}
            seed=${SEEDS[$i]}
            RUN=$((RUN + 1))
            echo "=== [GPU 2] [${RUN}/${TOTAL}] ${checkpoint} fold=${fold} aug=${aug} ==="
            python few_shot_training.py --seed="${seed}" --fold="${fold}" \
                --checkpoint="${checkpoint}" --learning_rate="${lr}" \
                --freeze_layers="${freeze}" --per_device_train_batch_size="${bs}" \
                --data_augmentation="${aug}" ${COMMON}
        done
    done
done
echo "=== [GPU 2] Done: ${TOTAL}/${TOTAL} runs ==="
