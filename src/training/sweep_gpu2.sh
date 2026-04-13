#!/bin/bash
# GPU 2 - 4 models x 10 folds x 2 augmentations = 80 runs
# RTX 6000 Ada 49GB - bf16, large models with layer freezing
export CUDA_VISIBLE_DEVICES=2
export WANDB_PROJECT="meaningbert-checkpoint-sweep"

FOLDS=(0 1 2 3 4 5 6 7 8 9)
SEEDS=(42 43 44 45 46 47 48 49 50 51)
AUGMENTATIONS=("swap" "back_translation")
TOTAL=80
RUN=0

# Models stable in bf16
declare -A MODELS_BF16
MODELS_BF16=(
    ["microsoft/deberta-v2-xlarge"]="1e-5 12 32"
    ["Qwen/Qwen2.5-1.5B"]="1e-5 8 32"
)

# Models that need fp32 (bf16 causes NaN gradients)
declare -A MODELS_FP32
MODELS_FP32=(
    ["google/gemma-2-2b"]="1e-5 10 4"
    ["microsoft/phi-2"]="1e-5 10 4"
)

COMMON_BF16="--data_dir=./data --early_stopping_patience=50 --bf16 --dataloader_num_workers=4"
COMMON_FP32="--data_dir=./data --early_stopping_patience=50 --no-bf16 --dataloader_num_workers=4"

for checkpoint in "${!MODELS_BF16[@]}"; do
    read -r lr freeze bs <<< "${MODELS_BF16[$checkpoint]}"
    for aug in "${AUGMENTATIONS[@]}"; do
        for i in "${!FOLDS[@]}"; do
            fold=${FOLDS[$i]}
            seed=${SEEDS[$i]}
            RUN=$((RUN + 1))
            echo "=== [GPU 2] [${RUN}/${TOTAL}] ${checkpoint} fold=${fold} aug=${aug} ==="
            python few_shot_training.py --seed="${seed}" --fold="${fold}" \
                --checkpoint="${checkpoint}" --learning_rate="${lr}" \
                --freeze_layers="${freeze}" --per_device_train_batch_size="${bs}" \
                --data_augmentation="${aug}" ${COMMON_BF16}
        done
    done
done

for checkpoint in "${!MODELS_FP32[@]}"; do
    read -r lr freeze bs <<< "${MODELS_FP32[$checkpoint]}"
    for aug in "${AUGMENTATIONS[@]}"; do
        for i in "${!FOLDS[@]}"; do
            fold=${FOLDS[$i]}
            seed=${SEEDS[$i]}
            RUN=$((RUN + 1))
            echo "=== [GPU 2] [${RUN}/${TOTAL}] ${checkpoint} fold=${fold} aug=${aug} (fp32) ==="
            python few_shot_training.py --seed="${seed}" --fold="${fold}" \
                --checkpoint="${checkpoint}" --learning_rate="${lr}" \
                --freeze_layers="${freeze}" --per_device_train_batch_size="${bs}" \
                --data_augmentation="${aug}" ${COMMON_FP32}
        done
    done
done
echo "=== [GPU 2] Done: ${TOTAL}/${TOTAL} runs ==="
