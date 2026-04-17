#!/bin/bash
# Rerun 100 failed runs:
#   - 80 deberta (v2-xlarge, v3-small/base/large): nvrtc fix applied 2026-04-14
#   - 20 phi-2: now uses fp16 instead of fp32 (backward dtype mismatch fix)
# Run on caribou after libnvrtc-builtins.so.13.0 is confirmed installed.
export WANDB_PROJECT="meaningbert-checkpoint-sweep"

FOLDS=(0 1 2 3 4 5 6 7 8 9)
SEEDS=(42 43 44 45 46 47 48 49 50 51)
AUGMENTATIONS=("swap" "back_translation")

TOTAL=100
RUN=0

# --- GPU 2: deberta-v2-xlarge (bf16, same as original) ---
export CUDA_VISIBLE_DEVICES=2
COMMON_BF16="--data_dir=./data --early_stopping_patience=50 --bf16 --dataloader_num_workers=4"

for aug in "${AUGMENTATIONS[@]}"; do
    for i in "${!FOLDS[@]}"; do
        fold=${FOLDS[$i]}; seed=${SEEDS[$i]}; RUN=$((RUN + 1))
        echo "=== [GPU 2] [${RUN}/${TOTAL}] deberta-v2-xlarge fold=${fold} aug=${aug} ==="
        python few_shot_training.py --seed="${seed}" --fold="${fold}" \
            --checkpoint="microsoft/deberta-v2-xlarge" --learning_rate="1e-5" \
            --freeze_layers=12 --per_device_train_batch_size=32 \
            --data_augmentation="${aug}" ${COMMON_BF16}
    done
done

# phi-2: fp16 (bf16 -> NaN, fp32 -> backward dtype mismatch)
COMMON_FP16="--data_dir=./data --early_stopping_patience=50 --no-bf16 --fp16 --dataloader_num_workers=4"

for aug in "${AUGMENTATIONS[@]}"; do
    for i in "${!FOLDS[@]}"; do
        fold=${FOLDS[$i]}; seed=${SEEDS[$i]}; RUN=$((RUN + 1))
        echo "=== [GPU 2] [${RUN}/${TOTAL}] phi-2 fold=${fold} aug=${aug} (fp16) ==="
        python few_shot_training.py --seed="${seed}" --fold="${fold}" \
            --checkpoint="microsoft/phi-2" --learning_rate="1e-5" \
            --freeze_layers=10 --per_device_train_batch_size=4 \
            --data_augmentation="${aug}" ${COMMON_FP16}
    done
done

# --- GPU 1: deberta-v3-base + deberta-v3-large ---
export CUDA_VISIBLE_DEVICES=1

for aug in "${AUGMENTATIONS[@]}"; do
    for i in "${!FOLDS[@]}"; do
        fold=${FOLDS[$i]}; seed=${SEEDS[$i]}; RUN=$((RUN + 1))
        echo "=== [GPU 1] [${RUN}/${TOTAL}] deberta-v3-base fold=${fold} aug=${aug} ==="
        python few_shot_training.py --seed="${seed}" --fold="${fold}" \
            --checkpoint="microsoft/deberta-v3-base" --learning_rate="2e-5" \
            --freeze_layers=0 --per_device_train_batch_size=128 \
            --data_augmentation="${aug}" ${COMMON_BF16}
    done
done

for aug in "${AUGMENTATIONS[@]}"; do
    for i in "${!FOLDS[@]}"; do
        fold=${FOLDS[$i]}; seed=${SEEDS[$i]}; RUN=$((RUN + 1))
        echo "=== [GPU 1] [${RUN}/${TOTAL}] deberta-v3-large fold=${fold} aug=${aug} ==="
        python few_shot_training.py --seed="${seed}" --fold="${fold}" \
            --checkpoint="microsoft/deberta-v3-large" --learning_rate="2e-5" \
            --freeze_layers=6 --per_device_train_batch_size=64 \
            --data_augmentation="${aug}" ${COMMON_BF16}
    done
done

# --- GPU 0: deberta-v3-small ---
export CUDA_VISIBLE_DEVICES=0

for aug in "${AUGMENTATIONS[@]}"; do
    for i in "${!FOLDS[@]}"; do
        fold=${FOLDS[$i]}; seed=${SEEDS[$i]}; RUN=$((RUN + 1))
        echo "=== [GPU 0] [${RUN}/${TOTAL}] deberta-v3-small fold=${fold} aug=${aug} ==="
        python few_shot_training.py --seed="${seed}" --fold="${fold}" \
            --checkpoint="microsoft/deberta-v3-small" --learning_rate="2e-5" \
            --freeze_layers=0 --per_device_train_batch_size=128 \
            --data_augmentation="${aug}" ${COMMON_BF16}
    done
done

echo "=== Rerun done: ${TOTAL}/${TOTAL} ==="
