#!/bin/bash
# Full sweep on a single GPU (use sweep_gpu{0,1,2}.sh for multi-GPU).
# 14 models x 10 folds x 2 augmentations = 280 runs
# Reads pre-generated datasets from ./data (run prepare_datasets.py first)
export WANDB_PROJECT="meaningbert-checkpoint-sweep"

FOLDS=(0 1 2 3 4 5 6 7 8 9)
SEEDS=(42 43 44 45 46 47 48 49 50 51)
AUGMENTATIONS=("swap" "back_translation")
COMMON="--data_dir=./data --early_stopping_patience=50 --bf16 --dataloader_num_workers=4"

TOTAL=280
RUN=0

# checkpoint -> "learning_rate freeze_layers batch_size"
declare -A CHECKPOINTS
CHECKPOINTS=(
    ["bert-base-uncased"]="5e-5 0 128"
    ["microsoft/deberta-v3-small"]="2e-5 0 128"
    ["microsoft/deberta-v3-base"]="2e-5 0 128"
    ["microsoft/deberta-v3-large"]="2e-5 6 64"
    ["microsoft/deberta-v2-xlarge"]="1e-5 12 32"
    ["google/electra-large-discriminator"]="3e-5 6 64"
    ["answerdotai/ModernBERT-large"]="5e-5 6 64"
    ["openai-community/gpt2"]="2e-5 0 128"
    ["HuggingFaceTB/SmolLM2-135M"]="2e-5 0 128"
    ["HuggingFaceTB/SmolLM2-360M"]="2e-5 0 128"
    ["Qwen/Qwen2.5-0.5B"]="1e-5 0 64"
    ["Qwen/Qwen2.5-1.5B"]="1e-5 8 32"
    ["google/gemma-2-2b"]="1e-5 10 16"
    ["microsoft/phi-2"]="1e-5 10 16"
)

for checkpoint in "${!CHECKPOINTS[@]}"; do
    read -r lr freeze bs <<< "${CHECKPOINTS[$checkpoint]}"
    for aug in "${AUGMENTATIONS[@]}"; do
        for i in "${!FOLDS[@]}"; do
            fold=${FOLDS[$i]}
            seed=${SEEDS[$i]}
            RUN=$((RUN + 1))
            echo "=== [${RUN}/${TOTAL}] ${checkpoint} fold=${fold} aug=${aug} ==="
            python few_shot_training.py --seed="${seed}" --fold="${fold}" \
                --checkpoint="${checkpoint}" --learning_rate="${lr}" \
                --freeze_layers="${freeze}" --per_device_train_batch_size="${bs}" \
                --data_augmentation="${aug}" ${COMMON}
        done
    done
done
echo "=== Done: ${TOTAL}/${TOTAL} runs ==="
