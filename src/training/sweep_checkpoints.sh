#!/bin/bash
# Full sweep on a single GPU (use sweep_gpu{0,1,2}.sh for multi-GPU).
# 14 models x 10 seeds = 140 runs
# Reads pre-generated datasets from ./data (run prepare_datasets.py first)
export WANDB_PROJECT="meaningbert-checkpoint-sweep"

SEEDS=(42 43 44 45 46 47 48 49 50 51)
COMMON="--data_dir=./data --data_augmentation=swap --gradient_accumulation_steps=2 --early_stopping_patience=50"

declare -A CHECKPOINTS
# checkpoint -> "learning_rate freeze_layers"
CHECKPOINTS=(
    ["bert-base-uncased"]="5e-5 0"
    ["microsoft/deberta-v3-small"]="2e-5 0"
    ["microsoft/deberta-v3-base"]="2e-5 0"
    ["microsoft/deberta-v3-large"]="2e-5 6"
    ["microsoft/deberta-v2-xlarge"]="1e-5 12"
    ["google/electra-large-discriminator"]="3e-5 6"
    ["answerdotai/ModernBERT-large"]="5e-5 6"
    ["openai-community/gpt2"]="2e-5 0"
    ["HuggingFaceTB/SmolLM2-135M"]="2e-5 0"
    ["HuggingFaceTB/SmolLM2-360M"]="2e-5 0"
    ["Qwen/Qwen2.5-0.5B"]="1e-5 0"
    ["Qwen/Qwen2.5-1.5B"]="1e-5 8"
    ["google/gemma-2-2b"]="1e-5 10"
    ["microsoft/phi-2"]="1e-5 10"
)

for checkpoint in "${!CHECKPOINTS[@]}"; do
    read -r lr freeze <<< "${CHECKPOINTS[$checkpoint]}"
    for seed in "${SEEDS[@]}"; do
        echo "=== Training checkpoint=${checkpoint} seed=${seed} lr=${lr} freeze=${freeze} ==="
        python few_shot_training.py --seed="${seed}" --checkpoint="${checkpoint}" \
            --learning_rate="${lr}" --freeze_layers="${freeze}" ${COMMON}
    done
done
