#!/usr/bin/env bash
# v4 magnitude benchmark: the MeaningBERT v2 recipe on multilingual backbones, one GPU.
#
#   GPU=0 QUEUE="en|mdeberta|42;fr|mdeberta|42" \
#     nohup bash src/training/run_magnitude_queue.sh renard-g0-magnitude &
#
# The recipe is the published v2-large one (d_full, clamped head, effective batch 32, ten
# epochs, patience three); only the backbone and the corpus language change. A queue item is
# "<corpus>|<arch tag>|<seed>":
#
#   en    ~/MeaningBERT/data/v2/d_full            the v2 corpus, read in French zero-shot
#   fr    data/v4-magnitude/d_full-fr             the same corpus machine-translated to French
#   enfr  data/v4-magnitude/d_full-enfr           both
#
# An item whose results JSON exists is skipped, so a killed queue resumes where it stopped.

set -u

NAME="${1:?usage: run_magnitude_queue.sh <worker-name>}"
REPO="${REPO:-$HOME/MeaningBERT-v3}"
VENV="${VENV:-$HOME/.venvs/meaningbert-v2}"
GPU="${GPU:-0}"
ROOT="${ROOT:-$REPO/results/magnitude-v4}"
EPOCHS="${EPOCHS:-10}"
PATIENCE="${PATIENCE:-3}"
HEAD="${HEAD:-clamped}"

declare -A CORPUS_OF=(
    [en]="$HOME/MeaningBERT/data/v2/d_full"
    [fr]="$REPO/data/v4-magnitude/d_full-fr"
    [enfr]="$REPO/data/v4-magnitude/d_full-enfr"
)

# tag|checkpoint|micro batch|grad accum|learning rate
#
# The plain pretrained encoders, as the v2 trained deberta-v3-large and not an NLI head.
# Micro batches fit an 11 GB card at length 256 in fp32. XLM-R-large takes 1e-5: the
# trainer's family default (5e-5) is the one that diverges on large RoBERTa encoders.
ARCHS="\
mdeberta|microsoft/mdeberta-v3-base|8|4|2e-5
xlmr|FacebookAI/xlm-roberta-large|4|8|1e-5"

# XLM-R-large wedges the GTX 1080 Ti (run_polarity.sh, WEDGING): refused there, not tried.
GPU_NAME=$(nvidia-smi --id="$GPU" --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)

export CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$REPO/src" TOKENIZERS_PARALLELISM=false WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$ROOT"
echo "[$(date '+%F %T')] worker $NAME, GPU $GPU (${GPU_NAME:-inconnu})"

IFS=';' read -ra items <<< "${QUEUE:?QUEUE est vide}"
for item in "${items[@]}"; do
    [ -n "$item" ] || continue
    IFS='|' read -r corpus tag seed <<< "$item"
    data="${CORPUS_OF[$corpus]:-}"
    line=$(echo "$ARCHS" | awk -F'|' -v want="$tag" '$1 == want')
    if [ -z "$data" ] || [ -z "$line" ]; then
        echo "[FAIL] $item : corpus ou architecture inconnu"
        continue
    fi
    IFS='|' read -r _ checkpoint micro accum lr <<< "$line"
    if [ "$tag" = "xlmr" ] && [[ "$GPU_NAME" == *"GTX 1080 Ti"* ]]; then
        echo "[GATE] $item : XLM-R-large fige une $GPU_NAME. Refuse."
        continue
    fi
    cell="$ROOT/$tag-$corpus/seed$seed"
    if [ -f "$cell.json" ]; then
        echo "[SKIP] $item : deja fait"
        continue
    fi
    until [ -f "$data/dataset_dict.json" ] && { [ "$corpus" = "en" ] || [ -f "$data/READY" ]; }; do
        echo "[WAIT] $item : corpus $data pas encore pret"
        sleep 300
    done
    mkdir -p "$(dirname "$cell")"
    if ! mkdir "$cell.claim" 2>/dev/null; then
        echo "[BUSY] $item : reclamee ailleurs"
        continue
    fi
    echo "[RUN ] $item ($checkpoint) micro=$micro x accum=$accum lr=$lr"
    start=$(date +%s)
    MEANINGBERT_MODEL_DIR="$ROOT/models/$tag-$corpus" "$VENV/bin/python" "$REPO/src/training/few_shot_training.py" \
        --variant_path "$data" --checkpoint "$checkpoint" --output_head "$HEAD" \
        --num_epochs "$EPOCHS" --early_stopping_patience "$PATIENCE" --seed "$seed" \
        --learning_rate "$lr" --per_device_train_batch_size "$micro" --gradient_accumulation_steps "$accum" \
        --save_total_limit 1 --results_json "$cell.json" > "$cell.log" 2>&1
    status=$?
    rmdir "$cell.claim"
    if [ $status -eq 0 ] && [ -f "$cell.json" ]; then
        echo "[ OK ] $item en $(( ($(date +%s) - start) / 60 )) min"
    else
        echo "[FAIL] $item : code $status, voir $cell.log"
        tail -3 "$cell.log" | sed 's/^/       /'
    fi
done
echo "[$(date '+%F %T')] file $NAME terminee"
