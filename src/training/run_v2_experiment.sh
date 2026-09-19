#!/usr/bin/env bash
# Run the 4 x 2 factorial of the v2 experiment, one variant at a time.
#
#   bash src/training/run_v2_experiment.sh
#   VARIANTS="d_none d_full" bash src/training/run_v2_experiment.sh   # a subset
#
# One GPU, runs strictly sequential: the P5000 has 16 GB and nothing to gain from
# sharing it. See docs/serveur-entrainement-renard.md.
set -uo pipefail

REPO="${REPO:-$HOME/MeaningBERT}"
VENV="${VENV:-$HOME/.venvs/meaningbert-v2}"
DATA="${DATA:-$REPO/data/v2}"
LOGS="${LOGS:-$REPO/results/v2-runs}"
CHECKPOINT="${CHECKPOINT:-microsoft/deberta-v3-base}"
EPOCHS="${EPOCHS:-15}"
PATIENCE="${PATIENCE:-4}"
BATCH="${BATCH:-32}"
ACCUM="${ACCUM:-1}"
HEAD="${HEAD:-sigmoid}"
SEED="${SEED:-42}"
VARIANTS="${VARIANTS:-a_none b_none c_none d_none a_full b_full c_full d_full}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTHONPATH="$REPO/src"
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT="${WANDB_PROJECT:-meaningbert-v2}"

mkdir -p "$LOGS"
echo "GPU              : $CUDA_VISIBLE_DEVICES"
echo "checkpoint       : $CHECKPOINT"
echo "output head      : $HEAD   (correction C3)"
echo "epochs / patience: $EPOCHS / $PATIENCE"
echo "effective batch  : $((BATCH * ACCUM))"
echo "variants         : $VARIANTS"
echo
echo "Ordre : les quatre variantes sans augmentation d'abord. Elles sont les plus petites,"
echo "donc l'echelle a -> b -> c -> d est lisible bien avant la fin de l'experience."
echo

started=$(date +%s)
for variant in $VARIANTS; do
    path="$DATA/$variant"
    if [ ! -d "$path" ]; then
        echo "[SKIP] $variant : $path absent"
        continue
    fi
    log="$LOGS/$variant.log"
    echo "[RUN ] $variant -> $log"
    run_start=$(date +%s)
    "$VENV/bin/python" "$REPO/src/training/few_shot_training.py" \
        --variant_path "$path" \
        --checkpoint "$CHECKPOINT" \
        --output_head "$HEAD" \
        --num_epochs "$EPOCHS" \
        --early_stopping_patience "$PATIENCE" \
        --per_device_train_batch_size "$BATCH" \
        --gradient_accumulation_steps "$ACCUM" \
        --dataloader_num_workers 6 \
        --seed "$SEED" \
        --results_json "$LOGS/$variant.json" \
        > "$log" 2>&1
    status=$?
    elapsed=$(( $(date +%s) - run_start ))
    if [ $status -eq 0 ]; then
        echo "[ OK ] $variant en $((elapsed / 60)) min $((elapsed % 60)) s"
    else
        echo "[FAIL] $variant (code $status) apres $((elapsed / 60)) min, voir $log"
        tail -5 "$log" | sed 's/^/       /'
    fi
done
echo
echo "Total : $(( ($(date +%s) - started) / 60 )) min"
