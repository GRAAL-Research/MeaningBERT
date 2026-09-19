#!/usr/bin/env bash
# Run the 4 x 2 factorial of the v2 experiment, one variant at a time.
#
#   bash src/training/run_v2_experiment.sh
#   VARIANTS="d_none d_full" bash src/training/run_v2_experiment.sh   # a subset
#   FORCE=1 bash src/training/run_v2_experiment.sh                    # redo finished runs
#
# One GPU, strictly sequential: the P5000 has 16 GB and nothing to gain from sharing it.
# See docs/serveur-entrainement-renard.md.
set -uo pipefail

REPO="${REPO:-$HOME/MeaningBERT}"
VENV="${VENV:-$HOME/.venvs/meaningbert-v2}"
DATA="${DATA:-$REPO/data/v2}"
LOGS="${LOGS:-$REPO/results/v2-runs}"
CHECKPOINT="${CHECKPOINT:-microsoft/deberta-v3-base}"
EPOCHS="${EPOCHS:-10}"
PATIENCE="${PATIENCE:-3}"
EFFECTIVE_BATCH="${EFFECTIVE_BATCH:-32}"
HEAD="${HEAD:-sigmoid}"
SEED="${SEED:-42}"
FORCE="${FORCE:-0}"
VARIANTS="${VARIANTS:-a_none b_none c_none d_none a_full b_full c_full d_full}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTHONPATH="$REPO/src"
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT="${WANDB_PROJECT:-meaningbert-v2}"
# Long sequences make the allocator fragment badly on a 16 GB card.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Micro-batch per variant. The effective batch stays EFFECTIVE_BATCH everywhere through
# gradient accumulation, so the optimisation is comparable across the whole factorial.
#
# Conditions a, b and c top out at 209 tokens and sit at 13.7 GB of the 16 GB at a
# micro-batch of 32. The v2 corpora reach 512 after truncation, because SimpleText is
# scientific abstracts and PLABA is biomedical; a 512-token batch of 32 would not fit.
# Overridable so the same runner serves several backbones. bert-base-uncased is lighter
# than deberta-v3-base (110 M against 184 M, and no disentangled attention), so it takes a
# larger micro-batch for the same memory.
MICRO_SMALL="${MICRO_SMALL:-32}"
MICRO_D="${MICRO_D:-8}"

micro_batch_for() {
    case "$1" in
        d_*) echo "$MICRO_D" ;;
        *)   echo "$MICRO_SMALL" ;;
    esac
}

run_one() {
    local variant="$1" micro="$2" log="$3"
    local accum=$(( EFFECTIVE_BATCH / micro ))
    [ "$accum" -lt 1 ] && accum=1
    "$VENV/bin/python" "$REPO/src/training/few_shot_training.py" \
        --variant_path "$DATA/$variant" \
        --checkpoint "$CHECKPOINT" \
        --output_head "$HEAD" \
        --num_epochs "$EPOCHS" \
        --early_stopping_patience "$PATIENCE" \
        --per_device_train_batch_size "$micro" \
        --gradient_accumulation_steps "$accum" \
        --dataloader_num_workers 6 \
        --seed "$SEED" \
        --results_json "$LOGS/$variant.json" \
        > "$log" 2>&1
}

mkdir -p "$LOGS"
echo "GPU              : $CUDA_VISIBLE_DEVICES"
echo "checkpoint       : $CHECKPOINT"
echo "output head      : $HEAD   (correction C3)"
echo "epochs / patience: $EPOCHS / $PATIENCE"
echo "effective batch  : $EFFECTIVE_BATCH (micro-batch adapte par variante)"
echo "variants         : $VARIANTS"
echo

started=$(date +%s)
declare -a failed=()
for variant in $VARIANTS; do
    if [ ! -d "$DATA/$variant" ]; then
        echo "[SKIP] $variant : corpus absent"
        continue
    fi
    if [ "$FORCE" != "1" ] && [ -s "$LOGS/$variant.json" ]; then
        echo "[DONE] $variant : deja termine, resultat conserve"
        continue
    fi

    log="$LOGS/$variant.log"
    micro=$(micro_batch_for "$variant")
    echo "[RUN ] $variant  micro-batch=$micro  -> $log"
    run_start=$(date +%s)
    run_one "$variant" "$micro" "$log"
    status=$?

    # An out-of-memory kill is the one failure worth an automatic second try: halving the
    # micro-batch changes the memory, not the effective batch, so the result stays
    # comparable. Anything else is a real bug and must surface instead of being retried.
    if [ $status -ne 0 ] && grep -qiE "out of memory|CUDA error: out of memory" "$log"; then
        retry=$(( micro / 4 )); [ "$retry" -lt 1 ] && retry=1
        echo "       OOM a micro-batch=$micro, nouvelle tentative a $retry"
        run_one "$variant" "$retry" "$log.retry"
        status=$?
        [ $status -eq 0 ] && mv "$log.retry" "$log"
    fi

    elapsed=$(( $(date +%s) - run_start ))
    if [ $status -eq 0 ]; then
        echo "[ OK ] $variant en $((elapsed / 60)) min $((elapsed % 60)) s"
    else
        echo "[FAIL] $variant (code $status) apres $((elapsed / 60)) min, voir $log"
        tail -6 "$log" | sed 's/^/       /'
        failed+=("$variant")
    fi
done

echo
echo "Total : $(( ($(date +%s) - started) / 60 )) min"
if [ ${#failed[@]} -gt 0 ]; then
    echo "ECHECS : ${failed[*]}"
    exit 1
fi
echo "TOUTES LES VARIANTES SONT PASSEES"
