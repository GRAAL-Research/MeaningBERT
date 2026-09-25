#!/usr/bin/env bash
# Experiment 3 of docs/v3-echelle-signee.md: train the polarity head, one GPU per worker.
#
# Same shape as run_worker.sh, and deliberately no more than that: this experiment answers
# "does a three-class polarity head work on this corpus at all", so it needs a log, a
# resumable skip and an out-of-memory fallback, not an orchestration layer.
#
#   GPU=0 CELLS="microsoft/deberta-v3-base:42;microsoft/deberta-v3-base:43" \
#     nohup bash src/training/run_polarity.sh renard-gpu0 &
#
# A cell is "<checkpoint>:<seed>". A cell whose metrics.json already exists is skipped, so
# a killed worker resumes where it stopped instead of redoing hours of work.

set -u

NAME="${1:?usage: run_polarity.sh <worker-name>}"
REPO="${REPO:-$HOME/MeaningBERT-v3}"
VENV="${VENV:-$HOME/.venvs/meaningbert-v2}"
CORPUS="${CORPUS:-$REPO/datastore/polarity}"
ROOT="${ROOT:-$REPO/results/polarity}"
GPU="${GPU:-0}"
EPOCHS="${EPOCHS:-2}"
LR="${LR:-1e-5}"
MICRO="${MICRO:-8}"
ACCUM="${ACCUM:-4}"
MAXLEN="${MAXLEN:-256}"
DEV_SAMPLE="${DEV_SAMPLE:-4000}"
KEEP_MODEL="${KEEP_MODEL:-true}"
MIN_FREE_GB="${MIN_FREE_GB:-25}"

mkdir -p "$ROOT"
echo "[$(date '+%F %T')] worker $NAME demarre sur GPU $GPU, cellules : $CELLS"

run_cell() {
    local checkpoint="$1" seed="$2"
    local slug; slug=$(echo "$checkpoint" | tr '/' '-')
    local out="$ROOT/$slug/seed$seed"
    local log="$out.log"

    if [ -f "$out/metrics.json" ]; then
        echo "[SKIP] $slug seed $seed : deja fait"
        return 0
    fi

    # A full disk is how two of the three overnight failures of the v2 campaign happened,
    # and a run that dies at hour four costs more than the check that would have refused it.
    local free; free=$(df -BG --output=avail "$REPO" | tail -1 | tr -dc '0-9')
    if [ "$free" -lt "$MIN_FREE_GB" ]; then
        echo "[STOP] $slug seed $seed : ${free} Go libres, moins que $MIN_FREE_GB"
        return 1
    fi

    mkdir -p "$(dirname "$out")"
    local keep="--keep-model"; [ "$KEEP_MODEL" = "true" ] || keep="--no-keep-model"

    # Stepped fallback on out-of-memory, halving the micro-batch and doubling accumulation
    # so the effective batch, and therefore the result, stays the same.
    local micro="$MICRO" accum="$ACCUM"
    for attempt in 1 2 3; do
        echo "[RUN ] $slug seed $seed  micro=$micro x accum=$accum (essai $attempt)"
        local start; start=$(date +%s)
        CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$REPO/src" "$VENV/bin/python" \
            "$REPO/src/training/train_polarity.py" \
            --corpus "$CORPUS" --checkpoint "$checkpoint" --output-dir "$out" \
            --seed "$seed" --epochs "$EPOCHS" --lr "$LR" \
            --batch-size "$micro" --grad-accum "$accum" \
            --max-length "$MAXLEN" --dev-sample "$DEV_SAMPLE" $keep \
            > "$log" 2>&1
        local status=$? elapsed=$(( $(date +%s) - start ))

        if [ $status -eq 0 ]; then
            echo "[ OK ] $slug seed $seed en $((elapsed / 60)) min"
            grep -E "^test |^sonde " "$log" | sed 's/^/       /'
            return 0
        fi
        if grep -qi "out of memory" "$log"; then
            micro=$(( micro / 2 )); accum=$(( accum * 2 ))
            [ "$micro" -ge 1 ] || { echo "[FAIL] $slug seed $seed : OOM jusqu'a micro=1"; return 1; }
            echo "[OOM ] $slug seed $seed : on retombe sur micro=$micro"
            continue
        fi
        echo "[FAIL] $slug seed $seed : code $status, voir $log"
        tail -5 "$log" | sed 's/^/       /'
        return 1
    done
}

IFS=';' read -ra cells <<< "${CELLS:?CELLS est vide}"
for cell in "${cells[@]}"; do
    [ -n "$cell" ] || continue
    run_cell "${cell%%:*}" "${cell##*:}"
done

touch "$ROOT/WORKER_COMPLETE-$NAME"
echo "[$(date '+%F %T')] worker $NAME termine"
