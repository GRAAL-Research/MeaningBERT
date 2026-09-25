#!/usr/bin/env bash
# Experiment 3 of docs/v3-echelle-signee.md: the polarity head, one GPU per worker.
#
#   GPU=0 CONDITION=none CELLS="nli-deberta-v3-base:42;bert:42" \
#     nohup bash src/training/run_polarity.sh renard-gpu0 &
#
# A cell is "<arch tag>:<seed>", the tag coming from the registry below. A cell whose
# metrics.json already exists is skipped, so a killed worker resumes where it stopped
# instead of redoing hours of work.

set -u

NAME="${1:?usage: run_polarity.sh <worker-name>}"
REPO="${REPO:-$HOME/MeaningBERT-v3}"
VENV="${VENV:-$HOME/.venvs/meaningbert-v2}"
CONDITION="${CONDITION:-none}"
CORPUS="${CORPUS:-$REPO/datastore/polarity}"
ROOT="${ROOT:-$REPO/results/polarity}"
GPU="${GPU:-0}"
EPOCHS="${EPOCHS:-2}"
LR="${LR:-1e-5}"
MAXLEN="${MAXLEN:-256}"
# 0 means "use the stratified dev split whole", which is what a corpus built with
# --eval-per-class wants: sampling it again would undo the stratification at random.
DEV_SAMPLE="${DEV_SAMPLE:-0}"
KEEP_MODEL="${KEEP_MODEL:-true}"
MIN_FREE_GB="${MIN_FREE_GB:-25}"

# tag|checkpoint|micro batch|grad accum|minimum compute capability x10
#
# The micro batch is sized for an 11 GB card at length 256 in fp32, the smallest in the
# fleet, and the accumulation holds the effective batch at 32 everywhere so a result is
# comparable across machines.
#
# The last field is a hardware gate, not a preference. ModernBERT is built around
# FlashAttention and unpadded attention, both of which need compute 8.0 or better. renard
# and souris are Pascal 6.1: it would run, and it would run so degraded that comparing it
# to anything else would be meaningless. The gate refuses rather than producing a number
# nobody should trust.
ALL_ARCHS="\
bert|bert-base-uncased|8|4|0
deberta-v3-base|microsoft/deberta-v3-base|8|4|0
nli-deberta-v3-base|MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli|8|4|0
stsb-roberta-base|cross-encoder/stsb-roberta-base|8|4|0
deberta-v3-large|microsoft/deberta-v3-large|4|8|0
nli-deberta-v3-large|MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli|4|8|0
roberta-large-mnli|roberta-large-mnli|4|8|0
modernbert-base|answerdotai/ModernBERT-base|8|4|80
modernbert-large|answerdotai/ModernBERT-large|4|8|80"

compute_capability() {
    nvidia-smi --id="$GPU" --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
        | tr -d '. ' | head -1
}

arch_line() {
    echo "$ALL_ARCHS" | awk -F'|' -v want="$1" '$1 == want { print; found = 1 } END { exit !found }'
}

mkdir -p "$ROOT"
CAPABILITY=$(compute_capability)
echo "[$(date '+%F %T')] worker $NAME, GPU $GPU (compute ${CAPABILITY:-inconnu}), condition $CONDITION"
echo "  corpus  : $CORPUS"
echo "  cellules: $CELLS"

run_cell() {
    local tag="$1" seed="$2" line checkpoint micro accum needed

    if ! line=$(arch_line "$tag"); then
        echo "[FAIL] $tag : architecture inconnue"
        return 1
    fi
    IFS='|' read -r _ checkpoint micro accum needed <<< "$line"

    if [ -n "$CAPABILITY" ] && [ "$needed" -gt 0 ] && [ "$CAPABILITY" -lt "$needed" ]; then
        echo "[GATE] $tag : demande compute $needed, ce GPU est a $CAPABILITY. Refuse."
        return 0
    fi

    local out="$ROOT/$tag-$CONDITION/seed$seed"
    local log="$out.log"
    if [ -f "$out/metrics.json" ]; then
        echo "[SKIP] $tag-$CONDITION seed $seed : deja fait"
        return 0
    fi

    # Claim the cell before starting it. Two workers on the SAME machine share a results
    # tree, which is what lets a finished worker pick up another's backlog; without a claim
    # they would both start the cell the other is halfway through, because metrics.json is
    # only written at the end. mkdir is the atomic primitive here, touch is not.
    mkdir -p "$(dirname "$out")"
    if ! mkdir "$out.claim" 2>/dev/null; then
        local age=$(( ($(date +%s) - $(stat -c %Y "$out.claim" 2>/dev/null || date +%s)) / 3600 ))
        if [ "$age" -lt 24 ]; then
            echo "[BUSY] $tag-$CONDITION seed $seed : reclamee ailleurs il y a ${age} h"
            return 0
        fi
        # A claim older than a day belongs to a worker that died. Taking it back beats
        # leaving a hole in the grid.
        echo "[STALE] $tag-$CONDITION seed $seed : reclamation de ${age} h reprise"
    fi
    trap 'rmdir "$out.claim" 2>/dev/null' RETURN

    # A full disk is how two of the three overnight failures of the v2 campaign happened,
    # and a run that dies at hour four costs more than the check that would have refused it.
    local free; free=$(df -BG --output=avail "$REPO" | tail -1 | tr -dc '0-9')
    if [ "$free" -lt "$MIN_FREE_GB" ]; then
        echo "[STOP] $tag-$CONDITION seed $seed : ${free} Go libres, moins que $MIN_FREE_GB"
        return 1
    fi

    local keep="--keep-model"; [ "$KEEP_MODEL" = "true" ] || keep="--no-keep-model"

    # Stepped fallback on out-of-memory, halving the micro batch and doubling accumulation
    # so the effective batch, and therefore the result, stays the same.
    for attempt in 1 2 3; do
        echo "[RUN ] $tag-$CONDITION seed $seed  ($checkpoint)  micro=$micro x accum=$accum  essai $attempt"
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
            rmdir "$out.claim" 2>/dev/null
            echo "[ OK ] $tag-$CONDITION seed $seed en $((elapsed / 60)) min"
            grep -E "^test |^sonde " "$log" | sed 's/^/       /'
            return 0
        fi
        if grep -qi "out of memory" "$log"; then
            micro=$(( micro / 2 )); accum=$(( accum * 2 ))
            [ "$micro" -ge 1 ] || { echo "[FAIL] $tag seed $seed : OOM jusqu'a micro=1"; return 1; }
            echo "[OOM ] $tag seed $seed : on retombe sur micro=$micro"
            continue
        fi
        echo "[FAIL] $tag-$CONDITION seed $seed : code $status, voir $log"
        tail -5 "$log" | sed 's/^/       /'
        return 1
    done
}

IFS=';' read -ra cells <<< "${CELLS:?CELLS est vide}"
for cell in "${cells[@]}"; do
    [ -n "$cell" ] || continue
    run_cell "${cell%%:*}" "${cell##*:}"
done

touch "$ROOT/WORKER_COMPLETE-$NAME-$CONDITION"
echo "[$(date '+%F %T')] worker $NAME termine"
