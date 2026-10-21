#!/usr/bin/env bash
# Score every finished v4 head on the French corpora as soon as it exists, on souris.
#
#   setsid nohup bash src/training/score_v4_daemon.sh > results/v4/scorer.log 2>&1 &
#
# The heads train on renard, whose two cards are full; scoring there would risk an
# out-of-memory that restarts an eight-hour run. souris pulls each finished head over the
# campus network instead, scores it with discovery_fr.py, deletes it (souris has little
# disk), and regenerates the report. It waits for the corpus translation to end first, so
# the two never share the card.
#
# A v4 head is finished when its metrics.json exists and its claim is gone; a v3 seed is
# there once its model.safetensors is, since the transfer moves the folder in last.

set -u

REPO="${REPO:-$HOME/MeaningBERT-v3}"
VENV="${VENV:-$HOME/.venvs/meaningbert-v2}"
TRAIN_HOST="${TRAIN_HOST:-dabea241@132.203.114.27}"
REMOTE_REPO="${REMOTE_REPO:-MeaningBERT-v3}"
OUT="$REPO/results/v4/discovery"
SCRATCH="${SCRATCH:-$REPO/results/v4/scoring-tmp}"
POLL="${POLL:-600}"
SEEDS="42 43 44 45 46 47 48 49 50 51"
EXPECTED=88  # 30 XLM-R and 30 mDeBERTa v4 heads, 10 v3 seeds, then 15 ablation and replay
# mDeBERTa heads and 3 replay XLM-R heads of the second wave (2026-10-21)

cd "$REPO" || exit 1
mkdir -p "$OUT" "$SCRATCH"

scored() {
    grep -lq "\"$1\"" "$OUT"/scores-*.json 2>/dev/null
}

# name|family|remote model folder|remote readiness test
candidates() {
    local seed regime
    for seed in $SEEDS; do
        echo "v3-auglex-seed$seed|v3|results/polarity/nli-deberta-v3-large-fulllex/seed$seed/model|test -f results/polarity/nli-deberta-v3-large-fulllex/seed$seed/model/model.safetensors"
        for regime in en fr enfr; do
            cell="results/polarity/xlmr-large-nli-v4$regime/seed$seed"
            echo "v4-xlmr-$regime-seed$seed|v4-$regime|$cell/model|test -f $cell/metrics.json -a -f $cell/model/model.safetensors -a ! -d $cell.claim"
            cell="results/polarity/mdeberta-v3-base-nli-v4$regime/seed$seed"
            echo "v4-mdeberta-$regime-seed$seed|v4m-$regime|$cell/model|test -f $cell/metrics.json -a -f $cell/model/model.safetensors -a ! -d $cell.claim"
        done
    done
    for seed in 42 43 44 45 46; do
        for regime in none aug replay; do
            cell="results/polarity/mdeberta-v3-base-nli-v4$regime/seed$seed"
            echo "v4-mdeberta-$regime-seed$seed|v4m-$regime|$cell/model|test -f $cell/metrics.json -a -f $cell/model/model.safetensors -a ! -d $cell.claim"
        done
    done
    for seed in 42 43 44; do
        cell="results/polarity/xlmr-large-nli-v4replay/seed$seed"
        echo "v4-xlmr-replay-seed$seed|v4-replay|$cell/model|test -f $cell/metrics.json -a -f $cell/model/model.safetensors -a ! -d $cell.claim"
    done
}

# WAIT_TRANSLATION=0 scores alongside the translation: the card has room for one head, and
# the translation halves its batch on out-of-memory rather than dying.
while [ "${WAIT_TRANSLATION:-1}" = "1" ] && pgrep -f "[t]ranslate_polarity_corpus.py" > /dev/null; do
    echo "[WAIT] traduction en cours, nouvel essai dans $POLL s"
    sleep "$POLL"
done

# XNLI joined the evaluated corpora on 2026-10-11, which changes the pair set: the scores
# computed before it no longer merge into the report. The off-the-shelf heads are scored
# again here once; the v3 seed 42 falls back into the loop below by losing its file.
if [ ! -f "$OUT/BASELINES_XNLI" ]; then
    rm -f "$OUT/scores-v3.json"
    for family in offshelf-en offshelf-multi; do
        echo "[$(date '+%F %T')] $family : score avec XNLI"
        PYTHONPATH=src "$VENV/bin/python" src/diagnostics/discovery_fr.py score \
            --heads "$family" --out "$OUT/scores-$family.json" < /dev/null || { echo "[FAIL] $family"; exit 1; }
    done
    touch "$OUT/BASELINES_XNLI"
fi

while true; do
    done_count=0
    while IFS='|' read -r name family folder ready; do
        if scored "$name"; then
            done_count=$((done_count + 1))
            continue
        fi
        ssh -n -o BatchMode=yes "$TRAIN_HOST" "cd $REMOTE_REPO && $ready" || continue

        local_dir="$SCRATCH/$folder"
        echo "[$(date '+%F %T')] $name : rapatriement"
        mkdir -p "$local_dir"
        if ! rsync -az "$TRAIN_HOST:$REMOTE_REPO/$folder/" "$local_dir/" < /dev/null; then
            echo "[FAIL] $name : rsync"
            rm -rf "$local_dir"
            continue
        fi
        # discovery_fr.py reads results/polarity/... relative to the repository, so the
        # pulled head is linked where the registry expects it.
        mkdir -p "$(dirname "$folder")"
        ln -sfn "$local_dir" "$folder"
        echo "[$(date '+%F %T')] $name : score"
        if PYTHONPATH=src "$VENV/bin/python" src/diagnostics/discovery_fr.py score \
            --heads "$family" --only "$name" --out "$OUT/scores-$name.json" < /dev/null; then
            done_count=$((done_count + 1))
            PYTHONPATH=src "$VENV/bin/python" src/diagnostics/discovery_fr.py report \
                --scores "$OUT" --out "$OUT/report.json" < /dev/null || echo "[FAIL] rapport"
        else
            echo "[FAIL] $name : score"
        fi
        rm -f "$folder"
        rm -rf "$local_dir"
    done < <(candidates)

    echo "[$(date '+%F %T')] $done_count / $EXPECTED tetes scorees"
    if [ "$done_count" -ge "$EXPECTED" ]; then
        touch "$OUT/ALL_SCORED"
        echo "[$(date '+%F %T')] toutes les tetes sont scorees"
        exit 0
    fi
    sleep "$POLL"
done
