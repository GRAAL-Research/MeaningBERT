#!/usr/bin/env bash
# Score the v4 heads on XNLI test in other languages (xnli_multi.py), on souris.
#
#   setsid nohup bash src/training/xnli_v4_daemon.sh > results/v4/xnli.log 2>&1 &
#
# Same pattern as score_v4_daemon.sh: pull each finished head from renard, score it, delete
# it, regenerate the report. Three seeds per family: the ensemble the French report shows
# rests on more, and the language question is about the recipe, not about seed variance.

set -u

REPO="${REPO:-$HOME/MeaningBERT-v3}"
VENV="${VENV:-$HOME/.venvs/meaningbert-v2}"
TRAIN_HOST="${TRAIN_HOST:-dabea241@132.203.114.27}"
REMOTE_REPO="${REMOTE_REPO:-MeaningBERT-v3}"
OUT="$REPO/results/v4/xnli"
SCRATCH="${SCRATCH:-$REPO/results/v4/xnli-tmp}"
LANGS="${LANGS:-all}"  # the 14 XNLI languages; de,ar,es,zh was the first wave
POLL="${POLL:-900}"

cd "$REPO" || exit 1
mkdir -p "$OUT" "$SCRATCH"
# Heads are read from $SCRATCH/results/polarity/..., where they are pulled: scoring runs from
# $SCRATCH so the French daemon's links under $REPO/results/polarity are never touched.
py() { (cd "$SCRATCH" && PYTHONPATH="$REPO/src" "$VENV/bin/python" "$REPO/$1" "${@:2}" < /dev/null); }
# souris filled its disk on 2026-10-21 (Opus-MT models plus pulled heads): below this much
# free space a pass pulls nothing and waits, rather than dying halfway through a write.
MIN_FREE_GB="${MIN_FREE_GB:-4}"
disk_ok() {
    local free
    free=$(df -BG --output=avail "$REPO" | tail -1 | tr -dc '0-9')
    [ "$free" -ge "$MIN_FREE_GB" ] || { echo "[DISK] ${free} Go libres, moins que $MIN_FREE_GB : passage saute"; return 1; }
}

scored() { grep -lq "\"$1\"" "$OUT"/scores-*.json 2>/dev/null; }


baselines() {
    local family
    for family in offshelf-en offshelf-multi; do
        [ -f "$OUT/scores-$family.json" ] && continue
        echo "[$(date '+%F %T')] $family"
        py src/diagnostics/xnli_multi.py score --heads "$family" --langs "$LANGS" --out "$OUT/scores-$family.json" \
            --translations "$OUT/translations" || echo "[FAIL] $family"
    done
    # souris has little disk: the translations are cached in $OUT, the Opus-MT models are
    # not needed again once the English heads have read every language.
    [ -f "$OUT/scores-offshelf-en.json" ] && rm -rf "$HOME"/.cache/huggingface/hub/models--Helsinki-NLP--opus-mt-*-en
}

# name|family|remote model folder
candidates() {
    local seed regime
    for seed in 42 43 44; do
        echo "v3-auglex-seed$seed|v3|results/polarity/nli-deberta-v3-large-fulllex/seed$seed/model"
        for regime in en fr enfr none aug replay mreplay; do
            echo "v4-xlmr-$regime-seed$seed|v4-$regime|results/polarity/xlmr-large-nli-v4$regime/seed$seed/model"
        done
        for regime in en fr enfr none aug replay mreplay; do
            echo "v4-mdeberta-$regime-seed$seed|v4m-$regime|results/polarity/mdeberta-v3-base-nli-v4$regime/seed$seed/model"
        done
    done
}
total=$(candidates | wc -l)

while true; do
    # Retried every pass: a baseline that failed (out of memory on the shared card) is
    # what every bar of the report rests on.
    baselines
    done_count=0
    while IFS='|' read -r name family folder; do
        if scored "$name"; then done_count=$((done_count + 1)); continue; fi
        cell="${folder%/model}"
        disk_ok || break
        ssh -n -o BatchMode=yes "$TRAIN_HOST" \
            "cd $REMOTE_REPO && test -f $folder/model.safetensors -a ! -d $cell.claim" || continue
        local_dir="$SCRATCH/$folder"
        mkdir -p "$local_dir"
        echo "[$(date '+%F %T')] $name : rapatriement"
        rsync -az "$TRAIN_HOST:$REMOTE_REPO/$folder/" "$local_dir/" < /dev/null \
            || { echo "[FAIL] $name : rsync"; rm -rf "$local_dir"; continue; }
        echo "[$(date '+%F %T')] $name : score"
        if py src/diagnostics/xnli_multi.py score --heads "$family" --only "$name" --langs "$LANGS" \
            --out "$OUT/scores-$name.json" --translations "$OUT/translations"; then
            done_count=$((done_count + 1))
            py src/diagnostics/xnli_multi.py report --scores "$OUT" --out "$OUT/report.json" --langs "$LANGS" \
                || echo "[FAIL] rapport"
        else
            echo "[FAIL] $name : score"
        fi
        rm -rf "$local_dir"
    done < <(candidates)
    echo "[$(date '+%F %T')] $done_count / $total tetes scorees"
    [ "$done_count" -ge "$total" ] && { touch "$OUT/ALL_SCORED"; exit 0; }
    sleep "$POLL"
done
