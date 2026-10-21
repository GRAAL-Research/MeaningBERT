#!/usr/bin/env bash
# WiSE-FT sweep of the v4 heads trained in English, interpolated and scored on souris.
#
#   setsid nohup bash src/training/wise_v4_daemon.sh > results/v4/wise.log 2>&1 &
#
# For each backbone and seed, the fine-tuned head is pulled once from renard, mixed with the
# off-the-shelf head it was trained from at every alpha (interpolate_heads.py), and each mix
# is scored with discovery_fr.py then deleted: souris has little disk, and a mix is cheap to
# rebuild. A mix already scored is skipped, so the sweep resumes after a reboot.

set -u

REPO="${REPO:-$HOME/MeaningBERT-v3}"
VENV="${VENV:-$HOME/.venvs/meaningbert-v2}"
TRAIN_HOST="${TRAIN_HOST:-dabea241@132.203.114.27}"
REMOTE_REPO="${REMOTE_REPO:-MeaningBERT-v3}"
OUT="$REPO/results/v4/discovery"
SCRATCH="${SCRATCH:-$REPO/results/v4/wise-tmp}"
ALPHAS="${ALPHAS:-0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9}"
SEEDS="${SEEDS:-42 43 44 45 46 47 48 49 50 51}"
# English first: the best regime of the first wave gives the answer soonest.
REGIMES="${REGIMES:-en fr enfr}"

# backbone key|off-the-shelf checkpoint|remote folder prefix of the v4 grid
BACKBONES="\
mdeberta|MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7|results/polarity/mdeberta-v3-base-nli-v4
xlmr|joeddav/xlm-roberta-large-xnli|results/polarity/xlmr-large-nli-v4"

cd "$REPO" || exit 1
mkdir -p "$OUT" "$SCRATCH"
py() { PYTHONPATH=src "$VENV/bin/python" "$@" < /dev/null; }

# One pass per hour until every mix is scored: seeds still training on renard join later.
total=$(( $(echo "$BACKBONES" | wc -l) * $(echo $REGIMES | wc -w) * $(echo $SEEDS | wc -w) * $(echo $ALPHAS | wc -w) ))
while [ "$(ls "$OUT"/scores-v4-wise-*.json 2>/dev/null | wc -l)" -lt "$total" ]; do
for regime in $REGIMES; do
while IFS='|' read -r key base prefix; do
    remote="$prefix$regime"
    for seed in $SEEDS; do
        tuned="$SCRATCH/$key-$regime-seed$seed"
        # A seed still training (or never trained) is skipped; the next pass picks it up.
        ssh -n -o BatchMode=yes "$TRAIN_HOST" \
            "cd $REMOTE_REPO && test -f $remote/seed$seed/metrics.json -a ! -d $remote/seed$seed.claim" || continue
        for alpha in $ALPHAS; do
            name="v4-wise-$key-$regime-a$alpha-seed$seed"
            grep -lq "\"$name\"" "$OUT"/scores-*.json 2>/dev/null && continue
            if [ ! -f "$tuned/model.safetensors" ]; then
                echo "[$(date '+%F %T')] $key seed $seed : rapatriement"
                mkdir -p "$tuned"
                rsync -az "$TRAIN_HOST:$REMOTE_REPO/$remote/seed$seed/model/" "$tuned/" < /dev/null \
                    || { echo "[FAIL] $key seed $seed : rsync"; rm -rf "$tuned"; continue 2; }
            fi
            mix="results/polarity/wise/$name/model"
            echo "[$(date '+%F %T')] $name : interpolation"
            py src/training/interpolate_heads.py --base "$base" --tuned "$tuned" --alpha "$alpha" --out "$mix" \
                || { echo "[FAIL] $name : interpolation"; rm -rf "results/polarity/wise/$name"; continue; }
            echo "[$(date '+%F %T')] $name : score"
            py src/diagnostics/discovery_fr.py score --heads "wise-$key" --only "$name" \
                --out "$OUT/scores-$name.json" || echo "[FAIL] $name : score"
            rm -rf "results/polarity/wise/$name"
        done
        # The report once per seed, not per mix: it reads every score file and grows with them.
        [ -d "$tuned" ] && { py src/diagnostics/discovery_fr.py report --scores "$OUT" --out "$OUT/report.json" \
            || echo "[FAIL] rapport"; }
        rm -rf "$tuned"
    done
done <<< "$BACKBONES"
done
sleep 3600
done

touch "$OUT/WISE_DONE"
echo "[$(date '+%F %T')] balayage WiSE-FT termine"
