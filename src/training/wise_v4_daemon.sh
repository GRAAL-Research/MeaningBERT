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
ALPHAS="${ALPHAS:-0.2 0.4 0.5 0.6 0.8}"
SEEDS="${SEEDS:-42 43 44}"

# backbone key|off-the-shelf checkpoint|remote folder of the English regime
BACKBONES="\
mdeberta|MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7|results/polarity/mdeberta-v3-base-nli-v4en
xlmr|joeddav/xlm-roberta-large-xnli|results/polarity/xlmr-large-nli-v4en"

cd "$REPO" || exit 1
mkdir -p "$OUT" "$SCRATCH"
py() { PYTHONPATH=src "$VENV/bin/python" "$@" < /dev/null; }

while IFS='|' read -r key base remote; do
    for seed in $SEEDS; do
        tuned="$SCRATCH/$key-seed$seed"
        for alpha in $ALPHAS; do
            name="v4-wise-$key-en-a$alpha-seed$seed"
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
            if py src/diagnostics/discovery_fr.py score --heads "wise-$key" --only "$name" \
                --out "$OUT/scores-$name.json"; then
                py src/diagnostics/discovery_fr.py report --scores "$OUT" --out "$OUT/report.json" \
                    || echo "[FAIL] rapport"
            else
                echo "[FAIL] $name : score"
            fi
            rm -rf "results/polarity/wise/$name"
        done
        rm -rf "$tuned"
    done
done <<< "$BACKBONES"

touch "$OUT/WISE_DONE"
echo "[$(date '+%F %T')] balayage WiSE-FT termine"
