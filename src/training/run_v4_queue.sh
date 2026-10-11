#!/usr/bin/env bash
# v4 polarity grid: one queue per GPU, seed-major across the three regimes.
#
#   GPU=1 QUEUE="en|xlmr-large-nli|42;fr|xlmr-large-nli|42;enfr|xlmr-large-nli|42" \
#     nohup bash src/training/run_v4_queue.sh renard-gpu1 &
#
# A queue item is "<regime>|<arch tag>|<seed>". The regime names the corpus:
#
#   en    datastore/polarity-full-lex       AUG+LEX in English, the V4-H2 rung; read in
#                                           French it is also the zero-shot regime
#   fr    datastore/polarity-full-lex-fr    the same corpus machine-translated to French
#   enfr  datastore/polarity-full-lex-enfr  both, the multilingual regime
#
# Seed-major order is the point: seed 42 of all three regimes lands first, so the first
# comparison exists after one pass instead of after thirty runs. An item whose corpus is
# not built yet waits for it rather than failing, since the translation runs on another
# host while the English rung trains.

set -u

NAME="${1:?usage: run_v4_queue.sh <worker-name>}"
REPO="${REPO:-$HOME/MeaningBERT-v3}"
POLL="${POLL:-300}"

declare -A CORPUS_OF=(
    [en]="$REPO/datastore/polarity-full-lex"
    [fr]="$REPO/datastore/polarity-full-lex-fr"
    [enfr]="$REPO/datastore/polarity-full-lex-enfr"
)

IFS=';' read -ra items <<< "${QUEUE:?QUEUE est vide}"
for item in "${items[@]}"; do
    [ -n "$item" ] || continue
    IFS='|' read -r regime tag seed <<< "$item"
    corpus="${CORPUS_OF[$regime]:-}"
    if [ -z "$corpus" ]; then
        echo "[FAIL] regime inconnu : $regime"
        continue
    fi
    until [ -f "$corpus/corpus/dataset_dict.json" ] && [ -f "$corpus/READY" ]; do
        echo "[WAIT] $regime : corpus $corpus pas encore pret, nouvel essai dans $POLL s"
        sleep "$POLL"
    done
    CONDITION="v4$regime" CORPUS="$corpus" CELLS="$tag:$seed" \
        bash "$REPO/src/training/run_polarity.sh" "$NAME-$regime-$seed"
done

echo "[$(date '+%F %T')] file $NAME terminee"
