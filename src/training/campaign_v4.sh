#!/usr/bin/env bash
# The whole v4 campaign of one renard card, in order: polarity grid, magnitude benchmark,
# second wave. One process per card, so two queues never share a GPU.
#
#   setsid nohup bash src/training/campaign_v4.sh gpu0 >> results/v4/campaign-gpu0.log 2>&1 &
#
# Every segment skips what is already done (metrics.json for a polarity cell, the results
# JSON for a magnitude run), so the same command resumes after a reboot (resume_v4.sh) and
# costs one SKIP line per finished cell. The magnitude runs are short next to the polarity
# ones, so the first seed of each corpus goes early and gives a first reading within a day.

set -u

CARD="${1:?usage: campaign_v4.sh gpu0|gpu1}"
REPO="${REPO:-$HOME/MeaningBERT-v3}"
cd "$REPO" || exit 1

seeds() { seq "$1" "$2"; }

polarity() {
    local name="$1" queue="$2"
    QUEUE="${queue#;}" bash src/training/run_v4_queue.sh "$name"
}

magnitude() {
    local name="$1" tag="$2" first="$3" last="$4" seed corpus q=""
    for seed in $(seeds "$first" "$last"); do
        for corpus in en fr enfr; do q="$q;$corpus|$tag|$seed"; done
    done
    QUEUE="${q#;}" bash src/training/run_magnitude_queue.sh "$name"
}

case "$CARD" in
    gpu0)
        export GPU=0
        arch=mdeberta-v3-base-nli
        q=""
        for seed in $(seeds 42 51); do q="$q;en|$arch|$seed;fr|$arch|$seed;enfr|$arch|$seed"; done
        polarity renard-g0-grille "$q"
        magnitude renard-g0-magnitude mdeberta 42 42
        q=""
        for seed in $(seeds 42 46); do q="$q;replay|$arch|$seed;none|$arch|$seed;aug|$arch|$seed"; done
        polarity renard-g0-vague2 "$q"
        magnitude renard-g0-magnitude mdeberta 43 46
        q=""
        for seed in $(seeds 42 51); do
            q="$q;mreplay|$arch|$seed"
            [ "$seed" -ge 47 ] && q="$q;replay|$arch|$seed;none|$arch|$seed;aug|$arch|$seed"
        done
        polarity renard-g0-vague2b "$q"
        ;;
    gpu1)
        export GPU=1
        arch=xlmr-large-nli
        q=""
        for seed in $(seeds 42 51); do q="$q;en|$arch|$seed;fr|$arch|$seed;enfr|$arch|$seed"; done
        polarity renard-gpu1-grille "$q"
        magnitude renard-gpu1-magnitude xlmr 42 42
        polarity renard-gpu1-replay "replay|$arch|42;replay|$arch|43;replay|$arch|44"
        magnitude renard-gpu1-magnitude xlmr 43 46
        q=""
        for seed in $(seeds 42 51); do
            q="$q;mreplay|$arch|$seed"
            [ "$seed" -ge 45 ] && q="$q;replay|$arch|$seed"
            [ "$seed" -le 46 ] && q="$q;none|$arch|$seed;aug|$arch|$seed"
        done
        polarity renard-gpu1-vague2b "$q"
        ;;
    *)
        echo "carte inconnue : $CARD"
        exit 1
        ;;
esac
echo "[$(date '+%F %T')] campagne $CARD terminee"
