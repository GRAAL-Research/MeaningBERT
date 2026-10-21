#!/usr/bin/env bash
# Resume the whole v4 campaign after a reboot, from cron:
#
#   @reboot sleep 120 && bash $HOME/MeaningBERT-v3/src/training/resume_v4.sh >> $HOME/MeaningBERT-v3/results/v4/resume.log 2>&1
#
# Renard and souris both rebooted on 2026-10-20 and stopped everything until a manual
# relaunch. Nothing here needs to know what was done: a cell whose metrics.json exists is
# skipped by run_polarity.sh, a head already scored is skipped by the daemons. What a reboot
# leaves behind is the claim of the cell that was training, which run_polarity.sh would
# respect for a day; no worker survives a reboot, so every claim is stale and goes first.

set -u

REPO="${REPO:-$HOME/MeaningBERT-v3}"
cd "$REPO" || exit 1
mkdir -p results/v4
echo "[$(date '+%F %T')] reprise sur $(hostname)"

launch() {
    local log="$1"
    shift
    setsid nohup "$@" >> "$log" 2>&1 < /dev/null &
}

# The full plan of each card, in the order it was queued. Done cells cost one SKIP line.
plan_gpu0() {
    local seed q=""
    for seed in 42 43 44 45 46 47 48 49 50 51; do
        q="$q;en|mdeberta-v3-base-nli|$seed;fr|mdeberta-v3-base-nli|$seed;enfr|mdeberta-v3-base-nli|$seed"
    done
    for seed in 42 43 44 45 46; do
        q="$q;replay|mdeberta-v3-base-nli|$seed;none|mdeberta-v3-base-nli|$seed;aug|mdeberta-v3-base-nli|$seed"
    done
    for seed in 42 43 44 45 46 47 48 49 50 51; do
        q="$q;mreplay|mdeberta-v3-base-nli|$seed"
        [ "$seed" -ge 47 ] && q="$q;replay|mdeberta-v3-base-nli|$seed;none|mdeberta-v3-base-nli|$seed;aug|mdeberta-v3-base-nli|$seed"
    done
    echo "${q#;}"
}

plan_gpu1() {
    local seed q=""
    for seed in 42 43 44 45 46 47 48 49 50 51; do
        q="$q;en|xlmr-large-nli|$seed;fr|xlmr-large-nli|$seed;enfr|xlmr-large-nli|$seed"
    done
    q="$q;replay|xlmr-large-nli|42;replay|xlmr-large-nli|43;replay|xlmr-large-nli|44"
    for seed in 42 43 44 45 46 47 48 49 50 51; do
        q="$q;mreplay|xlmr-large-nli|$seed"
        [ "$seed" -ge 45 ] && q="$q;replay|xlmr-large-nli|$seed"
        [ "$seed" -le 46 ] && q="$q;none|xlmr-large-nli|$seed;aug|xlmr-large-nli|$seed"
    done
    echo "${q#;}"
}

case "$(hostname)" in
    renard)
        find results/polarity -maxdepth 2 -name "seed*.claim" -type d -print -exec rm -rf {} +
        GPU=0 QUEUE="$(plan_gpu0)" launch results/v4/queue-gpu0-resume.log \
            bash src/training/run_v4_queue.sh renard-g0-resume
        GPU=1 QUEUE="$(plan_gpu1)" launch results/v4/queue-gpu1-resume.log \
            bash src/training/run_v4_queue.sh renard-gpu1-resume
        ;;
    souris)
        rm -rf results/polarity/wise/* results/v4/wise-tmp/* results/v4/xnli-tmp/* results/v4/scoring-tmp/*
        WAIT_TRANSLATION=0 launch results/v4/scorer.log bash src/training/score_v4_daemon.sh
        launch results/v4/wise.log bash src/training/wise_v4_daemon.sh
        launch results/v4/xnli.log bash src/training/xnli_v4_daemon.sh
        ;;
    *)
        echo "hote inconnu, rien a reprendre"
        ;;
esac
