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

case "$(hostname)" in
    renard)
        find results/polarity -maxdepth 2 -name "seed*.claim" -type d -print -exec rm -rf {} +
        find results/magnitude-v4 -maxdepth 2 -name "seed*.claim" -type d -print -exec rm -rf {} + 2>/dev/null
        # The plan of each card lives in campaign_v4.sh, the one place it is written down.
        launch results/v4/campaign-gpu0.log bash src/training/campaign_v4.sh gpu0
        launch results/v4/campaign-gpu1.log bash src/training/campaign_v4.sh gpu1
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
