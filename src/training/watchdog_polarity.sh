#!/usr/bin/env bash
# Keeps the experiment 3 grid alive over the days it takes, unattended.
#
# Installed as a cron entry, every ten minutes:
#   */10 * * * * bash $HOME/MeaningBERT-v3/src/training/watchdog_polarity.sh <driver> >> ... 2>&1
#
# It handles the three ways the v2 campaign lost a night, and nothing else. A watchdog that
# tries to be clever is a second thing that can break at three in the morning.

set -u

DRIVER="${1:?usage: watchdog_polarity.sh <chemin du script pilote>}"
REPO="${REPO:-$HOME/MeaningBERT-v3}"
ROOT="${ROOT:-$REPO/results/polarity}"
STALL_MINUTES="${STALL_MINUTES:-45}"
STAMP=$(date '+%F %T')

# 1. The hang. A wedged run keeps the GPU at 100 % and never writes another line; that is
#    how bert froze the 1080 Ti three times in v2 and once more today. Detection is by log
#    age and not by GPU usage, because a healthy run also sits at 100 %. Killing it makes
#    run_polarity.sh record the failure and move to the next cell, which is what we want:
#    one wedged cell must not cost the remaining days.
stall_check() {
    local pid log age
    for pid in $(pgrep -f "train_polarity.py" 2>/dev/null); do
        log=$(readlink -f "/proc/$pid/fd/1" 2>/dev/null) || continue
        case "$log" in *.log) ;; *) continue ;; esac
        age=$(( ( $(date +%s) - $(stat -c %Y "$log" 2>/dev/null || date +%s) ) / 60 ))
        if [ "$age" -ge "$STALL_MINUTES" ]; then
            echo "[$STAMP] BLOQUE depuis ${age} min, on tue : $log"
            echo "$STAMP bloque a ${age} min" >> "$log.stalled"
            kill -9 "$pid" 2>/dev/null
        fi
    done
}

# 2. The driver dying, from a reboot or an OOM killer. Restarting it is safe because
#    run_polarity.sh skips a cell that already has its metrics.json and claims the one it
#    starts, so nothing is recomputed and nothing is run twice.
driver_check() {
    local name; name=$(basename "$DRIVER")
    if ! pgrep -f "$name" > /dev/null; then
        if [ -f "$ROOT/.grid_done" ]; then
            return 0
        fi
        echo "[$STAMP] pilote $name absent, on relance"
        cd "$REPO" || return 1
        setsid nohup bash "$DRIVER" >> "$REPO/results/driver-$name.log" 2>&1 < /dev/null &
    fi
}

# 3. A claim left behind by a worker killed with -9, which never ran its trap. Without this
#    the cell is skipped forever and the grid quietly finishes with a hole in it.
claim_check() {
    local claim age
    for claim in "$ROOT"/*/*.claim; do
        [ -d "$claim" ] || continue
        age=$(( ( $(date +%s) - $(stat -c %Y "$claim") ) / 60 ))
        if [ "$age" -ge 120 ] && ! pgrep -f "$(basename "${claim%.claim}")" > /dev/null; then
            echo "[$STAMP] reclamation orpheline de ${age} min, liberee : $claim"
            rmdir "$claim" 2>/dev/null
        fi
    done
}

stall_check
claim_check
driver_check
