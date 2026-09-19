#!/usr/bin/env bash
# Restart the v2 experiment if it is not running and not finished.
#
# Meant for cron, every ten minutes, so an unattended multi-day run survives a crash, an
# OOM kill by the OS, or a reboot. Restarting costs nothing: every runner skips a variant
# whose JSON already exists, so the chain resumes rather than starting over.
#
#   crontab -e
#   */10 * * * * /home/dabea241/MeaningBERT/src/training/watchdog.sh
set -uo pipefail
REPO="${REPO:-$HOME/MeaningBERT}"
MARKER="$REPO/results/EXPERIMENT_COMPLETE"
LOG="$REPO/results/watchdog.log"

cd "$REPO" 2>/dev/null || exit 0
mkdir -p results

# Finished. Nothing to do, and no log line either: a watchdog that writes every ten
# minutes for two days buries the one line that matters.
[ -f "$MARKER" ] && exit 0

if pgrep -f "bash src/training/run_everything.sh" > /dev/null; then
    exit 0
fi

# Not running. Either it never started, or it died. Either way, resume.
{
    echo "--- $(date '+%F %T') : chaine absente, relance"
    if ! nvidia-smi -L > /dev/null 2>&1; then
        echo "    nvidia-smi ne repond pas ; probablement un pilote desaligne apres mise a jour."
        echo "    Comparer /proc/driver/nvidia/version et dpkg -l | grep nvidia-driver, puis redemarrer."
        exit 0
    fi
} >> "$LOG" 2>&1

setsid nohup bash "$REPO/src/training/run_everything.sh" \
    >> "$REPO/results/everything.log" 2>&1 < /dev/null &
echo "    relancee, pid $!" >> "$LOG"
