#!/usr/bin/env bash
# Relance la tranche d'un worker si elle n'est ni en cours ni terminee.
#
# Pour cron, toutes les dix minutes, pour qu'un run de plusieurs heures survive a un
# plantage, a un OOM du noyau ou a un redemarrage. Relancer ne coute rien : run_grid.sh
# saute toute variante dont le JSON existe deja.
#
#   */10 * * * * /home/dabea241/MeaningBERT/src/training/watchdog_worker.sh renard-gpu0
set -uo pipefail
REPO="${REPO:-$HOME/MeaningBERT}"
WORKER="${1:?usage: watchdog_worker.sh <nom-du-worker>}"
CONF="$REPO/env/workers/$WORKER.env"
MARKER="$REPO/results/WORKER_COMPLETE-$WORKER"
LOG="$REPO/results/watchdog-$WORKER.log"

cd "$REPO" 2>/dev/null || exit 0
mkdir -p results
[ -f "$MARKER" ] && exit 0
[ -f "$CONF" ] || { echo "$(date '+%F %T') : $CONF absent" >> "$LOG"; exit 0; }

# Un seul processus par worker. On cible le nom du worker et pas le nom du script, sinon
# les deux tranches d'une meme machine se prennent l'une pour l'autre.
if pgrep -f "run_worker.sh $WORKER\$" > /dev/null; then
    exit 0
fi

if ! nvidia-smi -L > /dev/null 2>&1; then
    {
        echo "--- $(date '+%F %T') : nvidia-smi muet, pilote probablement desaligne"
        echo "    comparer /proc/driver/nvidia/version et dpkg -l | grep nvidia-driver, puis redemarrer"
    } >> "$LOG"
    exit 0
fi

echo "--- $(date '+%F %T') : tranche $WORKER absente, relance" >> "$LOG"
setsid nohup bash "$REPO/src/training/run_worker.sh" "$WORKER" \
    >> "$REPO/results/worker-$WORKER.log" 2>&1 < /dev/null &
echo "    relancee, pid $!" >> "$LOG"
