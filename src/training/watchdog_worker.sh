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

# Un run fige n'est pas un run absent. Les 19 et 20 septembre, trois entrainements se sont
# arretes d'avancer en gardant leur processus vivant, a 100 % d'un coeur et la carte a
# 100 % : le chien de garde ne verifiait que la presence du processus, et le processus
# etait la. Quinze heures perdues la premiere fois, quatre la deuxieme.
#
# Le signe qui distingue les deux est l'age du journal. Un entrainement sain ecrit une
# ligne de barre de progression par seconde ; meme une phase d'evaluation ne reste pas
# muette plus d'une poignee de minutes. Au-dela de STALL_MINUTES sans une seule ecriture,
# le run est mort debout. On le tue, et la boucle du worker fait le reste : elle enchaine,
# et rien n'est perdu puisque le JSON n'est ecrit qu'a la fin d'un run reussi.
STALL_MINUTES="${STALL_MINUTES:-20}"

stall_check() {
    local pid log age
    for pid in $(pgrep -f "few_shot_training.py" 2>/dev/null); do
        # Le journal du run est le fichier vers lequel sa sortie est redirigee. On le
        # retrouve par le descripteur 1 du processus plutot que de deviner son chemin.
        log=$(readlink -f "/proc/$pid/fd/1" 2>/dev/null) || continue
        case "$log" in *.log) ;; *) continue ;; esac
        [ -f "$log" ] || continue
        age=$(( ($(date +%s) - $(stat -c %Y "$log")) / 60 ))
        if [ "$age" -ge "$STALL_MINUTES" ]; then
            {
                echo "--- $(date '+%F %T') : run fige, $age min sans ecriture dans $log"
                echo "    pid $pid tue ; le worker enchaine sur la suite de sa tranche"
            } >> "$LOG"
            # Trace laissee a cote du journal : run_grid.sh la lit avant de relancer la
            # cellule, et cesse d'y revenir au bout de trois blocages. Sans ce compteur,
            # une cellule qui bloque a tous les coups serait tuee et relancee toutes les
            # vingt minutes jusqu'a la fin des temps.
            date '+%F %T' >> "$log.stalled"
            kill -9 "$pid" 2>/dev/null
        fi
    done
}
stall_check

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
