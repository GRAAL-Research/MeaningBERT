#!/usr/bin/env bash
# Un worker = un GPU = une tranche de la grille.
#
# La grille est sequentielle par construction : un GPU, un entrainement a la fois. Le seul
# moyen d'aller plus vite est donc d'ajouter des GPU, et le seul moyen d'ajouter des GPU
# sans faire deux fois le meme travail est de partitionner la grille par architecture.
# Chaque worker recoit sa liste d'architectures et n'y touche que celles-la.
#
# Le partitionnement est STATIQUE et non dynamique parce que les machines ne partagent pas
# de systeme de fichiers : renard et souris ne peuvent pas se lire un verrou mutuel. Une
# tranche par worker, aucune coordination, aucune collision possible.
#
#   bash src/training/run_worker.sh renard-gpu0
#
# Le nom du worker est passe en ARGUMENT et non seulement par l'environnement : il doit
# apparaitre dans la ligne de commande, sinon le chien de garde ne peut pas distinguer
# les deux tranches d'une meme machine et en relance une par-dessus l'autre.
set -uo pipefail

REPO="${REPO:-$HOME/MeaningBERT}"
VENV="${VENV:-$HOME/.venvs/meaningbert-v2}"
WORKER="${1:-${WORKER:-}}"
[ -n "$WORKER" ] || { echo "usage: run_worker.sh <nom-du-worker>" >&2; exit 2; }
# Le fichier de tranche porte GPU, ARCHS et le nombre de chargeurs. Passer par lui
# garantit qu'une relance par cron reprend exactement la meme tranche.
CONF="${CONF:-$REPO/env/workers/$WORKER.env}"
if [ -f "$CONF" ]; then set -a; . "$CONF"; set +a; fi
GPU="${GPU:-0}"                     # index physique, celui que voit nvidia-smi -i
# Une tranche se decrit par des LOTS separes par ";", chaque lot etant
#   "<architectures>|<variantes>"   (variantes vide = les quatre de la grille).
# Le lot existe parce que deberta-v3-large ne se partitionne pas par architecture : il
# est seul et coute trois fois le reste, donc il se partitionne par rang de corpus. Le
# rang d (512 jetons) va sur la carte de 16 Go, le rang c (209 jetons) ailleurs.
SLICES="${SLICES:-${ARCHS:+$ARCHS|}}"
[ -n "$SLICES" ] || { echo "SLICES ou ARCHS doit decrire la tranche" >&2; exit 2; }
HEADS="${HEADS:-clamped}"
NUM_WORKERS="${NUM_WORKERS:-6}"
WAIT_FOR_GPU="${WAIT_FOR_GPU:-1}"   # attendre que la carte soit libre avant de demarrer

cd "$REPO" || exit 1
mkdir -p results
MARKER="$REPO/results/WORKER_COMPLETE-$WORKER"

say() { printf '\n\033[1m===== %s  (%s) =====\033[0m\n' "$*" "$(date '+%a %H:%M')"; }

[ -f "$MARKER" ] && { echo "tranche $WORKER deja terminee"; exit 0; }

# Une carte occupee par un autre entrainement n'a pas la memoire pour un second : sur 11 a
# 16 Go, deux runs concurrents finissent en OOM tous les deux. On attend plutot que de
# tuer le voisin.
if [ "$WAIT_FOR_GPU" = "1" ]; then
    while :; do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" 2>/dev/null)
        [ -z "$used" ] && break
        [ "$used" -lt 1000 ] && break
        echo "$(date '+%H:%M') GPU $GPU occupe (${used} Mio), attente"
        sleep 120
    done
fi

say "WORKER $WORKER  GPU $GPU"
echo "tranche     : $SLICES"
echo "hote        : $(hostname)"
echo "tetes       : $HEADS"
echo "chargeurs   : $NUM_WORKERS"

status=0
IFS=';' read -r -a lots <<< "$SLICES"
for lot in "${lots[@]}"; do
    [ -z "${lot// }" ] && continue
    archs="${lot%%|*}"
    variants="${lot#*|}"
    [ "$variants" = "$lot" ] && variants=""
    echo
    echo ">>> lot : archs=[$archs] variantes=[${variants:-defaut}]"
    # VARIANTS s'exporte plutot que de se prefixer a la commande : la liste contient des
    # espaces, et un prefixe non quote la ferait eclater en plusieurs mots.
    if [ -n "$variants" ]; then export VARIANTS="$variants"; else unset VARIANTS; fi
    CUDA_VISIBLE_DEVICES="$GPU" ARCHS="$archs" HEADS="$HEADS" NUM_WORKERS="$NUM_WORKERS" \
        REPO="$REPO" VENV="$VENV" bash "$REPO/src/training/run_grid.sh"
    lot_status=$?
    [ $lot_status -ne 0 ] && status=$lot_status
done
echo "code de sortie de la tranche : $status"

# Le marqueur n'est ecrit que si toute la tranche est passee. Une tranche en echec reste
# relancable par le chien de garde, et la relance saute ce qui est deja fait.
if [ $status -eq 0 ]; then
    touch "$MARKER"
    say "TRANCHE $WORKER TERMINEE"
else
    say "TRANCHE $WORKER INCOMPLETE (code $status)"
fi
exit $status
