#!/usr/bin/env bash
# Rapatrie les resultats de la grille depuis tous les workers vers le poste local.
#
# renard et souris ne partagent pas de systeme de fichiers : chaque machine ecrit ses
# JSON chez elle. Tant qu'ils ne sont pas rassembles, aucune analyse ne voit la grille
# entiere. Ce script les rassemble, sans jamais ecraser : les tranches sont disjointes,
# donc deux machines n'ecrivent jamais le meme fichier.
#
#   bash src/training/collect_results.sh            # rapatrie et analyse
#   ANALYSE=0 bash src/training/collect_results.sh  # rapatrie seulement
set -uo pipefail

LOCAL="${LOCAL:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
HOSTS="${HOSTS:-renard souris}"
REMOTE="${REMOTE:-MeaningBERT}"
ANALYSE="${ANALYSE:-1}"
PY="${PY:-$LOCAL/.venv/bin/python}"

mkdir -p "$LOCAL/results/grid" "$LOCAL/results/v2-runs"

for host in $HOSTS; do
    echo "--- $host"
    # Les JSON et les logs, pas les checkpoints : un checkpoint pese des centaines de Mo
    # et ne sert a rien pour l'analyse.
    rsync -az --include '*/' --include '*.json' --include '*.log' --exclude '*' \
        "$host:$REMOTE/results/grid/" "$LOCAL/results/grid/" 2>/dev/null \
        && echo "    grille rapatriee" || echo "    pas de grille sur $host"
    rsync -az --include '*/' --include '*.json' --include '*.log' --exclude '*' \
        "$host:$REMOTE/results/v2-runs/" "$LOCAL/results/v2-runs/" 2>/dev/null \
        && echo "    phase 1 rapatriee" || echo "    pas de phase 1 sur $host"
done

echo
echo "runs de phase 1 : $(find "$LOCAL/results/v2-runs" -name '*.json' | wc -l) / 8"
echo "runs de grille  : $(find "$LOCAL/results/grid" -name '*.json' | wc -l) / 20"
find "$LOCAL/results/grid" -name '*.json' | sed "s|$LOCAL/results/grid/||" | sort | sed 's/^/  /'

[ "$ANALYSE" = "1" ] || exit 0
[ -x "$PY" ] || { echo "pas d'interpreteur local en $PY, analyse sautee"; exit 0; }

echo
PYTHONPATH="$LOCAL/src" "$PY" "$LOCAL/src/figures_generator/analyze_v2_experiment.py" \
    --runs-dir "$LOCAL/results/v2-runs" --markdown-out "$LOCAL/results/RESULTATS.md"
PYTHONPATH="$LOCAL/src" "$PY" "$LOCAL/src/figures_generator/analyze_v2_experiment.py" \
    --runs-dir "$LOCAL/results/grid" --markdown-out "$LOCAL/results/RESULTATS-grille.md"
