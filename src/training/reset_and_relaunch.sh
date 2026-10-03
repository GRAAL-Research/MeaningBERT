#!/bin/bash
# reset_and_relaunch.sh
#
# Tue les processus d'entraînement en cours, nettoie wandb (failed + doublons),
# supprime les checkpoints locaux résiduels, puis relance les runs manquants
# sur 3 GPUs en parallèle.
#
# Usage:
#   bash reset_and_relaunch.sh            # exécution complète
#   bash reset_and_relaunch.sh --dry-run  # prévisualisation sans rien modifier

set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
    echo "=== MODE DRY RUN (aucune modification) ==="
    echo ""
fi

export WANDB_PROJECT="meaningbert-checkpoint-sweep"

VENV_NVRTC=$(python -c "import site; print(site.getsitepackages()[0])")/nvidia/cu13/lib
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:${VENV_NVRTC}"
echo "LD_LIBRARY_PATH: ajout de ${VENV_NVRTC}"
echo ""

# ---------------------------------------------------------------------------
# 1. Tuer les processus rerun_failed.py et few_shot_training.py en cours
# ---------------------------------------------------------------------------
echo "=== 1. Processus en cours ==="
for PATTERN in rerun_failed.py few_shot_training.py; do
    PIDS=$(pgrep -f "$PATTERN" || true)
    if [[ -z "$PIDS" ]]; then
        echo "  Aucun processus $PATTERN actif."
    else
        echo "  PIDs $PATTERN : $PIDS"
        if [[ $DRY_RUN -eq 0 ]]; then
            kill $PIDS
            echo "  Tués."
        else
            echo "  [dry-run] kill $PIDS"
        fi
    fi
done
[[ $DRY_RUN -eq 0 ]] && sleep 2
echo ""

# ---------------------------------------------------------------------------
# 2. Nettoyage wandb : failed/crashed + doublons finished
# ---------------------------------------------------------------------------
echo "=== 2. Nettoyage wandb ==="

if [[ $DRY_RUN -eq 1 ]]; then
    echo "  [dry-run] python cleanup.py --clean-wandb --dedup-wandb"
    python cleanup.py --clean-wandb --dedup-wandb
else
    python cleanup.py --clean-wandb --dedup-wandb --delete
fi
echo ""

# ---------------------------------------------------------------------------
# 3. Suppression des checkpoints locaux résiduels
#    (les dossiers meaning_bert_train_* sont partagés par modèle : un checkpoint
#     d'un fold crashé ferait reprendre le Trainer sur le mauvais fold)
# ---------------------------------------------------------------------------
echo "=== 3. Checkpoints locaux ==="

if [[ $DRY_RUN -eq 1 ]]; then
    echo "  [dry-run] python cleanup.py"
    python cleanup.py
else
    python cleanup.py --delete
fi
echo ""

# ---------------------------------------------------------------------------
# 4. Prévisualisation des runs à relancer (toujours affiché)
# ---------------------------------------------------------------------------
echo "=== 4. Runs à relancer (dry-run par GPU) ==="
for GPU in 0 1 2; do
    echo "--- GPU ${GPU} ---"
    python rerun_failed.py --gpu "${GPU}" --hardcoded --dry-run
    echo ""
done

# ---------------------------------------------------------------------------
# 5. Lancement sur 3 GPUs en parallèle
# ---------------------------------------------------------------------------
echo "=== 5. Lancement ==="

if [[ $DRY_RUN -eq 1 ]]; then
    echo "  [dry-run] Lancement non effectué."
    echo ""
    echo "Pour lancer : bash reset_and_relaunch.sh"
    exit 0
fi

nohup python rerun_failed.py --gpu 0 --hardcoded > rerun_gpu0.log 2>&1 &
PID0=$!
echo "  GPU 0 démarré (PID $PID0) → rerun_gpu0.log"

nohup python rerun_failed.py --gpu 1 --hardcoded > rerun_gpu1.log 2>&1 &
PID1=$!
echo "  GPU 1 démarré (PID $PID1) → rerun_gpu1.log"

nohup python rerun_failed.py --gpu 2 --hardcoded > rerun_gpu2.log 2>&1 &
PID2=$!
echo "  GPU 2 démarré (PID $PID2) → rerun_gpu2.log"

echo ""
echo "=== Tout est lancé ==="
echo ""
echo "Suivre la progression :"
echo "  tail -f rerun_gpu0.log rerun_gpu1.log rerun_gpu2.log"
echo ""
echo "Vérifier l'état wandb :"
echo "  python ../../figures_generator/analyze_sweep_results.py --progress_only"
echo ""
echo "Vérifier que les processus tournent :"
echo "  ps -p $PID0,$PID1,$PID2"
