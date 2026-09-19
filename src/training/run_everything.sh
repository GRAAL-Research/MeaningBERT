#!/usr/bin/env bash
# Run the entire v2 experiment unattended, survive disconnection, resume after a crash.
#
# Nothing here depends on an interactive session. Launch with:
#
#   ssh renard 'cd ~/MeaningBERT && setsid nohup bash src/training/run_everything.sh \
#       > results/everything.log 2>&1 < /dev/null &'
#
# Resumability is the point. Every run writes a JSON when it finishes, and every runner
# skips a variant whose JSON already exists. Re-running this script after any interruption
# picks up where it stopped instead of starting over.
set -uo pipefail
REPO="${REPO:-$HOME/MeaningBERT}"
VENV="${VENV:-$HOME/.venvs/meaningbert-v2}"
cd "$REPO"

say() { printf '\n\033[1m===== %s  (%s) =====\033[0m\n' "$*" "$(date '+%a %H:%M')"; }
analyse() {
    PYTHONPATH="$REPO/src" "$VENV/bin/python" \
        src/figures_generator/analyze_v2_experiment.py --runs-dir results \
        --markdown-out results/RESULTATS.md 2>/dev/null || true
}

say "DEBUT"
echo "hote  : $(hostname)"
echo "GPU   : ${CUDA_VISIBLE_DEVICES:-1}"

# Phase 1. Conditions a and b exist only to isolate the source-sentence leak (H5) and the
# permuted labels (H6); they need one architecture, not five. The sigmoid head is kept
# here as the witness against which the clamped head is measured.
say "PHASE 1/2  deberta-v3-base, tete sigmoide, 8 variantes"
CHECKPOINT=microsoft/deberta-v3-base LOGS="$REPO/results/v2-runs" \
    MICRO_SMALL=32 MICRO_D=16 HEAD=sigmoid \
    bash src/training/run_v2_experiment.sh
echo "phase 1, code de sortie : $?"
analyse

# Phase 2. The grid proper: architecture x corpus x augmentation, clamped head, because
# the score domain is closed at both ends and roughly half the training labels sit exactly
# on an endpoint.
say "PHASE 2/2  grille clamped, cinq architectures"
HEADS=clamped bash src/training/run_grid.sh
echo "phase 2, code de sortie : $?"

say "TERMINE"
analyse
# The watchdog reads this marker and stops relaunching. Written only at the very end, so
# an interrupted chain is always resumed.
touch "$REPO/results/EXPERIMENT_COMPLETE"
echo
echo "Resultats : $REPO/results/RESULTATS.md"
cat results/RESULTATS.md 2>/dev/null | head -40
