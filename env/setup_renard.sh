#!/usr/bin/env bash
# Cree l'environnement d'entrainement v2 sur renard et le verifie.
# Idempotent : relancer apres chaque mise a jour du serveur.
set -euo pipefail

HOST="${HOST:-renard}"
VENV="${VENV:-\$HOME/.venvs/meaningbert-v2}"
REPO="${REPO:-\$HOME/MeaningBERT}"
BRANCH="${BRANCH:-v2/integration}"
GPU="${GPU:-1}"
INDEX="https://download.pytorch.org/whl/cu126"

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

say "Verification de l'acces a $HOST"
ssh -o BatchMode=yes -o ConnectTimeout=10 "$HOST" 'echo "  hote: $(hostname)"'

say "Verification du pilote GPU"
if ! ssh "$HOST" 'nvidia-smi -L' 2>/dev/null; then
    echo "  ECHEC: nvidia-smi ne repond pas." >&2
    echo "  Cause frequente apres une mise a jour: le module noyau charge ne correspond" >&2
    echo "  plus aux bibliotheques userspace. Comparer:" >&2
    echo "    cat /proc/driver/nvidia/version" >&2
    echo "    dpkg -l | grep nvidia-driver" >&2
    echo "  Si les versions different, redemarrer la machine." >&2
    exit 1
fi

say "Depot"
ssh "$HOST" "
    set -e
    if [ ! -d '$REPO/.git' ]; then
        echo '  ERREUR: $REPO absent. Cloner le depot d abord.' >&2
        exit 1
    fi
    cd '$REPO' && git fetch --all --quiet && git checkout '$BRANCH' --quiet && git pull --quiet
    echo \"  branche: \$(git rev-parse --abbrev-ref HEAD) @ \$(git rev-parse --short HEAD)\"
"

say "Environnement virtuel"
ssh "$HOST" "
    set -e
    if [ ! -x '$VENV/bin/python' ]; then
        python3 -m venv '$VENV'
        echo '  cree: $VENV'
    else
        echo '  existant: $VENV'
    fi
    '$VENV/bin/python' -m pip install --quiet --upgrade pip wheel
    echo \"  python: \$('$VENV/bin/python' -V)\"
"

say "Installation des dependances epinglees (canal cu126, pour les noyaux sm_61)"
ssh "$HOST" "
    set -e
    '$VENV/bin/pip' install \
        --index-url '$INDEX' \
        --extra-index-url https://pypi.org/simple \
        -r '$REPO/env/renard-requirements.txt'
"

say "Verification de l'environnement"
ssh "$HOST" "
    cd '$REPO' && CUDA_VISIBLE_DEVICES='$GPU' PYTHONPATH=src \
        '$VENV/bin/python' src/diagnostics/verify_training_env.py --gpu 0
"

say "Termine. Lancer un entrainement avec :"
cat <<TXT
  ssh $HOST "cd $REPO && CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=src \\
      $VENV/bin/python src/training/few_shot_training.py --help"
TXT
