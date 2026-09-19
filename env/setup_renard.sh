#!/usr/bin/env bash
# Installe et verifie l'environnement d'entrainement v2 sur renard.
# Idempotent : relancer apres chaque mise a jour du serveur.
#
#   bash env/setup_renard.sh              # sync + install + verification
#   SKIP_SYNC=1 bash env/setup_renard.sh  # sans re-synchroniser le code
set -euo pipefail

HOST="${HOST:-renard}"
VENV="${VENV:-\$HOME/.venvs/meaningbert-v2}"
REPO_REMOTE="${REPO_REMOTE:-\$HOME/MeaningBERT}"
REPO_LOCAL="${REPO_LOCAL:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
GPU="${GPU:-1}"
INDEX="https://download.pytorch.org/whl/cu126"

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
die() { printf '\033[1;31mECHEC: %s\033[0m\n' "$*" >&2; exit 1; }

say "Acces a $HOST"
ssh -o BatchMode=yes -o ConnectTimeout=10 "$HOST" 'echo "  hote: $(hostname)"' || die "$HOST injoignable"

say "Pilote GPU"
if ! ssh "$HOST" 'nvidia-smi -L'; then
    cat >&2 <<'MSG'
  nvidia-smi ne repond pas.
  Cause frequente apres une mise a jour: le module noyau charge ne correspond plus aux
  bibliotheques userspace. Comparer sur l hote:
      cat /proc/driver/nvidia/version
      dpkg -l | grep nvidia-driver
  Si les versions different, redemarrer la machine.
MSG
    die "GPU indisponible"
fi

if [ "${SKIP_SYNC:-0}" != "1" ]; then
    say "Synchronisation du code vers $HOST:$REPO_REMOTE"
    # rsync plutot que git: la branche v2 n est pas poussee sur le depot public.
    rsync -az --delete \
        --exclude '.venv' --exclude '__pycache__' --exclude 'datastore/raw' \
        --exclude '.git/worktrees' --exclude 'wandb' \
        "$REPO_LOCAL/" "$HOST:$REPO_REMOTE/"
    echo "  synchronise depuis $REPO_LOCAL"
fi

say "Environnement virtuel"
ssh "$HOST" "
    set -e
    [ -x '$VENV/bin/python' ] || python3 -m venv '$VENV'
    '$VENV/bin/python' -m pip install --quiet --upgrade pip wheel
    echo \"  \$('$VENV/bin/python' -V) dans $VENV\"
"

say "Dependances (canal cu126, pour les noyaux sm_61)"
# On installe depuis renard-constraints.txt, pas renard-requirements.txt : les epingles
# exactes doivent etre resolues CONTRE la version de Python de l hote. Un fichier fige
# depuis une autre machine reference des versions qui n existent pas pour cette version
# de Python. Le resultat reel est ensuite gele dans env/renard-lock.txt.
ssh "$HOST" "
    set -e
    '$VENV/bin/pip' install \
        --index-url '$INDEX' --extra-index-url https://pypi.org/simple \
        -r '$REPO_REMOTE/env/renard-constraints.txt'
"

say "Gel du resultat reel dans env/renard-lock.txt"
ssh "$HOST" "'$VENV/bin/pip' freeze" > "$REPO_LOCAL/env/renard-lock.txt"
echo "  $(wc -l < "$REPO_LOCAL/env/renard-lock.txt") paquets geles"

say "Verification"
ssh "$HOST" "
    cd '$REPO_REMOTE' && CUDA_VISIBLE_DEVICES='$GPU' PYTHONPATH=src \
        '$VENV/bin/python' src/diagnostics/verify_training_env.py --gpu 0 \
        --json-out results/env-renard.json
" || die "l environnement n est pas utilisable pour l entrainement"

say "Pret. Lancer un entrainement avec :"
cat <<TXT
  ssh $HOST "cd $REPO_REMOTE && CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=src \\
      $VENV/bin/python src/training/few_shot_training.py --help"
TXT
