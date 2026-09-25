#!/usr/bin/env bash
# The v2 grid on caribou, for the architectures renard cannot run.
#
# Why a second host. renard is a Quadro P5000, Pascal, compute capability 6.1. That rules
# out three things by hardware, not by configuration:
#
#   bf16            needs 8.0
#   torch.compile   needs Triton, which needs 7.0
#   FlashAttention  needs 8.0
#
# ModernBERT is built around FlashAttention and unpadded attention. It runs without them,
# but so slowly and with such different memory behaviour that a Pascal result would say
# nothing useful about the architecture. caribou (RTX 6000 Ada, compute 8.9) has all three.
#
# Everything else is held identical to renard: same corpus variants, same effective batch,
# same output head, same seed, same epoch budget. Only the precision differs, and that
# difference is the point.
#
#   bash src/training/run_grid_caribou.sh
#   VARIANTS="d_none" bash src/training/run_grid_caribou.sh
set -uo pipefail

REPO="${REPO:-$HOME/MeaningBERT}"
VENV="${VENV:-$HOME/.venvs/meaningbert-v2}"
DATA="${DATA:-$REPO/data/v2}"
ROOT="${ROOT:-$REPO/results/grid}"
EPOCHS="${EPOCHS:-10}"
PATIENCE="${PATIENCE:-3}"
EFFECTIVE_BATCH="${EFFECTIVE_BATCH:-32}"
HEAD="${HEAD:-clamped}"
SEED="${SEED:-42}"
VARIANTS="${VARIANTS:-c_none c_full d_none d_full}"
GPU="${GPU:-0}"

# tag|checkpoint|micro-batch. 49 GB per card, so no gradient accumulation is needed.
ALL_ARCHS="\
modernbert-base|answerdotai/ModernBERT-base|32
modernbert-large|answerdotai/ModernBERT-large|32"
ARCHS="${ARCHS:-}"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$REPO/src"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

die() { printf '\033[1;31mECHEC: %s\033[0m\n' "$*" >&2; exit 1; }

echo "=== Verification du materiel ==="
"$VENV/bin/python" - <<'PY' || die "l hote ne convient pas pour ModernBERT"
import sys
try:
    import torch
except ImportError:
    sys.exit("torch absent")
if not torch.cuda.is_available():
    sys.exit("aucune GPU visible")
major, minor = torch.cuda.get_device_capability(0)
name = torch.cuda.get_device_name(0)
print(f"  {name}, capacite {major}.{minor}, {torch.cuda.get_device_properties(0).total_memory / 1024**3:.0f} Go")
print(f"  torch {torch.__version__}, build CUDA {torch.version.cuda}")
if major < 8:
    sys.exit(
        f"capacite {major}.{minor} : ModernBERT exige 8.0 pour bf16 et FlashAttention. "
        "Ce script est fait pour caribou, pas pour renard."
    )
print("  bf16 : oui")
PY

echo
echo "=== Verification de transformers ==="
"$VENV/bin/python" - <<'PY' || die "transformers trop ancien pour ModernBERT"
import sys
from transformers import AutoConfig
try:
    AutoConfig.from_pretrained("answerdotai/ModernBERT-base")
except Exception as error:  # noqa: BLE001 - the message is the whole point
    sys.exit(f"ModernBERT introuvable ou non supporte : {error}\nIl faut transformers >= 4.48.")
print("  ModernBERT reconnu")
PY

echo
started=$(date +%s)
declare -a failed=()
while IFS='|' read -r tag checkpoint micro; do
    [ -z "${tag// }" ] && continue
    [ -n "$ARCHS" ] && ! echo " $ARCHS " | grep -q " $tag " && continue
    echo "--- $tag ($checkpoint)"
    dir="$ROOT/$tag"; mkdir -p "$dir"
    for variant in $VARIANTS; do
        json="$dir/$variant.json"; log="$dir/$variant.log"
        [ -s "$json" ] && { echo "[DONE] $tag/$variant"; continue; }
        [ -d "$DATA/$variant" ] || { echo "[SKIP] $tag/$variant : corpus absent"; continue; }
        accum=$(( EFFECTIVE_BATCH / micro )); [ "$accum" -lt 1 ] && accum=1
        echo "[RUN ] $tag/$variant  micro=$micro x accum=$accum  bf16"
        t0=$(date +%s)
        WANDB_PROJECT="meaningbert-v2-$tag" "$VENV/bin/python" "$REPO/src/training/few_shot_training.py" \
            --variant_path "$DATA/$variant" --checkpoint "$checkpoint" --output_head "$HEAD" \
            --num_epochs "$EPOCHS" --early_stopping_patience "$PATIENCE" \
            --per_device_train_batch_size "$micro" --gradient_accumulation_steps "$accum" \
            --bf16 --dataloader_num_workers 8 --seed "$SEED" --results_json "$json" > "$log" 2>&1
        status=$?
        dt=$(( $(date +%s) - t0 ))
        if [ $status -eq 0 ]; then
            echo "[ OK ] $tag/$variant en $((dt / 60)) min $((dt % 60)) s"
        else
            echo "[FAIL] $tag/$variant (code $status), voir $log"
            tail -8 "$log" | sed 's/^/       /'
            failed+=("$tag/$variant")
        fi
    done
done <<< "$ALL_ARCHS"

echo
echo "Total : $(( ($(date +%s) - started) / 60 )) min"
[ ${#failed[@]} -gt 0 ] && { echo "ECHECS : ${failed[*]}"; exit 1; }
echo "GRILLE CARIBOU TERMINEE"
echo
echo "Rapatrier les resultats vers renard ou la machine de travail :"
echo "  rsync -az caribou:$ROOT/ <destination>/results/grid/"
