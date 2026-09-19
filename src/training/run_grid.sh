#!/usr/bin/env bash
# The v2 experiment grid: architecture x corpus x augmentation.
#
# The three factors, and why each is there:
#
#   architecture  several encoders, starting with bert-base-uncased because that is what
#                 MeaningBERT v1 actually shipped. Without it, "v2 beats v1" would compare
#                 two different models rather than two different corpora.
#   corpus        c = CSMD v1 with the H6 correction, d = the four v2 corpora. Both on the
#                 grouped split, so the comparison is about the corpus and nothing else.
#   augmentation  none, or the three together (swap, back-translation, generated pairs).
#
# Conditions a and b are NOT in the grid. They are the diagnostics that isolate the
# source-sentence leak (H5) and the permuted labels (H6), and they only need to run once,
# on the reference architecture. Putting them in the grid would multiply their cost by the
# number of architectures for no extra information.
#
#   bash src/training/run_grid.sh
#   ARCHS=bert VARIANTS="d_none d_full" bash src/training/run_grid.sh
set -uo pipefail

REPO="${REPO:-$HOME/MeaningBERT}"
VENV="${VENV:-$HOME/.venvs/meaningbert-v2}"
DATA="${DATA:-$REPO/data/v2}"
ROOT="${ROOT:-$REPO/results/grid}"
EPOCHS="${EPOCHS:-10}"
PATIENCE="${PATIENCE:-3}"
EFFECTIVE_BATCH="${EFFECTIVE_BATCH:-32}"
HEADS="${HEADS:-clamped sigmoid}"
SEED="${SEED:-42}"
VARIANTS="${VARIANTS:-c_none c_full d_none d_full}"
# Deux workers sur la meme machine se disputent les coeurs : 6 chargeurs chacun sur
# 12 coeurs sature la machine et ralentit les deux. Reglable par worker.
NUM_WORKERS="${NUM_WORKERS:-6}"

# tag|checkpoint|micro-batch for c|micro-batch for d
#
# The d rung truncates at 512 tokens where c tops out at 209, so it takes a smaller
# micro-batch. The effective batch stays EFFECTIVE_BATCH everywhere through gradient
# accumulation, so optimisation is comparable across the whole grid.
# Ordered by information per hour, not by size. bert-base-uncased comes first because it
# is what MeaningBERT v1 published: without it, "v2 beats v1" compares two different
# models. deberta-v3-large comes last because it costs roughly three times the rest and
# answers the least interesting question.
ALL_ARCHS="\
bert|bert-base-uncased|32|16
deberta-v3-base|microsoft/deberta-v3-base|32|16
nli-deberta-v3-base|MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli|32|16
stsb-roberta-base|cross-encoder/stsb-roberta-base|32|16
deberta-v3-large|microsoft/deberta-v3-large|8|8"

ARCHS="${ARCHS:-}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTHONPATH="$REPO/src"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Garde-fou de partitionnement. Quand la grille est repartie entre plusieurs GPU (voir
# env/workers/ et src/training/run_worker.sh), un appel SANS ARCHS refait la grille
# entiere sur une seule carte, en double de ce que les autres workers sont deja en train
# de calculer. C'est exactement ce que ferait la phase 2 de run_everything.sh, qui a ete
# ecrite quand il n'y avait qu'un GPU. On refuse plutot que de doubler le travail.
if [ -z "$ARCHS" ] && [ -f "$REPO/results/GRID_PARTITIONED" ]; then
    echo "Grille partitionnee entre plusieurs workers (results/GRID_PARTITIONED existe)."
    echo "Un appel sans ARCHS refera tout en double : refuse."
    echo "Lancer une tranche : bash src/training/run_worker.sh <nom-du-worker>"
    exit 0
fi

started=$(date +%s)
declare -a failed=()

run_one() {
    local tag="$1" checkpoint="$2" variant="$3" micro="$4" head="$5"
    local dir="$ROOT/$tag-$head" accum=$(( EFFECTIVE_BATCH / micro ))
    [ "$accum" -lt 1 ] && accum=1
    mkdir -p "$dir"
    local json="$dir/$variant.json" log="$dir/$variant.log"
    local HEAD="$head"

    [ -s "$json" ] && { echo "[DONE] $tag-$head/$variant"; return 0; }
    [ -d "$DATA/$variant" ] || { echo "[SKIP] $tag-$head/$variant : corpus absent"; return 0; }

    echo "[RUN ] $tag-$head/$variant  micro=$micro x accum=$accum"
    local t0=$(date +%s)
    WANDB_PROJECT="meaningbert-v2-$tag-$head" "$VENV/bin/python" "$REPO/src/training/few_shot_training.py" \
        --variant_path "$DATA/$variant" --checkpoint "$checkpoint" --output_head "$HEAD" \
        --num_epochs "$EPOCHS" --early_stopping_patience "$PATIENCE" \
        --per_device_train_batch_size "$micro" --gradient_accumulation_steps "$accum" \
        --dataloader_num_workers "$NUM_WORKERS" --seed "$SEED" --results_json "$json" > "$log" 2>&1
    local status=$?

    # Only an out-of-memory kill earns a second try: halving the micro-batch changes the
    # memory, not the effective batch. Any other failure is a bug and must surface.
    if [ $status -ne 0 ] && grep -qi "out of memory" "$log"; then
        local retry=$(( micro / 4 )); [ "$retry" -lt 1 ] && retry=1
        echo "       OOM a $micro, nouvelle tentative a $retry"
        accum=$(( EFFECTIVE_BATCH / retry )); [ "$accum" -lt 1 ] && accum=1
        WANDB_PROJECT="meaningbert-v2-$tag-$head" "$VENV/bin/python" "$REPO/src/training/few_shot_training.py" \
            --variant_path "$DATA/$variant" --checkpoint "$checkpoint" --output_head "$HEAD" \
            --num_epochs "$EPOCHS" --early_stopping_patience "$PATIENCE" \
            --per_device_train_batch_size "$retry" --gradient_accumulation_steps "$accum" \
            --dataloader_num_workers "$NUM_WORKERS" --seed "$SEED" --results_json "$json" > "$log" 2>&1
        status=$?
    fi

    local dt=$(( $(date +%s) - t0 ))
    if [ $status -eq 0 ]; then
        echo "[ OK ] $tag-$head/$variant en $((dt / 60)) min $((dt % 60)) s"
    else
        echo "[FAIL] $tag-$head/$variant (code $status), voir $log"
        tail -6 "$log" | sed 's/^/       /'
        failed+=("$tag-$head/$variant")
    fi
}

echo "Grille : architecture x corpus x augmentation x tete de sortie"
echo "variantes : $VARIANTS"
echo "tetes     : $HEADS"
echo
echo "La tete compte parce que le domaine est [0, 100] FERME : un score nul existe et"
echo "signifie qu'aucun sens n'est preserve. 100*sigmoid a pour image l'intervalle OUVERT"
echo "(0, 100), donc ni 0 ni 100 n'y sont jamais atteints. Or 31 a 48 % des etiquettes"
echo "d'entrainement valent exactement 0 ou exactement 100. La tete clamped atteint les"
echo "bornes ; la sigmoide est gardee comme temoin."
echo
while IFS='|' read -r tag checkpoint micro_c micro_d; do
    [ -z "${tag// }" ] && continue
    [ -n "$ARCHS" ] && ! echo " $ARCHS " | grep -q " $tag " && continue
    echo "--- $tag ($checkpoint)"
    for head in $HEADS; do
        for variant in $VARIANTS; do
            case "$variant" in d_*) micro="$micro_d" ;; *) micro="$micro_c" ;; esac
            run_one "$tag" "$checkpoint" "$variant" "$micro" "$head"
        done
    done
done <<< "$ALL_ARCHS"

echo
echo "Total grille : $(( ($(date +%s) - started) / 60 )) min"
[ ${#failed[@]} -gt 0 ] && { echo "ECHECS : ${failed[*]}"; exit 1; }
echo "GRILLE TERMINEE"
