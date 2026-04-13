#!/bin/bash
# Clean local training artifacts and wandb runs
set -e

echo "=== Cleaning local artifacts ==="

# Remove saved model directories
rm -rf meaningbert_best_model_*
echo "Removed meaningbert_best_model_* directories"

# Remove training output directory
rm -rf meaning_bert_train
echo "Removed meaning_bert_train directory"

# Remove wandb local logs
rm -rf wandb
echo "Removed wandb/ local logs"

# Remove __pycache__
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
echo "Removed __pycache__ directories"

echo ""
echo "=== Cleaning wandb remote runs ==="

# Check if wandb is available
if ! command -v wandb &> /dev/null; then
    echo "wandb CLI not found. Install with: pip install wandb"
    echo "Skipping remote cleanup."
    exit 0
fi

# List wandb project runs (adjust entity/project as needed)
ENTITY=$(wandb whoami 2>/dev/null | head -1 | awk '{print $1}' || echo "")
PROJECT="MeaningBERT"

if [ -z "$ENTITY" ]; then
    echo "Not logged in to wandb. Run: wandb login"
    echo "Skipping remote cleanup."
    exit 0
fi

echo "Wandb entity: $ENTITY, project: $PROJECT"
echo "To delete all runs in the project, run:"
echo "  wandb runs delete $ENTITY/$PROJECT --all"
echo ""
echo "Or delete specific runs via the wandb UI: https://wandb.ai/$ENTITY/$PROJECT"

echo ""
echo "=== Done ==="
