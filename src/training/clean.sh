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
echo "Deleting all remote runs..."

# Use wandb API to delete all runs in the project
python3 -c "
import wandb
api = wandb.Api()
runs = api.runs('$ENTITY/$PROJECT')
print(f'Found {len(runs)} runs to delete')
for run in runs:
    print(f'  Deleting {run.name} ({run.id})')
    run.delete()
print('All remote runs deleted.')
"

echo ""
echo "=== Done ==="
