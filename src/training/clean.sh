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

# Use wandb Python API directly (more reliable than CLI parsing)
python3 -c "
import wandb

api = wandb.Api()
entity = api.default_entity
project = 'MeaningBERT'

if not entity:
    print('Not logged in to wandb. Run: wandb login')
    exit(0)

print(f'Wandb entity: {entity}, project: {project}')
print('Deleting all remote runs...')

runs = api.runs(f'{entity}/{project}')
print(f'Found {len(runs)} runs to delete')
for run in runs:
    print(f'  Deleting {run.name} ({run.id})')
    run.delete()
print('All remote runs deleted.')
"

echo ""
echo "=== Done ==="
