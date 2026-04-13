#!/bin/bash
# Launch all 3 GPU sweeps in parallel. Run from src/training/.
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

pkill -f sweep_gpu 2>/dev/null
sleep 1

nohup bash sweep_gpu0.sh > sweep_gpu0.log 2>&1 &
nohup bash sweep_gpu1.sh > sweep_gpu1.log 2>&1 &
nohup bash sweep_gpu2.sh > sweep_gpu2.log 2>&1 &

echo "Launched 3 GPU sweeps (PIDs: $(jobs -p | tr '\n' ' '))"
echo "Monitor: tail -f sweep_gpu0.log sweep_gpu1.log sweep_gpu2.log"
