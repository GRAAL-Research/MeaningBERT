#!/bin/bash
# Relaunch 100 failed runs across 3 GPUs in parallel.
# GPU 0: deberta-v3-small (20 runs)
# GPU 1: deberta-v3-base + deberta-v3-large (40 runs)
# GPU 2: deberta-v2-xlarge + phi-2 (40 runs)
export WANDB_PROJECT="meaningbert-checkpoint-sweep"

nohup python rerun_failed.py --gpu 0 --hardcoded > rerun_gpu0.log 2>&1 &
PID0=$!
echo "GPU 0 started (PID $PID0) -> rerun_gpu0.log"

nohup python rerun_failed.py --gpu 1 --hardcoded > rerun_gpu1.log 2>&1 &
PID1=$!
echo "GPU 1 started (PID $PID1) -> rerun_gpu1.log"

nohup python rerun_failed.py --gpu 2 --hardcoded > rerun_gpu2.log 2>&1 &
PID2=$!
echo "GPU 2 started (PID $PID2) -> rerun_gpu2.log"

echo ""
echo "All 3 GPUs running. Monitor with:"
echo "  tail -f rerun_gpu0.log rerun_gpu1.log rerun_gpu2.log"
echo ""
echo "Check completion with:"
echo "  ps -p $PID0,$PID1,$PID2"
