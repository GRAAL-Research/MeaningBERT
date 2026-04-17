---

# MeaningBERT - Multi-checkpoint sweep - 2026-04-12

## Contexte
Sweep systematique de 14 modeles pretrained (encoders + decoders) pour le fine-tuning de MeaningBERT (regression de preservation du sens). Objectif : trouver un meilleur backbone que bert-base-uncased et mesurer l'impact du data augmentation (swap commutative + back-translation EN->FR->EN).

## Ou en est-on
- [x] Refactor few_shot_training.py : --checkpoint, --fold, --data_augmentation, --learning_rate, --freeze_layers, bf16, early stopping
- [x] prepare_datasets.py : corpus merge (base+identical+unrelated=2042 dedup), k-fold stratifie, swap, back-translation GPU
- [x] validate_datasets.py : 0 leakage confirme sur tous les folds
- [x] 3 sweep scripts GPU (sweep_gpu0/1/2.sh) + launch_sweeps.sh
- [x] Wandb artifact logging du best model par run
- [x] cleanup.py pour nettoyer checkpoints + runs crashees
- [x] Fix : eval_strategy, processing_class, metrics squeeze, fp32 pour phi-2/gemma
- [ ] **Sweeps en cours** sur caribou (3x RTX 6000 Ada 49GB)
- [ ] **Relancer les 100 runs echoues via `rerun_failed.sh`** (deberta-v3-small/base/large + deberta-v2-xlarge + phi-2). Script pret dans `src/training/rerun_failed.sh`. Lancer avec `nohup bash rerun_failed.sh > rerun.log 2>&1 &` depuis `src/training/`. 2 causes :
  - 80 runs deberta-v2/v3 : nvrtc `libnvrtc-builtins.so.13.0` (installe 2026-04-14), bf16 ok
  - 20 runs phi-2 : backward dtype mismatch en fp32, fix = fp16 (`--fp16` ajoute au parser)
- [ ] Surveiller wandb : runs qui divergent, early stopping
- [ ] Notebook d'analyse post-sweep (R2, Pearson par checkpoint/aug/fold)
- [ ] Ensemble des meilleurs modeles
- [ ] Deploiement HuggingFace du meilleur modele

## Fichiers cles
- `src/training/few_shot_training.py` - script d'entrainement principal (14 modeles supportes)
- `src/training/prepare_datasets.py` - generation corpus (dedup, k-fold stratifie, swap, back-translation)
- `src/training/validate_datasets.py` - validation zero-leakage des folds
- `src/training/sweep_gpu0.sh` - GPU 0 : BERT, deberta-v3-small, gpt2, SmolLM2-135M, SmolLM2-360M
- `src/training/sweep_gpu1.sh` - GPU 1 : deberta-v3-base/large, electra-large, ModernBERT-large, Qwen2.5-0.5B
- `src/training/sweep_gpu2.sh` - GPU 2 : deberta-v2-xlarge, Qwen2.5-1.5B, gemma-2-2b (fp32), phi-2 (fp32)
- `src/training/launch_sweeps.sh` - lance les 3 GPUs en parallele
- `src/training/cleanup.py` - nettoyage checkpoints + wandb runs crashees
- `src/training/ensemble_evaluate.py` - evaluation ensemble post-sweep
- `src/training/metrics/metrics.py` - compute_metrics avec squeeze + nan_to_num

## Notes pour reprendre
- Serveur : caribou (dabea241), 3x RTX 6000 Ada 49GB, driver 595.58.03, CUDA 13.2, PyTorch cu126
- Branche : `experiment/multi-checkpoint-sweep`, PR #5
- Wandb projet : `davebulaval/meaningbert-checkpoint-sweep`
- phi-2 et gemma-2-2b : fp32 obligatoire (bf16 cause NaN), batch_size=4
- Qwen2.5-1.5B : 279 500 steps par run, surveiller early stopping
- Le cleanup auto des checkpoints intermediaires est dans few_shot_training.py (shutil.rmtree apres artifact upload)
- Les anciens sweep scripts (sweep_data_augmentation*.sh, sweep_no_data_augmentation.sh) sont obsoletes
- requirements.txt : torch>=2.6.0, transformers>=4.40.0, sentencepiece ajoute
