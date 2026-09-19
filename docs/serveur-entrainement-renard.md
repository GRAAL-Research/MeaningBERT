# Serveur d'entrainement v2 : renard

Decide par David le 2026-09-19. **Remplace `caribou`** partout dans la documentation v2.
Releve materiel effectue le 2026-09-19.

## GPU a utiliser : **GPU 1, la Quadro P5000 uniquement**

Arbitre par David le 2026-09-19. Tout entrainement v2 s'execute avec :

```
CUDA_VISIBLE_DEVICES=1
```

La GTX 1080 Ti (GPU 0) n'est **pas** utilisee. Ne pas lancer de run multi-GPU.

## Materiel

| | |
|---|---|
| GPU 0 | NVIDIA GeForce GTX 1080 Ti, 11 264 MiB |
| GPU 1 | NVIDIA Quadro P5000, 16 384 MiB |
| Architecture | Pascal, capacite de calcul 6.1 |
| Pilote | 535.261.03 |
| Toolkit CUDA | 11.8 |
| CPU | 12 coeurs |
| RAM | 62 Go |
| Disque libre | 369 Go |
| Python systeme | 3.11.2 |
| uv, conda | absents |
| Depot MeaningBERT | absent, a cloner |

## Les deux contraintes qui changent le plan

### 1. Pas de bf16

Le bf16 exige Ampere, capacite 8.0 ou plus. Pascal ne l'a pas. **Tous les scripts de
sweep existants passent `--bf16` et echoueront ou tomberont silencieusement en fp32.**

Le fp16 est disponible mais Pascal n'a pas de coeurs tensoriels fp16 : le gain est
marginal et le risque d'instabilite numerique est reel, d'autant que le sweep a deja
montre des divergences en NaN sur `deberta-v3-large`. Voir
`docs/H1-diagnostic-calibration.md`.

**Defaut retenu : fp32.** Le fp16 avec `GradScaler` reste une option a mesurer, pas un
point de depart.

### 2. Memoire

Estimation AdamW en fp32, soit environ 16 octets par parametre pour poids, gradients et
etats de l'optimiseur, avant les activations.

| Checkpoint | Parametres (embeddings comprises) | Etats | Verdict sur renard |
|---|---|---|---|
| deberta-v3-small | ~142 M | ~2,3 Go | passe partout |
| deberta-v3-base | ~184 M | ~3,0 Go | passe partout |
| deberta-v3-large | ~434 M | ~7,0 Go | passe confortablement sur les 16 Go de la P5000 |
| deberta-v2-xlarge | ~900 M | ~14,4 Go | **ne passe pas** : 14,4 Go d'etats sur 16 Go, aucune marge pour les activations |

Le sweep d'origine tournait sur 3x RTX 6000 Ada de 49 Go. On perd `deberta-v2-xlarge`.

## Pourquoi ca ne bloque pas la v2

`docs/H1-diagnostic-calibration.md` etablit que les quatre checkpoints convergent vers le
**meme plancher de RMSE apres recalibrage affine, 22,1 a 23,1**, et que le Pearson est
plat a 0,78-0,80 quel que soit le backbone. Le backbone n'est pas le goulot de la v2, la
donnee l'est. Perdre `deberta-v2-xlarge` ne coute donc rien de scientifique.

Plan d'entrainement retenu :

- **Ablations H2** (CSMD seul, puis CSMD plus chaque corpus un a un) sur
  **deberta-v3-base**. C'est rapide, ca tient sur les deux GPU, et c'est suffisant pour
  comparer des corpus entre eux.
- **Modele final v2** sur **deberta-v3-large**, sur la P5000. Les 16 Go suffisent sans
  gradient checkpointing a batch modere ; l'activer si l'OOM se presente.
- `deberta-v2-xlarge` et `deberta-v3-small` sortent du plan v2.

Consequence sur le debit : Pascal en fp32 tourne autour de 9 a 11 TFLOPS contre plus de
90 en bf16 sur une RTX 6000 Ada. Compter environ un ordre de grandeur de plus par run.
Le sweep de 80 runs n'est pas reproductible ici ; la v2 vise un petit nombre de runs
cibles, pas un balayage.

## A faire avant le premier entrainement

1. Cloner le depot sur renard et se placer sur la branche v2.
0. Exporter `CUDA_VISIBLE_DEVICES=1` dans tout script de lancement.
2. Installer un environnement Python avec `torch` compile pour CUDA 11.8 ou 12.1.
   `uv` n'est pas installe sur renard.
3. Retirer `--bf16` des scripts de sweep, ou le rendre conditionnel a la capacite de
   calcul detectee.
4. Verifier que les corrections C1 a C3 de `docs/H1-diagnostic-calibration.md` sont en
   place. Lancer un entrainement sans elles reproduirait la compression d'amplitude.
