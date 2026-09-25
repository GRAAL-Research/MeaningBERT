# Serveur d'entrainement : renard

**Fichier genere.** Ne pas editer a la main. Regenerer avec :

```
PYTHONPATH=src python src/diagnostics/probe_training_host.py --host renard --gpu 1 \
    --json-out results/host-renard.json --markdown-out docs/serveur-releve-renard.md
```

## Releve

| | |
|---|---|
| Hote | renard |
| Pilote | 535.309.01 |
| Toolkit CUDA | 11.8 |
| Python | 3.11.2 |
| torch | absent |
| torch CUDA | n/a |
| CPU | 12 coeurs |
| RAM | 62 Go |
| Disque libre | 388G |

## GPU

| Index | Nom | Memoire | Capacite | bf16 | Coeurs tensoriels fp16 |
|---|---|---|---|---|---|
| 0 | NVIDIA GeForce GTX 1080 Ti | 11264 MiB | 6.1 | non | non |
| 1 **(retenue)** | Quadro P5000 | 16384 MiB | 6.1 | non | non |

## Precision

- bf16 disponible sur les GPU retenues : **non** (exige une capacite de calcul >= 8.0).
- Coeurs tensoriels fp16 : **non**.
- Precision retenue par defaut : **fp32**.

> Les scripts de sweep existants passent `--bf16`. Ils echoueront ou tomberont
> silencieusement en fp32 sur cette machine. A corriger avant tout entrainement.

## GPU a utiliser : index 1

```
CUDA_VISIBLE_DEVICES=1
```

## Ce qui rentre en fine-tuning

Estimation AdamW fp32, 16 octets par parametre pour poids,
gradients et moments, avec 35 % de la carte reservee aux
activations et au contexte CUDA. C'est un tri, pas une garantie.

| Checkpoint | Parametres | Etats | Verdict |
|---|---|---|---|
| microsoft/deberta-v3-small | 142 M | 2.1 Go | passe |
| microsoft/deberta-v3-base | 184 M | 2.7 Go | passe |
| microsoft/deberta-v3-large | 434 M | 6.5 Go | passe |
| microsoft/deberta-v2-xlarge | 900 M | 13.4 Go | **ne passe pas** |
| answerdotai/ModernBERT-large | 395 M | 5.9 Go | passe |
| bert-base-uncased | 110 M | 1.6 Go | passe |

## Rappel

Le backbone n'est pas le goulot de la v2. `docs/H1-diagnostic-calibration.md` etablit
que les quatre checkpoints du sweep convergent vers le meme plancher de RMSE apres
recalibrage affine, 22,1 a 23,1, et que le Pearson est plat a 0,78-0,80. Perdre un gros
checkpoint faute de memoire ne coute donc rien de scientifique.
