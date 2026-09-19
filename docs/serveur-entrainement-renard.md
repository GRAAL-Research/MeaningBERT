# Plan d'entrainement v2 : renard

Ce fichier porte les **decisions**. Le **releve materiel** vit dans
`docs/serveur-releve-renard.md`, qui est genere et ne doit pas etre edite a la main.

## Decisions

| Point | Decision | Date |
|---|---|---|
| Serveur d'entrainement | `renard`, et non `caribou` | 2026-09-19, David |
| GPU | **GPU 1 uniquement, la Quadro P5000**. `CUDA_VISIBLE_DEVICES=1` | 2026-09-19, David |
| Multi-GPU | interdit | 2026-09-19 |
| Precision | ce que dit le releve genere. Aujourd'hui `fp32`, faute de bf16 | derive |
| Ablations H2 | `deberta-v3-base` | 2026-09-19 |
| Modele final v2 | `deberta-v3-large` | 2026-09-19 |
| Hors plan v2 | `deberta-v2-xlarge` (ne rentre pas), `deberta-v3-small` | 2026-09-19 |

## Le serveur va bouger

David met a jour renard dans une autre session : version de Python, paquets, pilotes.
**Ne pas reecrire le releve a la main.** Une fiche materielle ecrite a la main pourrit en
silence, ce qui est la pire facon d'avoir tort.

Apres chaque mise a jour du serveur, une seule commande :

```
PYTHONPATH=src python src/diagnostics/probe_training_host.py \
    --host renard --gpu 1 \
    --json-out results/host-renard.json \
    --markdown-out docs/serveur-releve-renard.md
```

Elle re-derive depuis la machine elle-meme : capacite de calcul et donc support bf16,
memoire et donc checkpoints entrainables, versions de Python, de CUDA et de torch.

**Ce qui change quand renard est mis a jour, et ce qu'il faut relire :**

| Si ca change | Regarder |
|---|---|
| Le pilote et la capacite de calcul | La ligne bf16 du releve. Si elle passe a oui, `deberta-v2-xlarge` redevient envisageable et le plan de checkpoints se rouvre. |
| La version de Python | `pyproject.toml` cible py310 a py313. Au-dela il faut elargir la cible black, et verifier que `poutyne` et `transformers` suivent. |
| La version de torch et son CUDA | Les scripts de sweep. Et relire le defaut de precision. |
| La memoire GPU disponible | Le tableau "ce qui rentre" du releve genere, qui se recalcule seul. |

## Ce qui reste vrai quoi qu'il arrive au serveur

`docs/H1-diagnostic-calibration.md` etablit que les quatre checkpoints du sweep
convergent vers **le meme plancher de RMSE apres recalibrage affine, 22,1 a 23,1**, et que
le Pearson est plat a 0,78-0,80 quel que soit le backbone. Le backbone n'est pas le goulot
de la v2, la donnee l'est.

Consequence : perdre un gros checkpoint faute de memoire ne coute rien de scientifique, et
un serveur plus puissant ne rapprocherait pas la v2 de sa cible. La cible, Pearson 0,914,
s'atteint par le corpus.

## A faire avant le premier entrainement

1. Regenerer le releve (commande ci-dessus) et le lire.
2. Cloner le depot sur renard, se placer sur la branche v2.
3. Installer l'environnement Python. `uv` n'etait pas present au dernier releve ; verifier.
4. Rendre `--bf16` conditionnel a la capacite de calcul detectee, au lieu de le passer en
   dur. Les scripts de sweep actuels le passent toujours.
5. Verifier que les corrections C1 a C3 de `docs/H1-diagnostic-calibration.md` sont en
   place. Entrainer sans elles reproduirait la compression d'amplitude, et on aurait
   change le corpus pour rien.
