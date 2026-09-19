# Plan d'entrainement v2 : renard

Ce fichier porte les **decisions**. Le **releve materiel** vit dans
`docs/serveur-releve-renard.md`, qui est genere et ne doit pas etre edite a la main.

## Decisions

| Point | Decision | Date |
|---|---|---|
| Serveur d'entrainement | `renard`, et non `caribou` | 2026-09-19, David |
| GPU | ~~GPU 1 uniquement, la Quadro P5000~~ **remplace le 2026-09-19** : voir ci-dessous | 2026-09-19, David |
| Parc d'entrainement | renard GPU 0 et GPU 1, plus souris GPU 0. Une tranche de grille par carte | 2026-09-19, David |
| Multi-GPU **dans un run** | toujours interdit : un entrainement, une carte | 2026-09-19 |
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

## Le parc, et pourquoi il a trois cartes

Decision du 2026-09-19, David : *separer la charge pour accelerer l'execution*.

La grille de la phase 2 est sequentielle par construction, un entrainement par carte a la
fois. Sur la seule P5000, ses vingt runs demandaient une nuit et une journee, pendant que
deux cartes dormaient a cote. Le verificateur d'environnement les donne a egalite sur la
charge de reference :

| Carte | Machine | Memoire | Debit de reference |
|---|---|---|---|
| Quadro P5000 | renard GPU 1 | 16 Go | mesure a l'installation |
| GeForce GTX 1080 Ti | renard GPU 0 | 10.9 Go | 41.0 pas/s |
| TITAN Xp | souris GPU 0 | 11.9 Go | 40.4 pas/s |

Les trois sont Pascal, `sm_61`, donc `fp32` partout et la meme roue `torch==2.14.0+cu126`,
gelee dans `env/renard-lock.txt` et installee telle quelle sur souris. Rien ne distingue
les runs d'une machine a l'autre : c'est ce qui autorise a les comparer.

### Une tranche par carte

`src/training/run_worker.sh <nom>` lit `env/workers/<nom>.env` et n'execute que la tranche
qui y est decrite, sous forme de lots `architectures|variantes` separes par `;`.

Le partitionnement est **statique**. renard et souris ne partagent aucun systeme de
fichiers, donc aucun verrou commun n'est possible et une file de travail partagee non
plus. Des tranches disjointes rendent la collision impossible sans coordination.

`deberta-v3-large` ne se partitionne pas par architecture, puisqu'il est seul et coute
trois fois le reste : il se coupe par rang de corpus. Son rang `d`, ou les sequences
montent a 512 jetons, va sur la P5000 et ses 16 Go.

Deux garde-fous :

- `run_worker.sh` attend que sa carte soit libre avant de demarrer. Deux entrainements
  concurrents sur 11 a 16 Go tombent en OOM tous les deux.
- `run_grid.sh` refuse un appel **sans** `ARCHS` tant que `results/GRID_PARTITIONED`
  existe. C'est ce que fait la phase 2 de `run_everything.sh`, ecrite quand il n'y avait
  qu'un GPU : elle relancerait la grille entiere sur une carte, en double du reste du parc.

### Reprise et collecte

Un chien de garde par tranche, toutes les dix minutes, sur chaque machine :

```
*/10 * * * * ~/MeaningBERT/src/training/watchdog_worker.sh renard-gpu0
```

Relancer ne coute rien : `run_grid.sh` saute toute variante dont le JSON existe deja. Une
tranche complete pose `results/WORKER_COMPLETE-<nom>` et le chien de garde se tait.

Les resultats vivent sur trois machines. Pour les rassembler et les analyser :

```
bash src/training/collect_results.sh
```
