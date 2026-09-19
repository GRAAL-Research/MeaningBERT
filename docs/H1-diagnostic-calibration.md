# H1 - Diagnostic de l'anomalie du sweep

Statut : **resolue**. Date : 2026-09-19.
Reproduire : `python src/diagnostics/calibration_audit.py --json-out results/calibration_audit.json`

## Question

`results/sweep_summary.csv` rapporte un Pearson de 0,78 a 0,80, un RMSE de 35 a 54 sur
une echelle 0-100, un R2 negatif et un ratio `Identical =100%` de exactement 0,000 sur
les 8 configurations. Est-ce la donnee, le modele, ou la couche de sortie ?

## Verdict

**La couche de sortie.** Le classement est appris, l'echelle ne l'est pas. Trois
defaillances distinctes, dont deux etaient invisibles dans le CSV.

### 1. Compression d'amplitude, sur 100 % des runs

Sur 80 runs termines, les predictions occupent une fraction de l'amplitude des labels.

| Checkpoint | n | Pearson | RMSE | pred_mean | pred_std |
|---|---|---|---|---|---|
| deberta-v3-large | 19 | 0,802 | 37,10 | 35,68 | 16,88 |
| deberta-v2-xlarge | 20 | 0,795 | 41,69 | 31,31 | 14,49 |
| deberta-v3-base | 20 | 0,784 | 51,41 | 21,64 | 9,22 |
| deberta-v3-small | 20 | 0,782 | 50,68 | 22,50 | 10,00 |

Les labels du corpus fusionne sont a **mean 62,66, std 37,01**. Les modeles predisent
entre 21 et 36 de moyenne, avec un ecart-type de 9 a 17. L'amplitude est comprimee d'un
facteur 2 a 4, en position comme en dispersion.

Le Pearson, lui, est plat a 0,78-0,80 quel que soit le backbone. Le RMSE rapporte n'est
pas une mesure de qualite du modele, c'est une mesure de sa compression : le classement
des quatre checkpoints par RMSE est exactement leur classement par `pred_mean`.

### 2. Une part massive du RMSE est recuperable sans toucher a rien

Le RMSE minimal atteignable par le meilleur recalibrage affine `a*p + b` des memes
predictions vaut `sigma_y * sqrt(1 - r^2)`. Tout ce qui depasse est une erreur de
position et d'echelle, donc de calibration.

| Checkpoint | RMSE rapporte | RMSE apres recalibrage affine | Part recuperable |
|---|---|---|---|
| deberta-v3-large | 37,10 | 22,12 | 40,4 % |
| deberta-v2-xlarge | 41,69 | 22,47 | 46,1 % |
| deberta-v3-base | 51,41 | 22,96 | 55,3 % |
| deberta-v3-small | 50,68 | 23,07 | 54,5 % |

Les quatre checkpoints convergent vers **le meme plancher, 22,1 a 23,1**. C'est la
confirmation quantitative de la premisse de `PRODUIT.md` : le backbone n'est pas le
goulot. Un recalibrage affine ajuste sur le dev fait passer le meilleur modele de 37,10
a 22,12 sans une ligne de donnee supplementaire et sans reentrainement.

### 3. Les runs effondres entrent dans l'agregat deguises en predicteurs de zero

Un run sur 80 s'est effondre vers une sortie constante. Trajectoire observee sur
`deberta-v3-large / swap` :

| epoch | pred_mean | pred_std | rmse | loss | pearson |
|---|---|---|---|---|---|
| 1 | 75,03 | 0,11 | 39,22 | 1538,5 | 0,000 |
| 2 | 58,75 | 0,00 | 37,60 | 1414,0 | nan |
| 30 | 63,75 | 0,00 | 37,36 | 1395,8 | nan |
| 44 | 0,00 | 0,00 | 73,30 | nan | nan |
| 58 | 0,00 | 0,00 | 73,30 | nan | nan |

Le modele sort une constante des l'epoch 2 (`pred_std = 0`), puis diverge en NaN a
l'epoch 44. `metrics._sanitize_predictions` applique `np.nan_to_num(nan=0.0)` : les NaN
deviennent des zeros, et le run est rapporte comme un modele qui predit 0 partout, avec
un RMSE de 73,30 parfaitement plausible. Rien ne signale la mort du run.

C'est le defaut le plus dangereux des trois. Il ne fausse pas seulement une metrique, il
fabrique un resultat credible a partir d'un entrainement rate, et ce resultat entre dans
la moyenne du CSV.

## Pourquoi `Identical =100%` est a 0,000

Consequence directe de la compression. Les paires identiques recoivent une prediction
moyenne de 32 la ou le label est 100. Aucune ne peut franchir le seuil de 95. Les paires
non reliees, elles, ont une prediction moyenne de 0,84 et passent leur test a 98 %. Le
test identique et le test non-relie ne sont pas symetriques : une sortie comprimee vers
le bas reussit automatiquement le second et echoue automatiquement le premier.

## Ce que ca fixe comme cible pour la v2

Avec `sigma_y = 37,01`, le RMSE minimal atteignable ne depend que du Pearson.

| RMSE vise | Pearson requis |
|---|---|
| 25 | 0,737 |
| 22 | 0,804 |
| 20 | 0,841 |
| 18 | 0,874 |
| **15** | **0,914** |
| 12 | 0,946 |

La cible de `PRODUIT.md` est RMSE < 15. La calibration seule amene a 22,1. **Le reste du
chemin exige de passer le Pearson de 0,80 a 0,914**, et c'est exactement ce que
l'expansion de corpus doit produire. La v2 a maintenant un critere de succes falsifiable
qui ne depend pas d'un seuil arbitraire.

## Corrections a porter

| # | Correction | Ou | Effet attendu |
|---|---|---|---|
| C1 | Arreter de convertir les NaN en 0. Faire echouer le run, ou le marquer `diverged` et l'exclure de l'agregat. | `src/training/metrics/metrics.py` | Les runs morts cessent de polluer les moyennes. |
| C2 | Detecter l'effondrement en cours d'entrainement (`pred_std` sous un seuil pendant N evaluations) et arreter le run. | callback dans `few_shot_training.py` | Economise le calcul et signale le probleme. |
| C3 | Borner la sortie : tete sigmoide x 100, ou cible normalisee sur [0, 1]. | `few_shot_training.py` | Supprime la cause de la compression a la racine. |
| C4 | Recalibrage affine ou isotone ajuste sur le dev, applique au test. | nouveau `src/training/calibration.py` | Filet meme si C3 ne suffit pas. Gain mesure : 40 a 55 % du RMSE. |
| C5 | Reporter `pred_mean` et `pred_std` a cote de chaque RMSE dans les tableaux. | `src/figures_generator/` | La compression redevient visible a l'oeil. |

C1 et C3 sont des prerequis a tout reentrainement v2. C4 est un filet, pas un substitut :
recalibrer un modele comprime restaure l'echelle mais pas l'information perdue.

## Portee du diagnostic

80 runs termines sur 103, tous `deberta-v2-xlarge`, `deberta-v3-small`, `deberta-v3-base`
et `deberta-v3-large`, en augmentation `swap` et `back_translation`. Les 23 autres runs
sont `failed` ou `crashed` et n'ont pas de metrique de test. Les checkpoints non-deberta
du sweep initial ne sont pas dans ce projet wandb.
