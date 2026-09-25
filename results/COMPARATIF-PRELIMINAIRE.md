# Comparatif preliminaire v2

Etat au 2026-09-20, 08 h 00. **10 runs de grille sur 20**, plus les 8 de la phase 1.
Manquent : `nli-deberta-v3-base` en entier, `stsb-roberta-base` sauf c_none,
`bert` d_none et d_full, `deberta-v3-large` d_none.

Colonnes : `ident mu` est le score moyen predit sur des paires identiques, ou la
verite est 100 ; `ident >95` la part de ces paires au-dessus de 95. `unrel mu` est le
score moyen sur des paires sans rapport, ou la verite est 0 ; `unrel <5` la part
au-dessous de 5. Ce sont les tests de bon sens : un modele qui les rate est inutilisable
quelle que soit sa correlation.

## 1. Tete de sortie, a architecture et corpus identiques

Le seul facteur teste des deux cotes : `deberta-v3-base`, memes quatre variantes,
meme graine, meme lot effectif.

| variante | Pearson sigmoide | Pearson clamped | ident mu sigmoide | ident mu clamped | ident >95 sigmoide | ident >95 clamped |
|---|---|---|---|---|---|---|
| c_none | 0.807 | 0.806 | 96.36 | 100.00 | 92.6 % | 100.0 % |
| c_full | 0.805 | 0.799 | 93.22 | 99.79 | 0.0 % | 97.2 % |
| d_none | 0.791 | 0.788 | 94.87 | 99.50 | 41.7 % | 98.1 % |
| d_full | 0.779 | 0.780 | 95.34 | 97.75 | 50.0 % | 77.8 % |

La correlation ne bouge pas, au millieme pres. Le test des paires identiques, lui,
passe de rate a reussi. Le cas `c_full` est le plus net : 0 % des paires identiques
au-dessus de 95 sous sigmoide, 97,2 % sous clamped, pour un Pearson qui perd 0,006.
La sigmoide n'atteint jamais 100 par construction, et environ la moitie des
etiquettes d'entrainement valent exactement 0 ou exactement 100.

## 2. Architectures, tete clamped, corpus c_none

Le seul point ou les quatre architectures sont comparables aujourd'hui.

| architecture | Pearson | RMSE | R2 | ident mu | ident >95 | unrel mu |
|---|---|---|---|---|---|---|
| deberta-v3-large | 0.843 | 18.16 | 0.707 | 99.49 | 97.2 % | 0.89 |
| deberta-v3-base | 0.806 | 20.53 | 0.625 | 100.00 | 100.0 % | 0.05 |
| bert-base-uncased | 0.799 | 22.22 | 0.561 | 99.33 | 95.4 % | 0.43 |
| stsb-roberta-base | 0.774 | 23.20 | 0.521 | 99.97 | 100.0 % | 0.00 |

`deberta-v3-large` mene de 0,037 sur `base` et de 0,069 sur `stsb-roberta-base`.
L'ecart depasse le bruit inter-variantes observe jusqu'ici, mais une seule graine a
tourne : c'est une tendance, pas une mesure.

## 3. Corpus v1 corrige (c) contre v2 (d), tete clamped

| architecture | variantes | Pearson | RMSE |
|---|---|---|---|
| deberta-v3-base | c_none -> d_none | 0.806 -> 0.788 (-0.018) | 20.53 -> 18.40 (-2.13) |
| deberta-v3-base | c_full -> d_full | 0.799 -> 0.780 (-0.019) | 22.43 -> 18.15 (-4.28) |

Le corpus v2 baisse la correlation d'environ 0,02 et baisse la RMSE de 2 a 4 points.
Les deux vont dans des sens opposes parce que la distribution des scores predits se
deplace vers le haut sur le rang d (moyenne predite 80 contre 60 sur le rang c).
Il faudra trancher au sens de l'objectif, pas d'une metrique seule.

## 4. Ecart a la cible de PRODUIT.md

Cible : Pearson >= 0,914 et RMSE < 15.
Meilleur Pearson legitime a ce jour : 0,843 (`deberta-v3-large` / c_none).
Meilleure RMSE : 17,15 (`deberta-v3-large` / d_full).
Aucune configuration n'atteint la cible, et aucune n'approche les deux a la fois.

Les conditions `a_none` et `a_full` affichent 0,894 et 0,879, au-dessus de tout le
reste. Ce sont les diagnostics de la fuite par phrase source (H5) : leur correlation
est gonflee par la fuite, elles ne sont pas des configurations candidates.
