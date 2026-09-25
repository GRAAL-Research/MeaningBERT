# Resultats de l'experience v2

```
Objectif : maximiser Pearson avec identiques et non reliees au plus pres de 100 %.
objectif = Pearson x (identiques>95) x (non reliees<5), en produit pour qu'un
seul terme faible tire tout vers le bas.

arch                   variant   objectif  Pearson  ident>95  unrel<5    RMSE   floor      R2  pred_mu  pred_sd   train   ep
--------------------------------------------------------------------------------------------------------------------------------------------
deberta-v3-base        b_full       0.787    0.787     100.0    100.0   22.12   22.81   0.565    65.60    32.13    8087   10
deberta-v3-base        c_none       0.657    0.807      92.6     88.0   20.01   21.87   0.644    59.89    29.84    1111    9
deberta-v3-base        d_full       0.379    0.779      50.0     97.2   17.57   23.21   0.586    76.64    24.69   23418    6
deberta-v3-base        d_none       0.308    0.791      41.7     93.5   17.06   22.62   0.610    76.50    24.65    3089   10
deberta-v3-base        a_none       0.182    0.894      26.9     75.9   16.83   16.57   0.793    65.86    33.08    1243    6
deberta-v3-base        b_none       0.064    0.796       9.3     87.0   20.67   22.40   0.620    62.34    30.00    1111    9
deberta-v3-base        a_full       0.032    0.879       3.7     99.1   17.73   17.66   0.770    61.92    34.09    7111    4
deberta-v3-base        c_full       0.000    0.805       0.0    100.0   20.60   21.93   0.622    64.95    29.59    8087    4

Corpus : v1 corrige (c) contre v2 (d), a augmentation egale
------------------------------------------------------------------------
  deberta-v3-base        no augmentation        Pearson -0.015   objectif -0.349
  deberta-v3-base        swap + BT + generated  Pearson -0.027   objectif +0.379

Augmentation : aucune contre les trois ensemble, a corpus egal
------------------------------------------------------------------------
  deberta-v3-base        v1 corrected (H6), grouped Pearson -0.001   ident -92.6 pts   objectif -0.657
  deberta-v3-base        v2 corpus, grouped         Pearson -0.013   ident +8.3 pts   objectif +0.070

Diagnostics, sur l'architecture de reference seulement
------------------------------------------------------------------------
  deberta-v3-base        a -> b  fuite par phrase source (H5)   no augmentat Pearson -0.098   ident -17.6 pts
  deberta-v3-base        a -> b  fuite par phrase source (H5)   swap + BT +  Pearson -0.091   ident +96.3 pts
  deberta-v3-base        b -> c  etiquettes permutees (H6)      no augmentat Pearson +0.011   ident +83.3 pts
  deberta-v3-base        b -> c  etiquettes permutees (H6)      swap + BT +  Pearson +0.018   ident -100.0 pts

Meilleure configuration au sens de l'objectif
------------------------------------------------------------------------
  deberta-v3-base / b_full
  Pearson 0.787   identiques>95 100.0 %   non reliees<5 100.0 %   RMSE 22.12
  cible PRODUIT.md : Pearson >= 0,914, RMSE < 15
  plancher de RMSE a cette correlation : 22.81
```
