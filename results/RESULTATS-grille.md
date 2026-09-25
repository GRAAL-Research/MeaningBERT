# Resultats de l'experience v2

```
Objectif : maximiser Pearson avec identiques et non reliees au plus pres de 100 %.
objectif = Pearson x (identiques>95) x (non reliees<5), en produit pour qu'un
seul terme faible tire tout vers le bas.

arch                   variant   objectif  Pearson  ident>95  unrel<5    RMSE   floor      R2  pred_mu  pred_sd   train   ep
--------------------------------------------------------------------------------------------------------------------------------------------
deberta-v3-large       c_full       0.813    0.829      98.1    100.0   19.99   20.71   0.645    64.93    32.76    8087    4
deberta-v3-base        c_none       0.806    0.806     100.0    100.0   20.53   21.91   0.625    60.40    32.27    1111    7
deberta-v3-large       d_full       0.777    0.799      98.1     99.1   17.15   22.25   0.606    78.50    25.07   23418    7
stsb-roberta-base      c_none       0.774    0.774     100.0    100.0   23.20   23.42   0.521    59.70    35.34    1111    9
deberta-v3-large       c_none       0.766    0.843      97.2     93.5   18.16   19.91   0.707    62.01    29.24    1111    6
deberta-v3-base        c_full       0.755    0.799      97.2     97.2   22.43   22.27   0.552    67.59    33.18    8087    5
bert-base-uncased      c_none       0.748    0.799      95.4     98.1   22.22   22.26   0.561    65.58    34.40    1111    9
deberta-v3-base        d_none       0.716    0.788      98.1     92.6   18.40   22.80   0.547    80.04    26.77    3089    7
bert-base-uncased      c_full       0.626    0.800      80.6     97.2   23.10   22.22   0.525    69.46    33.29    8087    4
deberta-v3-base        d_full       0.584    0.780      77.8     96.3   18.15   23.17   0.559    79.75    24.83   23418   10

Corpus : v1 corrige (c) contre v2 (d), a augmentation egale
------------------------------------------------------------------------
  deberta-v3-base        no augmentation        Pearson -0.018   objectif -0.090
  deberta-v3-base        swap + BT + generated  Pearson -0.019   objectif -0.171
  deberta-v3-large       swap + BT + generated  Pearson -0.030   objectif -0.036

Augmentation : aucune contre les trois ensemble, a corpus egal
------------------------------------------------------------------------
  bert-base-uncased      v1 corrected (H6), grouped Pearson +0.001   ident -14.8 pts   objectif -0.122
  deberta-v3-base        v1 corrected (H6), grouped Pearson -0.007   ident -2.8 pts   objectif -0.051
  deberta-v3-base        v2 corpus, grouped         Pearson -0.008   ident -20.4 pts   objectif -0.132
  deberta-v3-large       v1 corrected (H6), grouped Pearson -0.014   ident +0.9 pts   objectif +0.047

Diagnostics, sur l'architecture de reference seulement
------------------------------------------------------------------------

Meilleure configuration au sens de l'objectif
------------------------------------------------------------------------
  deberta-v3-large / c_full
  Pearson 0.829   identiques>95 98.1 %   non reliees<5 100.0 %   RMSE 19.99
  cible PRODUIT.md : Pearson >= 0,914, RMSE < 15
  plancher de RMSE a cette correlation : 20.71
```
