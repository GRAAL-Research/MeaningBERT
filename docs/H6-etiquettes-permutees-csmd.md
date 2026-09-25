# H6 - 360 etiquettes permutees dans CSMD, et le plafond qu'elles imposent

Statut : **confirmee et corrigee**. Date : 2026-09-19.
C'est le resultat le plus important du chantier v2.

## Le constat

CSMD est construit a partir de quatre corpus, dont **SimpDA_2022**. Le loader SimpEval
charge ce meme fichier depuis sa source d'origine, `simpDA_2022.csv` du depot LENS. Les
deux doivent donc porter les memes annotations sur les memes paires.

Sur les **360 paires communes** :

| Mesure | Valeur |
|---|---|
| Pearson | **+0,009** |
| Spearman | +0,060 |
| Kendall | +0,035 |
| Moyenne CSMD | 87,55 |
| Moyenne adequation source | 87,55 |
| Ecart-type CSMD | 12,9594 |
| Ecart-type source | 12,9594 |
| **Multiensembles des valeurs identiques** | **360 / 360** |
| **max abs(tri(CSMD) - tri(source))** | **0,000000** |
| Egalites exactes en place | 23 / 360 |
| Ecart absolu median | 9,67 |

Des marges rigoureusement identiques et une correlation nulle, ce n'est pas un desaccord
entre deux protocoles d'annotation. **Ce sont les memes etiquettes attachees aux mauvaises
paires.** Un desaccord d'annotation aurait change les valeurs ; ici les 360 valeurs sont
les memes, a 1e-6 pres, simplement redistribuees.

## Ce n'est pas un artefact d'ordre de tri

| Hypothese | r obtenu |
|---|---|
| ordre id-major des deux cotes | +0,009 |
| CSMD en systeme-major contre source en id-major | +0,052 |
| CSMD en id-major contre source en systeme-major | -0,027 |
| decalage de +1, +2, +3, -1, -2 | entre -0,101 et +0,082 |
| a l'interieur de chaque phrase source (60 groupes) | +0,079 |
| entre phrases sources (moyennes par id) | -0,117 |

La permutation est brouillee a l'interieur comme entre les phrases sources. Aucun
re-tri ne la defait ; seule une re-derivation depuis la source la corrige.

## Le sens de l'erreur

Le CSV source porte, **sur une meme ligne**, `Input.original`, `Input.simplified` et
`Answer.adequacy`. Une lecture ligne a ligne ne peut pas les desaligner. La copie dans
CSMD, elle, le peut. Le defaut est donc du cote de CSMD.

Verification de controle : les 127 paires que CSMD partage avec `simpeval_past`, qui vient
d'un autre fichier, correlent a **+0,683**. Les autres sources de CSMD ne sont pas
touchees ; le defaut est circonscrit a l'import de SimpDA_2022.

## Ce que ca explique

360 lignes sur les 1355 annotees, soit **26,6 %**, portent l'etiquette d'une autre paire.
Un predicteur parfait des vraies etiquettes ne peut donc pas depasser un certain Pearson,
que l'on calcule par simulation :

| Perimetre | Plafond de Pearson |
|---|---|
| corpus fusionne, 2073 lignes dont 360 permutees | **0,826 ± 0,011** |
| les 1355 lignes annotees seules | 0,736 ± 0,019 |

A comparer au sweep :

| Checkpoint | Pearson observe |
|---|---|
| deberta-v3-small (142 M) | 0,782 |
| deberta-v3-base (184 M) | 0,784 |
| deberta-v2-xlarge (900 M) | 0,795 |
| deberta-v3-large (434 M) | **0,802** |

**Les modeles saturent le plafond.** 0,802 contre une borne de 0,826 : ils en atteignent
97 %. C'est l'explication du fait le plus etrange du sweep, note dans `PRODUIT.md` et dans
`docs/H1-diagnostic-calibration.md` : le Pearson ne bouge pas de 44 M a 900 M de
parametres. Il ne bouge pas parce qu'il n'y a plus rien a apprendre. Ce n'etait pas une
limite de modele, c'etait un plafond de bruit d'etiquetage.

## Consequence sur la cible de la v2

`PRODUIT.md` vise **Pearson >= 0,914**, equivalent de RMSE < 15. Sur le corpus actuel,
cette cible est **inatteignable a n'importe quelle taille de modele et avec n'importe quel
volume de corpus ajoute**, puisque le plafond est a 0,826.

Corriger ces 360 etiquettes vaut donc plus, a soi seul, que toute l'expansion de corpus.

## La correction

`src/data/corrections.py`, correction `csmd-simpda2022-relabel`. Pour les seules paires
que CSMD partage avec le sous-ensemble 2022 de SimpEval, `label_raw` est re-derive depuis
la source. Aucune ligne supprimee, aucun texte modifie, aucune autre etiquette touchee.

Effet mesure :

| | avant | apres |
|---|---|---|
| r sur les 487 paires d'ancrage CSMD / SimpEval | +0,309 | **+0,806** |
| lignes CSMD modifiees | - | 360 |
| moyenne du corpus CSMD | 62,66 | 62,66 |
| ecart-type du corpus CSMD | 37,01 | 37,01 |

Les marges sont rigoureusement inchangees, ce qui confirme que la correction ne fait que
re-apparier : elle ne deplace aucune valeur, elle les remet en face de la bonne paire.

Le r de 0,806 entre les deux protocoles sur les memes paires devient enfin credible, et
c'est aussi la meilleure estimation dont on dispose de l'accord inter-protocole, donc du
plafond raisonnable pour une metrique entrainee dessus.

## A verifier ensuite

1. **Re-mesurer la ligne de base v1** sur le corpus corrige et un decoupage propre (H5).
   C'est la seule facon de savoir ce que la v2 gagne reellement.
2. **Remonter le defaut en amont** : le CSMD publie sur HuggingFace porte l'erreur. Une v2
   du corpus doit le corriger a la source, et l'article publie devrait le signaler.
3. Verifier les deux autres sources de CSMD, QuestEval et Simplicity-DA, par le meme test
   d'ancrage, si leurs fichiers d'origine sont accessibles.
