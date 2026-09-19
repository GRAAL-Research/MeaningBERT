# H5 - Fuite par phrase source dans le corpus d'entrainement

Statut : **confirmee sur le corpus reellement entraine**. Date : 2026-09-19.
Reproduire : `PYTHONPATH=src python src/diagnostics/leakage_audit.py --json-out results/leakage_audit.json`

## Le constat

Les 10 folds du sweep ont ete reproduits a l'identique. `create_fold_splits` est
deterministe a seed donnee, donc rejouer la fonction sur le corpus fusionne reconstruit
exactement ce que le sweep a vu.

| seed | train | dev | test | lignes de test fuitees | groupes de test fuites | doublons exacts |
|---|---|---|---|---|---|---|
| 42 | 1224 | 205 | 613 | 551 (89,9 %) | 326 (90,8 %) | 0 |
| 43 | 1224 | 205 | 613 | 549 (89,6 %) | 326 (90,8 %) | 0 |
| 44 | 1224 | 205 | 613 | 566 (92,3 %) | 338 (94,9 %) | 0 |
| 45 | 1224 | 205 | 613 | 547 (89,2 %) | 324 (90,3 %) | 0 |
| 46 | 1224 | 205 | 613 | 581 (94,8 %) | 348 (96,1 %) | 0 |
| 47 | 1224 | 205 | 613 | 551 (89,9 %) | 322 (91,5 %) | 0 |
| 48 | 1224 | 205 | 613 | 566 (92,3 %) | 342 (93,7 %) | 0 |
| 49 | 1224 | 205 | 613 | 539 (87,9 %) | 311 (90,1 %) | 0 |
| 50 | 1224 | 205 | 613 | 575 (93,8 %) | 354 (94,9 %) | 0 |
| 51 | 1224 | 205 | 613 | 566 (92,3 %) | 344 (93,2 %) | 0 |

**Mediane : 91,1 % des lignes de test ont leur phrase source presente dans le train.**
91,0 % pour le dev. Et **zero doublon exact**.

## Pourquoi personne ne l'a vu

Le corpus fusionne compte 2042 lignes pour **493 phrases sources distinctes**, soit 4,1
simplifications par phrase. Une phrase source porte plusieurs simplifications ; decouper a
la ligne les eparpille des deux cotes du mur.

`src/training/validate_datasets.py` verifie l'absence de fuite, et il a raison sur ce
qu'il verifie :

```python
def make_fingerprints(dataset) -> set[str]:
    """Create a set of fingerprints from (original, simplification) pairs."""
    return {f"{o}|||{s}" for o, s in zip(dataset["original"], dataset["simplification"])}
```

L'empreinte est la **paire exacte**. Deux lignes qui partagent la phrase source mais pas
la simplification ont deux empreintes differentes, donc aucune fuite detectee. Le script
rapporte 0, ce qui est exact et trompeur en meme temps.

## Sur les splits publies de CSMD, meme probleme

Ce n'est pas un defaut du pipeline de folds, il est en amont, dans le corpus publie.

| | |
|---|---|
| sources train ∩ test | 229, soit 82,7 % des sources du test |
| lignes de test dont la source est vue en train | 337/407, soit **82,8 %** |
| lignes de dev dont la source est vue en train | 78/95, soit **82,1 %** |
| sources des paires `identical` vues en train | 274/359, soit **76,3 %** |
| sources des paires `unrelated` vues en train | 274/359, soit **76,3 %** |
| **paires exactes dupliquees train ∩ test** | **12** |

Les 12 dernieres sont des doublons francs, pas de la fuite par groupe : la meme paire
`(original, simplification)` figure dans le train et dans le test du corpus publie.

Et les jeux dits `holdout` n'en sont pas : trois quarts de leurs phrases sources sont dans
le train. Sur 493 phrases sources, 359 apparaissent a la fois en `original`, `identical`
et `unrelated`.

## Ce que ca invalide

1. **Le Pearson de 0,78 a 0,80 du sweep est optimiste.** Il est mesure sur un test dont
   91 % des phrases sources etaient dans l'entrainement. La performance hors echantillon
   reelle est inconnue, et necessairement inferieure.
2. **La cible de la v2 doit etre re-etalonnee.** `PRODUIT.md` vise Pearson >= 0,914 pour
   descendre sous RMSE 15. Cette cible se compare a une ligne de base de 0,80 qui est
   elle-meme gonflee. **Il faut re-mesurer la ligne de base sur un decoupage propre avant
   de pouvoir juger l'apport des nouveaux corpus.** Sans ca, on ne saura pas distinguer un
   gain du a la donnee d'une simple disparition de la fuite.
3. **Les deux sanity checks ne sont pas des holdout.** Ce qui se combine avec la
   conclusion de `docs/H1-diagnostic-calibration.md`.

## Ce que ca aggrave pour la v2

Les corpus qui arrivent sont bien pires sous cet angle. SimpEval porte environ 2400
simplifications pour a peu pres **60 phrases sources**, soit 40 simplifications par
phrase. Un decoupage a la ligne sur SimpEval n'est pas un decoupage. Sans correction, la
v2 multiplierait le volume et amplifierait la fuite au lieu de la corriger, et la
correlation rapportee monterait pour la mauvaise raison.

## La correction

`src/data/splits.py`. L'unite de decoupage est le **groupe de phrase source**, jamais la
ligne.

- `split_by_source_sentence()` : aucune phrase source ne traverse deux splits. Les
  doublons exacts sont retires avant le decoupage. Les groupes sont places du plus gros au
  plus petit dans le split le plus en retard sur sa cible, parce que les tailles de groupe
  vont de 1 a 40 et qu'un tirage proportionnel raterait les proportions visees.
- Un split `sanity` distinct : une fraction des groupes portant des paires `identical` ou
  `unrelated` est reservee entierement. Leurs paires de sanity vont dans `sanity`, leurs
  paires ordinaires dans `test`, et le groupe entier reste hors du `train`. C'est ce qui
  rend le sanity check enfin hors echantillon.
- `assert_no_leakage()` verifie la phrase source **et** la paire exacte. `test` et
  `sanity` ont le droit de partager une phrase source, par construction, et c'est declare
  explicitement plutot que tolere en silence.

Resultat sur CSMD seul, seed 42 : 31 doublons exacts retires, puis train 1111 lignes sur
229 groupes, dev 159 sur 60, test 556 sur 196, sanity 216 sur 108. Zero fuite.

## Corrections a porter

| # | Correction | Ou |
|---|---|---|
| L1 | Remplacer `create_fold_splits` par `split_by_source_sentence`. | `src/training/prepare_datasets.py` |
| L2 | Ajouter la verification par groupe a cote de la verification par paire exacte. | `src/training/validate_datasets.py` |
| L3 | Re-mesurer la ligne de base v1 sur un decoupage propre, avant toute conclusion sur l'apport des corpus. | nouveau run |
| L4 | Exclure du back-translation les phrases sources presentes en dev ou test, pas seulement les paires exactes. | `src/training/prepare_datasets.py` |

L3 est la plus importante. Sans elle, la v2 ne pourra pas prouver que son gain vient de la
donnee.
