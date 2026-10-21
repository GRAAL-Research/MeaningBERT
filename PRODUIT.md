# PRODUIT.md - MeaningBERT

Fichier de cadrage produit. Lu avant de poser une question, ecrit apres chaque resultat.
Les non-objectifs verrouillent la portee.

## Probleme

MeaningBERT v1 est une metrique de preservation du sens entrainee sur 1355 paires
annotees (CSMD). Ce volume est le plafond de la metrique : le sweep multi-checkpoint de
2026-04 montre que passer de deberta-v3-small a deberta-v2-xlarge deplace le Pearson de
0,777 a 0,804, soit 0,027 pour 40x les parametres. Le backbone n'est pas le goulot, la
donnee l'est.

## Utilisateurs bloques

Chercheurs et praticiens en simplification de texte qui evaluent des systemes et qui
n'ont pas de metrique de preservation du sens fiable. Aujourd'hui ils utilisent BLEU ou
BERTScore, qui correlent mal avec le jugement humain sur cette dimension precise.

## Job to be done

Quand j'evalue une simplification automatique, je veux un score de preservation du sens
qui correle avec ce qu'un annotateur humain aurait donne, pour pouvoir comparer des
systemes sans lancer une campagne d'annotation.

## Resultat vise, mesurable

1. Corpus d'entrainement passe de 1355 a au moins 8000 paires portant un jugement humain.
2. Pearson sur un test externe tenu a l'ecart (Agrawal & Carpuat TACL 2024, ou un split
   SimpEval jamais vu) superieur au v1 mesure sur le meme test.
3. RMSE sur echelle 0-100 sous 15, et R2 positif. **Traduit en cible directe : Pearson
   >= 0,914**, mesure sur un decoupage groupe par phrase source (H5). **La ligne de base
   v1 doit etre re-mesuree sur ce meme decoupage avant toute conclusion** : le 0,80 publie
   est mesure sur un test dont 91 % des phrases sources etaient dans l'entrainement, donc
   on ne pourrait pas distinguer un gain du a la donnee d'une simple disparition de la
   fuite. Avec sigma_y = 37,01, le RMSE minimal atteignable vaut
   sigma_y * sqrt(1 - r^2), donc RMSE < 15 equivaut exactement a Pearson > 0,914. Le
   Pearson actuel est de 0,80. La calibration seule amene le RMSE a 22,1 ; le reste du
   chemin est le travail de l'expansion de corpus.
4. Les deux sanity checks v1 repassent : identique >= 95 et non reliee <= 5, sur un
   holdout reellement tenu a l'ecart.

## Non-objectifs de la v2

Verrouilles. Tout ce qui suit est explicitement hors portee et part en v3 ou v4.

- **Pas de tete de polarite, pas de scores negatifs, pas d'opposition.** v3.
- **Pas de benchmark de dissociation.** v3.
- **Pas de multilingue.** v2 est anglais seul. v4.
- **Pas de passage au document.** v2 reste phrase a phrase.
- **Pas de LLM-as-judge**, ni en baseline ni en professeur. v3.
- **Pas de sortie localisee** au niveau du mot ou du segment. v3.
- **Pas de nouvelle campagne d'annotation humaine.** v2 recycle des annotations qui
  existent deja.
- **Pas d'exploration de backbone au-dela de ce que le sweep a deja mesure.** Le choix se
  fait dans les checkpoints deja evalues.

## Hypotheses a risque

| # | Hypothese | Risque si fausse | Comment on la teste |
|---|---|---|---|
| H1 | ~~L'anomalie du sweep vient de la calibration de sortie.~~ **CONFIRMEE le 2026-09-19.** Voir `docs/H1-diagnostic-calibration.md`. | - | Resolue. 40 a 55 % du RMSE est recuperable par recalibrage affine ; les 4 checkpoints convergent vers le meme plancher de 22,1-23,1. |
| H2 | Les scores de SimpEval, SALSA, SimpleText et PLABA sont harmonisables sur une echelle 0-100 commune sans detruire le signal. | Le corpus fusionne est plus bruite que CSMD seul et la metrique se degrade. | Ablation : entrainer sur CSMD seul vs CSMD + chaque corpus, un a un. |
| H3 | Les 60 phrases partagees entre SimpEval2022 et SynthSimpliEval suffisent comme point d'ancrage inter-corpus. | L'harmonisation repose sur une normalisation arbitraire par corpus. | Mesurer l'accord sur les paires communes avant de fixer le mapping. |
| H4 | Les licences des cinq corpus permettent une rediffusion dans CSMD v2. | On ne peut pas republier le corpus, seulement les loaders. | Verification licence par licence en phase 1. |
| H6 | ~~Les etiquettes de CSMD correspondent a leurs paires.~~ **REFUTEE le 2026-09-19.** Voir `docs/H6-etiquettes-permutees-csmd.md`. | - | 360 lignes sur 1355, soit 26,6 %, portent l'etiquette d'une autre paire. Plafond de Pearson impose : 0,826. Le sweep plafonne a 0,802, soit 97 % de la borne. Corrigee par `src/data/corrections.py`. |
| H5 | ~~Les splits du corpus d'entrainement isolent correctement le test.~~ **REFUTEE le 2026-09-19.** Voir `docs/H5-fuite-par-phrase-source.md`. | - | 91,1 % des lignes de test ont leur phrase source en train, mediane sur les 10 folds. Le Pearson de 0,80 est gonfle. Corrige par `src/data/splits.py`. |

## Plan d'experience v2

Factoriel 4 x 2. Quatre conditions corpus/decoupage en echelle, chacune entrainee avec et
sans augmentation. Construit par `src/data/build_corpus.py`, lance par
`src/training/run_v2_experiment.sh`, lu par
`src/figures_generator/analyze_v2_experiment.py`.

| Condition | Corpus | Decoupage | Ce que le barreau isole |
|---|---|---|---|
| a | CSMD v1 | a la ligne, comme v1 | reproduit le chiffre publie |
| b | CSMD v1 | groupe par phrase source | la fuite (H5) |
| c | CSMD v1 corrige H6 | groupe | les etiquettes permutees (H6) |
| d | v2, quatre corpus | groupe | l'apport reel des corpus ajoutes |

Augmentation : `none`, ou les trois ensemble (swap, back-translation par pivot francais,
generation de paires identiques et non reliees). Le sweep v1 n'a jamais mesure `none`, et
traitait swap et back_translation comme deux conditions alors qu'elles etaient emboitees.

Sans les barreaux a et b, un gain mesure en d serait inattribuable : il pourrait venir des
nouveaux corpus comme de la simple disparition d'un defaut.

Backbone : `deberta-v3-base`, tete de sortie `sigmoid` (correction C3), fp32. Voir
`docs/serveur-entrainement-renard.md`.

## Metrique du resultat vise, instrumentation

Le test externe et les sanity checks tournent dans la CI a chaque entrainement, pas
seulement a la fin. Un entrainement qui ne produit pas ces quatre chiffres est un
entrainement non mesure.

## v4 : multilingue (cadrage du 2026-10-11)

Point de départ : tag `v3-arr` (`67e5a00`), l'échelle signée et AUG+LEX. Document complet :
`docs/produit/cadrage-v4-multilingue.md`.

### Problème

Dans une autre langue que l'anglais, rien ne dit si une réécriture contredit sa source.
Les métriques disponibles (BERTScore multilingue, LaBSE, JUDGEBERT) sont non signées, et on
ignore si traduire vers l'anglais avant d'appliquer la v3 suffit à le dire.

### Utilisateurs bloqués

Les chercheurs qui évaluent la simplification en français, en allemand ou en arabe, et qui
retombent sur une métrique de similarité au moment de comparer des systèmes.

### Job to be done

Quand j'évalue des simplifications en français, je veux savoir si une sortie dit le
contraire de sa source, afin de pénaliser un système qui inverse le sens sans lancer
d'annotation humaine.

### Résultat visé, mesurable

Étape 1, le pilote français, pour ARR février 2027. Les seuils sont fixés ici, avant toute
mesure française, pour qu'aucun résultat ne puisse être relu après coup. Détail dans
`docs/produit/cadrage-v4-multilingue.md`, section « Critères de passage ».

1. **Battre les deux barres sur DACCORD**, le seul corpus de polarité écrit en français.
   Chaque barre est la meilleure tête mesurée dans sa catégorie (décision du 2026-10-11) :
   - **1a, décide si on construit** : battre la meilleure tête multilingue prête à l'emploi
     qui lit le français (`XLM-R-large NLI`, AUC 90,95).
   - **1b, décide ce que l'article revendique** : battre le meilleur translate-test
     (`DeBERTa-v3-large NLI` sur la traduction, AUC 96,71).

   Le seuil est le même pour les deux : au moins 2 points d'AUC, avec un intervalle
   bootstrap à 95 % qui exclut 0. RTE3-FR, GQNLI-FR, FraCaS-fr et SICK-fr sont traduits
   de l'anglais, ce qui avantage translate-test : ils sont rapportés, mais ne décident pas.
2. **Parité avec l'anglais à 2 points, sur des items appariés.** La mesure se fait sur le
   test de XNLI, dont les items anglais et français sont les mêmes, et sur des paires
   minimales générées des mêmes gabarits dans les deux langues. La référence est la v3
   AUG+LEX réentraînée sur le même backbone multilingue (barreau V4-H2) : le français doit
   être au moins à 2 points sous l'anglais en AUC et en macro-F1.
3. **Garde-fous contre les fausses alarmes, valables pour les deux critères.** Au moins
   95 % des synonymes et hyperonymes et au moins 90 % des co-hyponymes restent au-dessus
   de zéro. Une tête qui met tout sous zéro ne passe pas.
4. **Magnitude empruntée.** Le Pearson sur FrJUDGE doit être au moins celui de BERTScore
   multilingue. JUDGEBERT est exclu de cette mesure s'il a été entraîné sur FrJUDGE.

Valeur actuelle (discovery du 2026-10-11, `results/v4/discovery/report.md`) : barre 1a
90,95, barre 1b 96,71. La v3 AUG+LEX fait 84,04 en translate-test et 84,70 en lisant le
français.

Point de décision à la fin de la discovery (vers le 2026-10-23) : si translate-test ou la
tête prête à l'emploi arrive à moins de 2 points d'AUC sur DACCORD, David tranche à ce
moment-là, au vu des chiffres. Les non-objectifs restent en vigueur jusque-là.

Séquence ensuite : étape 2 (FR + DE + AR, avec FrJUDGE, DEplain et SAMER), puis étape 3
(XNLI et PAWS-X à grande échelle).

### Non-objectifs de la v4

- **Pas d'allemand ni d'arabe avant que le français passe ses deux critères.**
- **Pas d'interlingue (source EN, cible FR) dans le résultat visé.** C'est une expérience
  secondaire possible : section, annexe ou autre article.
- **Aucune annotation ni validation humaine de données.** On s'appuie seulement sur des
  corpus déjà annotés et sur des paires générées ; le biais du générateur est déclaré en
  limitation.
- **Pas de changement de la formule** `magnitude x (1 - 2 p_contra)`.
- **Pas de passage au document, pas de LLM-as-judge comme métrique.** Un LLM peut servir de
  ligne de base.

### Hypothèses à risque v4

| # | Hypothèse | Risque si fausse | Comment on la teste |
|---|---|---|---|
| V4-H1 | Translate-test perd une part mesurable de la polarité en français. | La v4 ne bat rien et se réduit à une note de coût. | Translate-test et une tête NLI multilingue prête à l'emploi sur les held-out, sans entraînement. |
| V4-H2 | Un backbone multilingue atteint en anglais les chiffres de la v3 à 2 points près. | La parité FR/EN mélange langue et backbone. | Réentraîner AUG+LEX anglais sur le backbone multilingue, dix graines. |
| V4-H3 | ~~La composition tient avec une magnitude française empruntée.~~ **Réfutée pour MeaningBERT le 2026-10-11** (Pearson 0,28 contre 0,58 pour BERTScore sur FrJUDGE). | Pas d'ancrage humain de la magnitude en français. | Corrélation des magnitudes candidates sur les 297 paires de FrJUDGE. |
| V4-H4 | L'augmentation lexicale se porte en français sans apprendre le biais du générateur. | Le gain sur les antonymes ne se reproduit pas. | Lexique coupé en deux moitiés, une moitié tenue à l'écart. |
| V4-H5 | Le train traduit n'injecte pas d'artefacts de traduction. | Bons chiffres sur XNLI-fr, mauvais sur DACCORD. | Écart XNLI-fr contre DACCORD et RTE3-FR. |

## Journal

- 2026-10-21 (fin) : **magnitude au-delà du français, proxies ouverts seulement (décision
  de David).** Aucun corpus public ne note la préservation du sens en plusieurs langues.
  On ajoute STS22 (dix langues, similarité de 1 à 4) et les jugements binaires de contenu
  de TextDetox (neuf langues, groupées par écriture), sans demande d'accès ni annotation.
  Ce sont des proxies, nommés comme tels dans le rapport ; XFORMAL et SimpEvalDE restent
  hors champ.
- 2026-10-21 (suite) : **tout tester, sans contrainte de temps (décision de David).** On
  ajoute le rejeu multilingue (MultiNLI traduit en FR, DE, AR, ES et ZH), dix graines
  partout, l'ablation sur `XLM-R-large`, WiSE-FT d'alpha 0,1 à 0,9 sur les trois régimes, et
  XNLI dans les 14 langues ; deux items russes mal alignés sortent de toutes les langues.
  Magnitude : on entraîne la recette v2-large (`d_full`, tête `clamped`) sur `mDeBERTa-v3-base`
  et `XLM-R-large` bruts, en anglais, en français traduit et sur les deux, cinq graines,
  puis on compare à BERTScore XLM-R sur FrJUDGE (test de Williams, Holm). Aucun modèle
  JUDGEBERT n'est publié, ni sur Hugging Face ni dans le dépôt GitHub. Premier signal
  WiSE-FT (`mDeBERTa` EN, une graine) : aucun alpha ne tient les deux cibles ; alpha 0,7
  passe le garde-fou (90,13 %) et égale la v3 sur DACCORD (86,23), à 4,72 points sous 1a.
  Les deux serveurs relancent la campagne seuls au démarrage (`resume_v4.sh`, cron).
- 2026-10-21 : **seconde vague, trois pistes en parallèle (décision de David).** Les têtes v4
  tiennent les garde-fous (co-hyponymes au-dessus de zéro : 85,34 à 94,40 %, contre 27,51 %
  pour `XLM-R-large` prêt à l'emploi), mais elles perdent 7,80 à 13,41 points sur DACCORD
  (DeLong apparié, p < 0,01 après Holm, sur les six configurations). Le but est d'obtenir les
  deux à la fois : DACCORD à 90,95 ou plus et co-hyponymes à 90 % ou plus. Trois pistes :
  (1) WiSE-FT, une interpolation des poids entre la tête prête à l'emploi et la tête v4 EN
  (alpha de 0,2 à 0,8, trois graines), sans entraînement, sur souris ; (2) le rejeu,
  AUG+LEX plus 90 000 paires MultiNLI équilibrées, sur `mDeBERTa` (1080 Ti) et `XLM-R-large`
  (P5000, après la grille) ; (3) l'ablation sans AUG ni LEX, puis AUG seul, sur `mDeBERTa`,
  pour situer la perte. Le rapport porte désormais des tests : DeLong contre 1a, 1b et v3,
  Welch sur les graines, binomial et McNemar sur les garde-fous, et la parité EN/FR testée
  contre la marge de 2 points. Autres langues : XNLI test en DE, AR, ES et ZH (on vise
  ensuite les 14 langues), en zéro-shot, avec translate-test par Opus-MT, sans annotation.
  Renard et souris ont redémarré le 2026-10-20 vers 6 h 30 ; les grilles ont été relancées.
- 2026-10-17 : **le recul se reproduit sur un second backbone.** `mDeBERTa-v3-base` entraîné
  sur AUG+LEX fait 76,70 à 79,22 sur DACCORD, soit 11,62 à 14,14 points sous son propre
  point de départ prêt à l'emploi (90,84). Le défaut vient de la recette AUG+LEX, pas du
  backbone ni de la langue. Le classement des régimes est le même sur les deux : EN, puis
  FR traduit, puis EN+FR. `XLM-R-large` est exclu de la GTX 1080 Ti, où il figeait 7
  entraînements sur 12 ; `mDeBERTa` y tourne sans blocage.
- 2026-10-15 : **critère 1a, verdict provisoire (2 à 3 graines par régime).** Sur DACCORD,
  `XLM-R-large` entraîné sur AUG+LEX fait 82,33 (EN), 81,23 (FR traduit) et 78,60 (EN+FR),
  soit 8,63 à 12,35 points sous le même modèle prêt à l'emploi (90,95), avec des intervalles
  loin de 0. La recette AUG+LEX dégrade la généralité de la tête, en français comme en
  anglais (v3 : 86,08 ± 1,00 sur dix graines). Ce n'est pas la langue : traduire n'aide pas
  et mélanger EN+FR fait pire. La parité XNLI FR/EN est à 2,92 à 4,28 points, pour 2 visés.
  Décision de David : finir la grille avant de trancher. Trois runs ont figé la 1080 Ti.
- 2026-10-11 : **V4-H3 réfutée pour MeaningBERT.** Sur FrJUDGE (276 paires), `MeaningBERT
  v2-large` corrèle à 0,28 [0,17 ; 0,38] avec le jugement humain après traduction et à 0,27
  en lisant le français, contre 0,58 [0,48 ; 0,66] pour `BERTScore XLM-R-large`. La
  magnitude de la v3 ne s'emprunte pas. Comme la traduction n'aide pas, l'échec vient du
  domaine (juridique, phrases longues) plus que de la langue. Le choix de la magnitude
  française (BERTScore, JUDGEBERT ou une magnitude multilingue entraînée) est reporté au
  point de décision. Résultat : `results/v4/magnitude-frjudge.json`.
- 2026-10-11 : **discovery V4-H1, premier chiffre.** Sur DACCORD, le meilleur translate-test
  (`DeBERTa-v3-large NLI` sur la traduction) fait 96,71 d'AUC, et bat de 5,76 points
  [4,09 ; 7,43] la meilleure tête multilingue qui lit le français (`XLM-R-large NLI`,
  90,95). La v3 AUG+LEX ne fait que 84,04 : sa faiblesse vient du domaine (actualité contre
  scènes de SICK), pas de la langue. Lire le français coûte 0,45 à 1,24 point d'AUC, et
  traduire 0,62 à 2,43, sur les mêmes items de SICK-fr et Breaking-fr. Le critère 1 est
  scindé : 1a contre la meilleure tête multilingue, 1b contre le meilleur translate-test.
  V4-H1 est très menacée pour 1b. Une seule graine v3 mesurée.
- 2026-10-11 : **cadrage v4.** Le chantier se recentre sur une tête de polarité
  multilingue : la magnitude pourrait s'emprunter (hypothèse V4-H3, non vérifiée). Translate-test devient la ligne de base à battre, et V4-H1 est la première
  hypothèse à vérifier, parce qu'elle décide si la v4 a une raison d'être. Le commit
  `67e5a00` est tagué `v3-arr`.
- 2026-09-19 : **H6, le resultat central du chantier.** Les 360 paires que CSMD partage
  avec SimpDA_2022 portent exactement le meme multiensemble d'etiquettes (360/360, ecart
  max 0,000000) mais correlent a 0,009 : les etiquettes sont permutees entre les paires.
  26,6 % des lignes annotees de CSMD sont donc fausses. Le plafond de Pearson qui en
  decoule est 0,826, et le sweep plafonne a 0,802 : les modeles saturaient la borne, ce qui
  explique pourquoi le Pearson ne bougeait pas de 44 M a 900 M de parametres. La cible de
  0,914 etait inatteignable sur ce corpus quel que soit le volume ajoute. Corrige par
  `src/data/corrections.py` : l'accord inter-protocole passe de 0,309 a 0,806, a marges du
  corpus rigoureusement inchangees.
- 2026-09-19 : H5 ajoutee et refutee dans la foulee. Le decoupage du corpus
  d'entrainement fuit par phrase source : 2042 lignes pour 493 phrases sources, et 91,1 %
  des lignes de test ont leur phrase source en train. `validate_datasets.py` ne voyait
  rien parce qu'il compare des paires exactes. Corrige par `src/data/splits.py`, qui
  decoupe par groupe de phrase source et reserve un vrai holdout de sanity. Consequence :
  la ligne de base v1 doit etre re-mesuree avant de juger l'apport des corpus (L3).
- 2026-09-19 : H1 confirmee et fermee. Trois defaillances : compression d'amplitude sur
  100 % des runs (pred_mean 21-36 contre un label mean de 62,66), 40-55 % du RMSE
  recuperable par recalibrage affine, et un run effondre rapporte comme predicteur de
  zero par `np.nan_to_num`. Cinq corrections C1-C5 identifiees, C1 et C3 prerequises a
  tout reentrainement. Detail dans `docs/H1-diagnostic-calibration.md`.
- 2026-09-19 : cadrage v2 ecrit. Sequence v2 corpus / v3 polarite + dissociation /
  v4 multilingue arbitree par David. Voir ROADMAP.md pour le detail des corpus.
