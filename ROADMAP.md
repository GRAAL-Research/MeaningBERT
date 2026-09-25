# MeaningBERT - Feuille de route v2 / v3 / v4

Decidee le 2026-09-19. Ordre ferme : v2 d'abord et on pousse, v3 ensuite avec publication,
v4 en dernier.

## Etat v1

- Corpus : CSMD, 1355 paires annotees (ASSET, QuestEval, SimpDA_2022, Simplicity-DA)
  + 359 identiques (score 100) + 359 non reliees (score 0).
- Modele : bert-base-uncased, tete de regression num_labels=1, sortie non bornee, 0-100.
- Sanity checks : identique >= 95, non reliee <= 5.

## Blocage a lever avant la v2

1. `results/sweep_summary.csv` : R2 de -1,109 a +0,108, RMSE de 35 a 54 sur echelle 0-100,
   `Identical =100%` a 0,000 sur les 8 configurations, alors que Pearson tient a 0,78-0,80.
   Bon ordonnancement + erreur absolue enorme = probleme d'echelle ou de calibration de la
   sortie, pas du modele. A diagnostiquer avant d'interpreter le sweep.
2. `prepare_datasets.py` verse identiques et non reliees dans le pool avant le split
   stratifie. Pas de fuite ligne a ligne (validee par `validate_datasets.py`), mais les
   sanity checks deviennent in-distribution : ils ne mesurent plus ce qu'ils mesuraient
   dans l'article. Decider si on garde ce design ou si on restaure un vrai holdout.

## v2 - Corpus, anglais, architecture inchangee

Objectif : lever le plafond des 1355 paires annotees. Le sweep montre que le backbone
n'est pas le goulot (Pearson 0,777 a 0,804 de deberta-v3-small a deberta-v2-xlarge).

Corpus a integrer, par ordre de rendement :

| Corpus | Volume | Apport |
|---|---|---|
| SimpEval_past + SimpEval_2022 (LENS, ACL 2023) | 13K jugements sur 2,8K simplifications | Volume x10 sur la meme tache. github.com/Yao-Dou/LENS |
| SALSA (EMNLP 2023) | 19K annotations d'edition sur 840 simplifications | Signal au niveau de l'edition, severite 1-3. salsa-eval.com |
| CLEF SimpleText 2025 tache 2.2 | 11 452 phrases complexes + simplifications | Taxonomie de distorsion d'information. Remplit la queue basse absente de CSMD. |
| PLABA / TREC PLABA 2023-2024 | 750 abstracts, adaptations expertes | Biomedical, annotateurs experts, axes Accuracy et Completeness. |
| SynthSimpliEval (2025) | 1040 simplifications notees | Concu pour corriger les defauts d'annotation des benchmarks existants. |

Pre-entrainement optionnel avant fine-tune CSMD : WMT DA + MQM (~600k segments avec
scores humains continus d'adequation, jusqu'a WMT 2023, le carburant de COMET).

Test externe, pas d'entrainement : Agrawal & Carpuat TACL 2024 (preservation du sens
mesuree par comprehension de lecture).

Travaux techniques v2 :
- Harmoniser les echelles d'annotation entre corpus (Likert 5 points, 0-100, severite 1-3).
- Calibration : sigmoide x 100 ou regression beta + isotonie sur dev.
- Entrainer sur la distribution des annotateurs plutot que sur leur moyenne, sortir un
  intervalle. Les labels CSMD portent deja leur variance.
- Republier CSMD v2 et le modele sur HuggingFace.

## v3 - Polarite et dissociation, avec article

Constat qui motive la v3 : LexFlip (arxiv 2609.05296) montre que les sanity checks
identique / non reliee sont satisfaits par toute fonction monotone du recouvrement
lexical. Sur 373 perturbations minimales qui inversent la force legale en preservant 0,93
des tokens, BERTScore et les embeddings n'utilisent que 0,022 a 0,039 de leur amplitude.
Seuls les modeles NLI bidirectionnels bougent (0,670).

Deux chantiers, lies :

1. **Benchmark de dissociation anglais.** Paires minimales a sens inverse. Critere
   d'acceptation : la metrique doit consommer une fraction significative de son amplitude.
   Troisieme sanity check, a cote des deux existants.
2. **Architecture a deux tetes.** Tete magnitude (0-100, MeaningBERT v2 tel quel) + tete
   polarite (entail / neutral / contradict). Score signe = magnitude x (1 - 2 p_contradiction),
   forme a calibrer.

Corpus pour la tete polarite :

| Corpus | Volume | Nature |
|---|---|---|
| VitaminC (`tals/vitaminc`) | 450k+ paires claim-evidence contrastives, 100k revisions Wikipedia | Paires quasi identiques ou un detail factuel inverse le verdict. Le plus gros corpus de paires minimales existant. |
| ACES (EdinburghNLP/ACES) | 36 476 exemples, 68 phenomenes | Antonymes, negation, omission, ajout, nombres, entites, ordre des arguments. |
| PAWS / PAWS-X | 108k | Fort recouvrement lexical, non-paraphrase. |
| CAD / CF-SNLI (Kaushik) | ~13k paires revisees a la main | Editions minimales qui changent la relation. |
| MoNLI, NaN-NLI, CondaQA | Petits, cibles | Portee de la negation, monotonie descendante, negation sous-clausale. |
| SICK | 10k | Relatedness continue ET label entailment/contradiction sur les memes paires. Pont naturel entre les deux tetes. |
| LexFlip | 373 | Trop petit pour entrainer. Test de sortie. |

Complement generable : inversion de polarite synthetique sur CSMD (insertion de negation,
substitution d'antonymes, permutation d'arguments, changement de nombre ou de date).
Valider un echantillon avec des humains, sinon on entraine le modele sur le biais du
generateur.

Option ecartee pour l'instant : echelle signee [-100, 100] apprise directement. Aucun
corpus n'a d'annotations humaines signees et la semantique de -50 est mal definie.

Option gardee en reserve : penalites typees a la MQM (100 moins des penalites par type
d'erreur), entrainees sur ACES + SimpleText 2.2 + SALSA. Sortie interpretable, negatifs
possibles, mais refonte complete de la cible.

A inclure dans l'article v3 : baseline LLM-as-judge (un reviewer de 2026 la demandera),
et sortie localisee du segment qui perd le sens, facon LENS-SALSA au niveau du mot.

## v4 - Multilingue

Backbone XLM-R ou mDeBERTa. Corpus : DEplain et DETECT (allemand), German4All, SAMER
(arabe), benchmark multilingue non-anglais (Ryan et al.), corpus italien, francais
biomedical, WMT DA multilingue. Point d'appui francais : FrJUDGE et JUDGEBERT
(arxiv 2508.16870).

## Hors sequence

Extension au document (ModernBERT, 8k de contexte, ou decoupage + agregation).
MeaningBERT est phrase a phrase alors que la simplification reelle est document.
