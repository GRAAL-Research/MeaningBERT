# Cadrage v4 : l'échelle signée hors de l'anglais

Cadrage du 2026-10-11. Point de départ : tag `v3-arr` (`67e5a00`), l'article soumis à ACL ARR
octobre 2026. Décisions de David reprises telles quelles ; ce document les rend vérifiables.

## Solution ancrée

> « Rendre MeaningBERT multilingue » : remplacer les backbones anglais par XLM-R ou mDeBERTa,
> réentraîner sur des corpus multilingues, en commençant par le français.

Réévaluée à la fin (section « Réévaluation »).

## Regarder en dedans

**Le problème, en symptômes.** Une personne qui évalue un système de simplification en
français n'a aucune métrique capable de représenter l'opposition. BERTScore multilingue,
LaBSE ou JUDGEBERT rendent un score non négatif, et la v3 a montré sur l'anglais que ce
type de score ne sépare pas une négation d'une paraphrase (AUC de MeaningBERT : 60,81 % sur
notre test, 48,01 % sur SICK public). Rien n'indique que le français fasse mieux, et rien
ne l'a mesuré.

**Pourquoi ce n'est pas déjà résolu.**
- La v3 vient de sortir et elle est entièrement anglaise : backbones (`BERT-base` pour la
  magnitude, `DeBERTa-v3-large` pour la polarité), corpus, lexiques et diagnostics.
- Les corpus d'inférence, qui portent la tête de polarité, sont abondants en anglais et
  rares ailleurs. Le train de XNLI est MNLI traduit automatiquement.
- Le gain de la v3 sur les antonymes vient de lexiques écrits à la main pour le vocabulaire
  de SICK. Ils ne se traduisent pas.

**Notre part du problème.** On suppose que le multilingue demande de réentraîner. Or la v3
affirme elle-même qu'elle fonctionne « avec une métrique inchangée » : la composition
`magnitude x (1 - 2 p_contra)` accepte n'importe quelle magnitude. Le vrai manque en
français pourrait se réduire à la tête de polarité. On suppose aussi que la traduction
automatique abîme la négation et les antonymes ; c'est l'hypothèse que translate-test
réfute ou confirme à coût presque nul.

## Regarder en dehors

**Qui le vit.** Les chercheurs qui évaluent la simplification ou la génération en français,
en allemand ou en arabe (DEplain, SAMER, FrJUDGE), et qui retombent sur une métrique de
similarité au moment de comparer des systèmes. La conséquence concrète : un système qui
inverse le sens d'une phrase n'est pas pénalisé.

**Qui ne l'a pas.** Les anglophones, depuis la v3. Ceux qui paient une campagne
d'annotation humaine. Ceux qui traduisent tout vers l'anglais avant d'évaluer : ils n'ont
pas le problème si la traduction préserve la polarité, d'où le critère translate-test.

**À qui profite le statu quo.** Aux métriques non signées déjà établies, qui restent le
choix par défaut. Et à translate-test : s'il suffit, personne n'a besoin de la v4.

**Laissés de côté.** Les langues sans corpus d'inférence natif, qui ne seront couvertes qu'à
l'étape 3 (XNLI, traduit). Les paires interlingues, repoussées en expérience secondaire.

## Recadrage

Autrement dit, le problème est : **dans une autre langue que l'anglais, rien ne dit si une
réécriture contredit sa source, et on ignore si traduire vers l'anglais suffit pour le dire.**

Comment pourrions-nous **donner une polarité native en français** afin que **le score
signé y soit aussi fiable qu'en anglais, et plus fiable que de traduire avant d'évaluer** ?

## Job to be done

> Quand j'évalue des simplifications en français, je veux savoir si une sortie dit le
> contraire de sa source, afin de pénaliser un système qui inverse le sens sans lancer
> d'annotation humaine.

- **Freins actuels** : il n'y a pas de métrique signée hors de l'anglais ; traduire avant
  d'évaluer ajoute un système de traduction dont il faut justifier la fidélité ; et les
  reviewers demandent pourquoi on n'a pas traduit.
- **Gains visés** : un score par paire, dans la langue, avec le même sens qu'en anglais
  (0 sans rapport, sous zéro pour une contradiction).
- **Job social** : publier une évaluation non anglaise sans qu'on y voie une évaluation de
  seconde zone.

## Lean UX Canvas

1. **Problème d'affaires** : l'échelle signée de la v3 n'existe qu'en anglais, alors que la
   simplification est surtout étudiée là où l'anglais n'est pas la langue des lecteurs.
2. **Résultat d'affaires (GRAAL / Baseline)** : un article soumis à ARR février 2027, et une
   tête de polarité française publiée sur Hugging Face.
3. **Utilisateurs** : les chercheurs en évaluation de la simplification non anglophones ; en
   second lieu, la communauté francophone du traitement automatique des langues.
4. **Résultat utilisateur** : un score signé en français qui classe les contradictions sous
   zéro aussi bien qu'en anglais.
5. **Solutions (au moins trois)** :
   - A. **Translate-test** : traduire en anglais, puis appliquer la v3 telle quelle. Elle
     est gratuite, et c'est la ligne de base à battre.
   - B. **Tête de polarité NLI multilingue prête à l'emploi** (mDeBERTa-v3 XNLI), sans
     entraînement. C'est l'équivalent français de `DeBERTa-v3-large-MNLI`.
   - C. **Translate-train** : traduire le corpus de polarité de la v3 (et les paires
     générées), puis entraîner un backbone multilingue dessus.
   - D. **Tête de polarité native** : corpus multilingue et augmentation lexicale française,
     avec une magnitude empruntée (JUDGEBERT, BERTScore multilingue ou MeaningBERT
     traduit).
   - E. **Zéro-shot interlingue** : la v3 entraînée sur un backbone multilingue en anglais
     seulement, puis testée en français.
6. **Hypothèses** : voir le classement ci-dessous.
7. **À apprendre en premier** : est-ce que translate-test perd vraiment la polarité en
   français ?
8. **Plus petit travail** : appliquer A et B sur DACCORD, RTE3-FR, GQNLI-FR et FraCaS-fr,
   sans rien entraîner. Une journée.

## Hypothèses classées par risque

| # | Hypothèse | Si fausse, on perd quoi | Confiance | Coût pour vérifier |
|---|---|---|---|---|
| V4-H1 | Translate-test perd une part mesurable de la polarité en français (négation discontinue, antonymes, « ne » explétif). | La raison d'être de la v4 : une tête native ne bat rien, et l'article devient une note de coût et de latence. | Faible | Une journée : A et B sur les held-out français, sans entraînement. |
| V4-H2 | Un backbone multilingue (mDeBERTa-v3-base ou XLM-R-large) atteint en anglais les chiffres de la v3 à 2 points près. | La parité (critère 2) n'a pas de référence, et une défaite contre translate-test ne se distingue plus d'un effet de taille du modèle. | Moyenne | Un barreau : réentraîner AUG+LEX anglais sur le backbone multilingue, dix graines. |
| V4-H3 | La composition tient avec une magnitude française empruntée, validée sur les 297 paires de FrJUDGE. | La moitié magnitude n'a pas d'ancrage humain en français : FrJUDGE est petit et juridique. | Faible | Corrélation de chaque magnitude candidate sur FrJUDGE, puis composition sur DACCORD. |
| V4-H4 | L'augmentation lexicale se porte en français avec un lexique d'antonymes français (WOLF, JeuxDeMots ou écrit à la main) sans que la tête apprenne le biais du générateur. | Le gain sur les antonymes, le résultat le plus fort de la v3, ne se reproduit pas. | Moyenne | Lexique coupé en deux moitiés comme en v3. L'entraînement et le test tirent leurs paires de ressources distinctes (par exemple JeuxDeMots pour l'entraînement, WOLF pour le test) pour ne pas partager le même bruit. Le résultat sur les antonymes reste secondaire et soumis au biais du générateur, déclaré en limitation. |
| V4-H5 | Le train traduit (XNLI, VitaminC traduit) n'injecte pas d'artefacts de traduction que la tête exploite. | Bons chiffres sur XNLI-fr, mauvais sur DACCORD : la tête aurait appris la traduction. | Moyenne | Écart XNLI-fr contre DACCORD, le seul test natif ; le domaine diffère aussi, donc c'est un indice, pas une preuve. |
| V4-H6 | Les licences de DACCORD, RTE3-FR, GQNLI-FR et FraCaS-fr permettent l'évaluation et la rediffusion des scores. | Des tests non publiables. | Élevée | Vérification licence par licence au relevé. |

Les trois premières sont l'intrant de la discovery. V4-H1 passe en premier : c'est la
moins confiante et la moins chère à vérifier, et c'est elle qui décide si la v4 existe.

## Corpus français tenus à l'écart (jamais en entraînement)

| Corpus | Taille | Mesure | Nature |
|---|---|---|---|
| DACCORD (Skandalis et al., LREC-COLING 2024) | 1 034 | AUC, signe | contradictions écrites en français |
| RTE3-FR | 800 (test) | macro-F1, AUC | implication, traduction humaine |
| GQNLI-FR | 300 | précision | quantificateurs généralisés |
| FraCaS-fr | petit | précision | phénomènes, l'équivalent de MoNLI |
| XNLI-fr test | 5 010 | macro-F1 | traduction humaine, sert de comparaison avec le train traduit |
| FrJUDGE (Beauchemin et al., EMNLP 2025) | 297 | Pearson magnitude | simplification juridique, jugement humain, CC-BY 4.0 |
| Paires minimales FR (générées) | environ 500 | part sous zéro | négation, antonymes inédits, synonymes, co-hyponymes ; générées par substitution d'un mot à partir d'une ressource distincte de celle de l'entraînement, sans validation humaine : elles servent de garde-fou et de mesure rapportée, elles ne décident pas |

## Critères de passage (fixés avant toute mesure)

Seuls des corpus déjà annotés et des paires générées entrent en jeu. Le seul corpus de
polarité écrit en français est DACCORD. RTE3-FR, GQNLI-FR, FraCaS-fr et XNLI-fr sont
traduits de l'anglais : retraduire vers l'anglais retrouve souvent l'original, ce qui
avantage translate-test. Ils sont donc rapportés, mais ne décident pas du critère 1.

**Critère 1, battre translate-test.**
- L'AUC sur DACCORD doit dépasser translate-test d'au moins 2 points, avec un intervalle
  bootstrap à 95 % qui exclut 0.
- C'est **la seule mesure qui décide**. Les négations et les antonymes sous zéro viennent
  des paires générées : ils sont rapportés contre translate-test, mais ne décident pas,
  parce que le générateur n'est validé par personne.
- Les antonymes testés sont **inédits pour les deux concurrents** : leur traduction anglaise
  n'apparaît dans aucune moitié du lexique de la v3 (ni dans `FACTUAL_ANTONYMS`), et ils
  appartiennent à la moitié française tenue à l'écart. L'ensemble est généré **avant** la
  discovery, ce qui avance la coupe du lexique de V4-H4.
- Le translate-test qui décide est la v3 publiée (`DeBERTa-v3-large`), celle qu'un
  praticien utiliserait. Une variante sur le même backbone multilingue est rapportée pour
  séparer ce que l'écart doit à la taille du modèle et ce qu'il doit à la langue.
- La tête entraînée doit aussi battre B, la tête NLI multilingue prête à l'emploi. Sinon,
  la contribution se limite à « B composé avec une magnitude ».

**Critère 2, parité avec l'anglais.**
- Mesurée sur des items appariés : le test de XNLI, dont les items anglais et français sont
  les mêmes, et des paires minimales générées des mêmes gabarits dans les deux langues.
  Cela évite de comparer DACCORD à SICK, deux corpus de difficulté différente.
- La référence est la sortie du barreau V4-H2 : AUG+LEX anglais réentraîné sur le même
  backbone multilingue, dix graines. Le français doit être au moins à 2 points sous
  l'anglais en AUC et en macro-F1.

**Garde-fous contre les fausses alarmes, pour les deux critères.** Un garde-fou bloque, il
ne prouve rien : c'est le seul usage décisionnel des paires générées. Au moins 95 % des
synonymes et hyperonymes et au moins 90 % des co-hyponymes restent au-dessus de zéro (94,85 %
avec LEX en v3). C'est le piège que la v3 a montré : la tête prête à l'emploi plaçait 74,70 %
des co-hyponymes sous zéro.

**Magnitude.** Le Pearson sur FrJUDGE de la magnitude empruntée doit être au moins celui
de BERTScore multilingue. JUDGEBERT est exclu de cette mesure s'il a été entraîné sur
FrJUDGE. Tant que ce critère n'est pas passé, « la magnitude s'emprunte » reste une
hypothèse (V4-H3).

## Non-objectifs

- **Pas d'allemand ni d'arabe tant que le français n'a pas passé ses deux critères.** Les
  corpus DEplain et SAMER sont prêts et tentants, mais porter trois langues avant de savoir
  si translate-test suffit triplerait un travail qui pourrait être inutile.
- **Pas d'interlingue dans le résultat visé.** Une paire source EN / cible FR change ce que
  le score veut dire : la métrique deviendrait aussi une métrique de traduction. Elle reste
  possible en expérience secondaire (section, annexe ou autre article) une fois le
  monolingue établi.
- **Aucune annotation ni validation humaine de données, pour aucune langue.** Décision de
  David. Tout repose sur des corpus déjà annotés et sur des paires générées à partir de
  ressources existantes. Conséquence assumée : les paires minimales générées ne sont qu'une
  mesure secondaire, la preuve principale vient de DACCORD, RTE3-FR, GQNLI-FR et FraCaS-fr,
  et le biais du générateur est déclaré en limitation.
- **Pas de changement de la formule.** `magnitude x (1 - 2 p_contra)` reste celle de la v3.
  Recalibrer la forme en même temps que la langue rendrait tout écart inattribuable.
- **Pas de passage au document, pas de LLM-as-judge comme métrique.** Un LLM peut servir de
  ligne de base, pas de professeur.
- **Pas de modèle unique monolingue français (CamemBERT) comme cible.** Il peut servir de
  barreau de comparaison, mais l'étape 3 exige un backbone multilingue, donc le pilote doit
  le valider.

## Réévaluation de la solution ancrée

**Oui en partie.** Le cadrage déplace le centre du chantier : la v4 est d'abord une **tête
de polarité** multilingue, pas un MeaningBERT multilingue entier. La magnitude pourrait
s'emprunter, mais c'est l'hypothèse V4-H3, pas un fait. Et avant tout entraînement,
translate-test et la tête NLI prête à l'emploi doivent être mesurés.

**Point de décision, à la fin de la discovery (vers le 2026-10-23).** Il est déclenché si
translate-test ou la tête prête à l'emploi (B) arrive à moins de 2 points d'AUC, sur
DACCORD, de ce qu'une tête française peut atteindre. La suite sera tranchée par David à ce
moment-là, au vu des chiffres : article de mesure, pivot interlingue ou abandon de février.
Les non-objectifs restent en vigueur jusqu'à cette décision.

## Séquence

1. **Discovery (une semaine)** : V4-H1 (A et B sur les held-out), V4-H3 (magnitudes sur
   FrJUDGE), relevé et licences des corpus français, génération des paires minimales.
2. **Barreau backbone** (V4-H2), puis les régimes C, D et E sur le français.
3. **Article ARR février 2027.**
4. Étape 2 (FR + DE + AR), puis étape 3 (XNLI et PAWS-X à grande échelle).

## Sources

- Skandalis et al. 2024, DACCORD, RTE3-FR, GQNLI-FR : https://aclanthology.org/2024.lrec-main.1065
- Beauchemin et al. 2025, JUDGEBERT et FrJUDGE : https://aclanthology.org/2025.emnlp-main.5
- COLE, banc d'essai français qui reprend DACCORD : https://arxiv.org/abs/2510.05046
