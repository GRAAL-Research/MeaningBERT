# v3 : l'echelle signee [-100, 100]

Decision de David, 2026-09-25. Ce document fixe ce que le nombre veut dire, avant qu'on
entraine quoi que ce soit dessus. La v2 a montre le prix d'une metrique dont on mesure mal
la cible : le Pearson annonce etait gonfle de 0,088 par des paires triviales que personne
n'avait pensees comme faisant partie du test.

## Ce que le nombre veut dire

| score | cas | exemple |
|---|---|---|
| `+100` | meme sens | « je bois du lait » / « je bois du lait » |
| `+70` | sens preserve, formulation differente | « je bois du lait » / « je consomme du lait » |
| `0` | aucun rapport | « je bois du lait » / « la reunion est a quinze heures » |
| `-100` | sens oppose | « je bois du lait » / « je ne bois pas du lait » |

Le point qui decide de tout le reste : **le negatif ne mesure pas l'absence de sens commun,
il mesure la contradiction active.** Deux phrases sans rapport valent `0`, pas `-100`. Pour
se contredire, deux phrases doivent d'abord parler de la meme chose.

Il en decoule que le score porte deux informations et non une :

- **la magnitude**, de quoi parle-t-on ensemble, c'est la metrique v2 telle quelle ;
- **la polarite**, une fois le sujet partage, est-ce qu'on dit la meme chose ou le contraire.

`-50` n'est donc pas un point sur une droite unique. C'est soit une contradiction certaine
sur un sujet a demi partage, soit une contradiction partielle, un detail inverse dans une
phrase par ailleurs fidele. Les deux se composent de la meme facon et c'est pourquoi le
score se calcule au lieu de s'apprendre.

## Pourquoi on compose au lieu d'apprendre directement

Aucun corpus ne porte d'annotation humaine signee : personne n'a jamais demande a un
annotateur de noter « a quel point ces deux phrases disent le contraire » sur une echelle
continue. Apprendre `[-100, 100]` d'un seul coup obligerait a fabriquer la cible a partir
des etiquettes categorielles, c'est-a-dire a entrainer le modele sur notre propre regle de
conversion plutot que sur un jugement humain. Le modele apprendrait la regle, et l'article
mesurerait la regle.

La composition evite ca. Les deux composantes ont chacune des etiquettes humaines
existantes et abondantes :

    signe = magnitude x (1 - 2 x p_contradiction)

avec la magnitude dans `[0, 100]` et `p_contradiction` dans `[0, 1]`. Une paire identique
donne `100 x 1 = 100`. Une paire sans rapport donne `0 x quoi que ce soit = 0`. Une
negation sur le meme sujet donne `100 x (1 - 2) = -100`. La forme exacte reste a calibrer
sur SICK, qui est le seul corpus a porter les deux dimensions sur les memes paires.

Chaque tete reste verifiable separement, ce qui n'est pas un detail : quand le score final
deviendra faux, on saura laquelle des deux a bouge.

## Les corpus, et ce qu'ils apportent vraiment

Verifies le 2026-09-25 par `src/diagnostics/releve_corpus_v3.py`, qui lit les donnees et
non les fiches. Les neuf s'ouvrent, 634 818 lignes.

| corpus | lignes | paire | etiquette | apport |
|---|---|---|---|---|
| `tals/vitaminc` | 488 904 | `claim` / `evidence` | `SUPPORTS` / `REFUTES` / `NOT ENOUGH INFO` | le gros de la polarite, editions minimales de Wikipedia |
| `google-research-datasets/paws` | 65 401 | `sentence1` / `sentence2` | `0` / `1` | fort recouvrement lexical SANS contradiction : le controle qui empeche de confondre les deux |
| `nikitam/ACES` | 36 476 | `source` / `good-translation` / `incorrect-translation` | 68 phenomenes | negation, antonymes, nombres, entites, ordre des arguments |
| `lasha-nlp/CONDAQA` | 14 182 | question + passage | `YES` / `NO` / `DON'T KNOW` | portee de la negation |
| `mteb/sickr-sts` | 9 927 | `sentence1` / `sentence2` | score continu 1 a 5 | **le pont**, moitie proximite |
| `yangwang825/sick` | 9 840 | `text1` / `text2` | `0` / `1` / `2` | **le pont**, moitie implication, memes paires |
| `sentence-transformers/stsb` | 8 628 | `sentence1` / `sentence2` | score continu | calibration de la magnitude |
| `tasksource/monli` | 1 202 | `sentence1` / `sentence2` | `entailment` / `neutral` | monotonie descendante sous negation |
| `joey234/nan-nli` | 258 | `premise` / `hypothesis` | `entailment` / `contradiction` / `neutral` | negation sous-clausale |

L'alignement des espaces d'etiquettes est direct sur les quatre corpus d'inference :
`SUPPORTS` avec `entailment`, `REFUTES` avec `contradiction`, `NOT ENOUGH INFO` avec
`neutral`.

Deux corpus ne se lisent pas tels quels. **ACES** ne porte aucune classe : il porte trois
textes et un nom de phenomene, donc la paire minimale se construit, `source` contre
`incorrect-translation` pour le cas negatif et `source` contre `good-translation` pour le
positif. Il demande aussi un nom de configuration, il en a plusieurs. **CONDAQA** est de la
comprehension de lecture, question plus passage, avec des reponses libres melees aux
`YES` / `NO` : ce n'est pas une paire de phrases et il faudra decider s'il entre du tout.

**PAWS merite une mise en garde.** Binaire paraphrase ou non, ce qui n'est pas la meme
question que la polarite : deux phrases non-paraphrases peuvent tres bien ne pas se
contredire. Il sert de source de paires a fort recouvrement lexical, et surtout de controle
negatif. Un modele qui le classe comme contradictoire a appris le recouvrement lexical et
non le sens, ce qui est exactement le defaut que LexFlip reproche aux metriques actuelles.

## Le quatrieme test de bon sens

La v2 en a trois : paires identiques a 100, paires orthogonales a 0, symetrie. La v3 en
ajoute un, et c'est lui qui justifie la version.

**Dissociation.** Sur des paires minimales a sens inverse, la metrique doit consommer une
fraction significative de son amplitude. LexFlip mesure que sur 373 perturbations qui
inversent la force legale en preservant 93 % des tokens, BERTScore et les plongements
n'utilisent que 2 a 4 % de la leur. Seuls les modeles NLI bidirectionnels bougent, 67 %.

Ce test se mesure **avant d'entrainer quoi que ce soit**, sur les modeles deja publies. Il
donne la ligne de base que la v3 doit battre, et il coute une evaluation, pas un
entrainement.

## Les experiences, dans l'ordre

1. **Diagnostic de dissociation sur l'existant.** MeaningBERT v1, v2 `large`, et le modele
   NLI de taille base. Quelle fraction de leur amplitude consomment-ils sur les paires de
   contradiction de SICK, MoNLI et NaN-NLI ? Aucune graine, aucun GPU-jour. On attend un
   echec, et il faut le chiffrer avant de le corriger.
2. **Chargeurs et contrat.** Comme en v2 : un chargeur par corpus, une seule harmonisation.
   La nouveaute est qu'il y a deux cibles et non une, donc le contrat doit porter
   `polarity_raw` a cote de `label_raw`.
3. **Tete polarite seule.** Trois classes, entrainee sur VitaminC plus SICK plus MoNLI plus
   NaN-NLI, mesuree en exactitude et en matrice de confusion. Elle doit d'abord marcher
   comme classifieur avant de servir a composer quoi que ce soit.
4. **Calibration de la composition.** Sur SICK, le seul corpus ou les deux dimensions
   portent sur les memes paires. C'est la que la forme `magnitude x (1 - 2p)` se verifie ou
   se remplace.
5. **Grille complete**, dix graines, protocole d'evaluation de la v2 augmente du test de
   dissociation.

## Ce qui reste a trancher

La licence des neuf corpus : aucun ne la declare dans ses metadonnees. La v2 avait pose la
question comme hypothese H4 et elle avait decide de ce qui etait redistribuable. A refaire
ici, avant et non apres.
