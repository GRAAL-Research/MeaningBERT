# Disponibilite reelle des corpus v2

Releve au 2026-09-19, par quatre agents travaillant en parallele, chacun avec pour
consigne absolue de ne jamais fabriquer de donnees et de rapporter precisement ou ca
bloque. Trois des six corpus vises n'existent pas sous la forme annoncee.

## Resume

| Corpus | Vise au cadrage | Obtenu | Echelle | Originaux uniques | Rediffusable |
|---|---|---|---|---|---|
| CSMD (reference) | 2042 | **2073** | `da100` | 493 | oui, MIT |
| SimpEval | ~13 000 jugements | **2759 paires** | `da100` | 161 | **non, non specifiee** |
| SALSA | 19 000 editions / 840 simplifications | **0** | - | 0 | bloque |
| SimpleText CLEF 2025 | ~11 452 | **561** | `error_count` | 561 | **non, inconnue** |
| PLABA | 750 abstracts juges | **806 paires** | `likert3_signed` | 116 | **non, non specifiee** |
| SynthSimpliEval | ~1040 | **0** | - | 0 | bloque |
| **Total apres fusion et dedup** | 8000 vises | **5398** | | ~1200 | |

## Les trois blocages

### SALSA : le corpus n'est publie nulle part

Le depot GitHub et les cinq notebooks d'analyse referencent tous un dossier `data/` qui
n'existe ni dans l'arbre courant, ni dans les douze commits de l'historique, ni dans
l'unique branche, ni dans l'unique fork. Aucune release. `salsa-eval.com`, l'URL citee par
l'article EMNLP pour "our data, new metric, and annotation toolkit", redirige vers le meme
depot. La fiche HuggingFace `davidheineman/lens-salsa` y renvoie aussi.

Seules **50 paires** sont reellement obtenables, le jeu de demonstration de l'interface
d'annotation, servi par `thresh.tools`. C'est 50 contre 840.

Le second blocage tient meme si le corpus reapparait : SALSA annote des **editions**, pas
des phrases, et l'agregation edition vers phrase est une decision de modelisation. L'agent
s'est arrete plutot que d'en inventer une, comme demande.

**Action possible :** ecrire aux auteurs. Hors perimetre d'un agent autonome.

### SynthSimpliEval : aucune donnee par item publiee

L'article decrit des scores de jury LLM sur environ 1040 paires et un pilote humain sur 80
paires. Ni l'un ni l'autre n'est publie item par item. Consequence directe pour
l'hypothese H3 du `PRODUIT.md` : **le point d'ancrage des 60 phrases partagees avec
SimpEval2022 n'est pas mobilisable.**

### PLABA : le corpus parallele n'a pas de jugement, les jugements sont derriere un 401

Le corpus parallele (750 abstracts, CC-BY-4.0) est public mais ne porte **aucun score de
preservation du sens** : ce sont des adaptations, pas des jugements. Les jugements experts
existent, produits par la piste TREC PLABA, mais les fichiers bruts sont sur
`trec.nist.gov/results/`, qui renvoie HTTP 401 sans identifiants de participant. L'article
ne publie que des agregats par systeme.

L'agent a trouve une source de repli reellement accessible et en a tire **806 paires**,
sur une echelle de Likert signee a 3 points `{-1, 0, 1}` absente du contrat. Il s'est
arrete plutot que de reutiliser `severity3`, qui a la meme cardinalite mais des bornes
differentes et l'orientation inverse : le reutiliser aurait inverse le signal en silence.
L'echelle `likert3_signed` a ete ajoutee au contrat en consequence.

**Point de vigilance pour H2 :** la moyenne native de PLABA est 0,9417 sur `[-1, 1]`, tres
concentree contre la borne haute. Ce corpus apportera peu de variance.

## Ce que SimpEval apporte reellement

2759 lignes, mais **748 sont des doublons exacts de paires deja dans CSMD**, ce qui etait
previsible : CSMD est construit a partir de SimpDA_2022 et d'ASSET, et SimpEval puise aux
memes sources. L'apport net est de **2011 lignes**.

Surtout, SimpEval ne porte que **161 phrases sources uniques** pour 2759 lignes, soit 17
simplifications par phrase. Sous le decoupage groupe impose par H5, ce qui compte n'est pas
le nombre de lignes mais le nombre de groupes : de ce point de vue SimpEval est un petit
corpus.

Et c'est ce recouvrement qui a permis de decouvrir H6.

## Licences : H4 est largement negative

Trois des quatre corpus obtenus ne declarent pas de licence permettant la rediffusion.
**CSMD v2 ne pourra vraisemblablement pas republier les lignes, seulement les loaders.**

C'est une contrainte de publication, pas d'entrainement : rien n'empeche d'entrainer sur
ces donnees et de publier le modele. Mais la promesse "telecharger le corpus" de la v1 ne
tiendra pas telle quelle pour les corpus ajoutes.

**A trancher avec David avant la publication.**

## Consequence sur la cible du PRODUIT.md

La cible etait "au moins 8000 paires portant un jugement humain". On atteint **5398**.

Trois lectures possibles, a arbitrer :

1. **Accepter 5398** et le dire. Le corpus est 2,64 fois plus gros que la v1, et surtout la
   correction H6 remonte le plafond de Pearson de 0,826 a un niveau bien plus haut, ce qui
   pese plus lourd que le volume manquant.
2. **Ecrire aux auteurs de SALSA et de SynthSimpliEval.** Delai incertain, gain potentiel
   d'environ 1900 lignes.
3. **Ouvrir un cinquieme corpus.** Le pre-entrainement sur WMT DA et MQM, environ 600 000
   segments avec scores humains continus d'adequation, reste sur la table et ne demande
   aucune autorisation.

L'option 1 combinee a la 3 est la plus solide : H6 a montre que la qualite d'etiquetage
pese bien plus que le volume.
