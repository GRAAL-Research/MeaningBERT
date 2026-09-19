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
3. RMSE sur echelle 0-100 sous 15, et R2 positif. Le sweep actuel est a 35-54 de RMSE et
   R2 negatif ; c'est le symptome de calibration a corriger.
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
| H1 | L'anomalie du sweep (R2 negatif, RMSE 40, Identical=100% a 0,000) vient de la calibration de sortie, pas d'un defaut de donnee ou de modele. | On ajoute 8000 paires et on ne voit aucun gain parce que le pipeline est casse en aval. | Phase 1, diagnostic isole avant tout ajout de corpus. |
| H2 | Les scores de SimpEval, SALSA, SimpleText et PLABA sont harmonisables sur une echelle 0-100 commune sans detruire le signal. | Le corpus fusionne est plus bruite que CSMD seul et la metrique se degrade. | Ablation : entrainer sur CSMD seul vs CSMD + chaque corpus, un a un. |
| H3 | Les 60 phrases partagees entre SimpEval2022 et SynthSimpliEval suffisent comme point d'ancrage inter-corpus. | L'harmonisation repose sur une normalisation arbitraire par corpus. | Mesurer l'accord sur les paires communes avant de fixer le mapping. |
| H4 | Les licences des cinq corpus permettent une rediffusion dans CSMD v2. | On ne peut pas republier le corpus, seulement les loaders. | Verification licence par licence en phase 1. |

## Metrique du resultat vise, instrumentation

Le test externe et les sanity checks tournent dans la CI a chaque entrainement, pas
seulement a la fin. Un entrainement qui ne produit pas ces quatre chiffres est un
entrainement non mesure.

## Journal

- 2026-09-19 : cadrage v2 ecrit. Sequence v2 corpus / v3 polarite + dissociation /
  v4 multilingue arbitree par David. Voir ROADMAP.md pour le detail des corpus.
