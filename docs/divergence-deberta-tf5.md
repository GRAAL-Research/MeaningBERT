# DeBERTa-v3-large diverge sous transformers 5.x

## Ce qui s'est passé

Le 4 octobre 2026, les neuf graines d'AUC lancées sur caribou ont toutes divergé.
Chacune a tenu une carte trois heures et demie, puis écrit un `metrics.json`
d'apparence normale portant un macro-F1 de 16.67, soit exactement le plancher de
la classe majoritaire. Six sont entrées dans une moyenne avant que quiconque
ouvre la matrice de confusion, qui mettait les 12 000 paires de test dans la
seule classe `entailment`.

Signature dans le journal :

```
{'loss': 3.086e+12, 'grad_norm': 'nan', 'epoch': 0.021}
{'loss': 0,         'grad_norm': 'nan', 'epoch': 0.042}
```

La perte est rapportée à zéro alors que le gradient est déjà `nan`, ce qui rend
la divergence invisible à qui ne lit que la perte.

## Ce que ce n'est pas

Trois hypothèses ont été testées et écartées, dans cet ordre.

**La précision.** Le bf16 garde les huit bits d'exposant du fp32 ; il perd de la
mantisse, pas de la dynamique. Une passe avant et arrière sur des paires réelles
donne une perte de 0.0040 en fp32 contre 0.0031 en bf16, avec des normes de
gradient de 0.61 et 0.18. Et surtout, l'entraînement complet diverge dans les
**deux** précisions, `grad_norm` à `nan` au pas 100 dans les deux cas.

**L'implémentation d'attention.** DeBERTa-v3 n'expose pas SDPA
(`_supports_sdpa = False`) et tourne déjà en `eager`. Rien à désactiver.

**`align_padding`.** La fonction sort immédiatement quand le tokeniseur et le
modèle ont tous deux un jeton de remplissage, ce qui est le cas de DeBERTa-v3.
Elle ne touche pas ce modèle.

## La cause

La version de `transformers`.

| Machine | transformers | torch | DeBERTa-v3-large |
|---|---|---|---|
| renard | 4.57.6 | 2.14.0+cu126 | apprend, 91.10 de macro-F1 sur dix graines |
| caribou | 5.18.0 | 2.6.0+cu124 | `nan` au pas 100 |

Vérifié directement : un venv épinglé à 4.57.6 sur **la même machine**, le même
corpus, les mêmes hyperparamètres et la même carte donne une perte de 0.5176,
0.4037, 0.3928, 0.3968 sur les quatre premiers paliers, avec des normes de
gradient entre 6.7 et 11.8. SmolLM2 converge sous 5.18.0 sur la même machine,
donc le défaut est propre à DeBERTa-v3.

## Ce qui a été fait

1. `requirements.txt` borne `transformers` sous la version 5.
2. `train_polarity.py` reçoit deux gardes, parce que les deux échecs sont
   distincts :
   - `divergent_log` coupe dès la première ligne de journal où la perte ou la
     norme de gradient cesse d'être finie. La cellule meurt en deux minutes au
     lieu de trois heures et demie.
   - `degenerate` refuse d'écrire le `metrics.json` quand les prédictions
     tombent dans une seule classe ou que le macro-F1 n'est pas fini. Rien n'est
     écrit avant qu'on sache que c'est un résultat.
3. Les six cellules effondrées sont déplacées dans `results/rebut-nan/`, comme
   pièces du dossier, plutôt que supprimées.
4. Les neuf graines sont relancées depuis `~/.venvs/mb-tf4`, épinglé à 4.57.6,
   en fp32 comme les cellules publiées, avec un micro-lot de 8 au lieu de 4 :
   l'accumulation tient le lot effectif à 32, ce que l'annexe B donne comme
   condition de comparabilité entre machines.

## Ce qui reste ouvert

La ligne précise de `transformers` 5.x qui casse n'est pas identifiée. La borne
dans `requirements.txt` est une mise à l'écart, pas un correctif amont. Si
quelqu'un veut remonter la piste, la divergence apparaît entre le pas 1 et le
pas 100 d'un entraînement complet mais **pas** sur une passe avant et arrière
isolée, ce qui pointe vers l'optimiseur, l'accumulation de gradient ou le
remplissage par lots plutôt que vers le modèle seul.
