# SmolLM2 : ce qu'on a mesuré, et pourquoi il ne va pas dans l'article

Décision du 4 octobre 2026. Les trois tailles de SmolLM2 ont été retirées de la
grille v3. Ce fichier garde ce que les soixante cellules ont coûté et dit, pour
que la question ne soit pas reposée à neuf.

## La question posée

Une architecture plus grosse et d'une autre famille fait-elle mieux que les sept
encodeurs sur la classification de polarité à trois voies ? Trois tailles,
dix graines, deux conditions de corpus.

## Ce qui est mesuré

Macro-F1 sur le test stratifié de 12 000 paires, moyenne et écart-type sur dix
graines.

| Cellule | macro-F1 | Suites générées |
|---|---|---|
| SmolLM2-135M RAW | 32.27 ± 3.92 | 34.30 ± 7.82 |
| SmolLM2-135M AUG | 32.70 ± 3.98 | 39.45 ± 8.36 |
| SmolLM2-360M RAW | 34.20 ± 3.41 | 29.91 ± 5.44 |
| SmolLM2-360M AUG | 35.58 ± 4.37 | 49.43 ± 9.38 |
| SmolLM2-1.7B RAW | 48.31 ± 2.59 | 57.21 ± 5.37 |
| SmolLM2-1.7B AUG | 49.26 ± 3.57 | 78.29 ± 4.02 |

Pour comparaison : BERT-base, le plus faible des sept encodeurs, atteint
77.72 ± 0.38, et la meilleure cellule 91.10 ± 0.14. La ligne de base TF-IDF
atteint 45.64.

## Le défaut trouvé, et son poids réel

Ces soixante cellules portaient un défaut de montage. Le tokeniseur de SmolLM2
n'a pas de jeton séparateur, donc `tokenizer(premiere, seconde)` rendait la
concaténation nue :

```
'A man is playing a guitarA man is playing an instrument'
```

Aucune frontière entre les deux phrases. Les encodeurs reçoivent
`[CLS] a [SEP] b [SEP]` ; les décodeurs n'ont pas de gabarit de paire.

Le correctif est en place dans `train_polarity.py` (`needs_pair_template` et
`encode_pair`) et s'applique à tout tokeniseur sans séparateur. **Mais son poids
a été mesuré avant de relancer quoi que ce soit** : une cellule 135M avec le
gabarit `premise: … \n hypothesis: …` donne 38.01 contre 32.27 ± 3.92 sans,
soit environ +5.7 points, à peu près une graine et demie d'écart-type.

C'est réel, et c'est très loin d'expliquer les quarante-cinq points qui séparent
SmolLM2 de BERT-base. Refaire les soixante cellules coûtait vingt heures de
carte pour déplacer le 1.7B de 49.3 à peut-être 55, toujours vingt-deux points
sous le plus faible des encodeurs. La conclusion ne bougeait pas.

## Réserves sur ce constat

- La mesure du gain du séparateur porte sur **une seule graine**, avec un
  écart-type de référence de 3.9 points.
- Le gabarit retenu est un choix parmi d'autres ; un format mieux adapté aux
  décodeurs ferait peut-être mieux.
- Les hyperparamètres sont ceux des encodeurs, repris tels quels : deux époques,
  taux d'apprentissage 1e-5, lot effectif 32. Rien n'a été réglé pour un
  décodeur, et l'écart-type de 3 à 4 points sur la tâche, contre 0.14 pour les
  encodeurs, suggère que ces cellules n'étaient pas dans un régime stable.

Autrement dit : le banc montre qu'un décodeur de cette famille, sous le protocole
des encodeurs, échoue à la tâche. Il ne montre pas qu'un décodeur ne peut pas la
faire.

## Ce qui reste dans le dépôt

- `needs_pair_template` et `encode_pair` dans `train_polarity.py`, avec leurs
  tests. Le registre n'a plus de décodeur, mais le piège est réel et la garde
  coûte une comparaison par lot.
- `align_padding`, pour la même raison.
- Les soixante cellules sur caribou, sous `results/polarity/smollm2-*`.
