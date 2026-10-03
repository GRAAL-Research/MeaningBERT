# Intégrer SmolLM2 dans l'article v3

Le banc tourne sur caribou : trois tailles de SmolLM2 × dix graines × deux conditions,
soixante cellules. Ce fichier liste ce qu'il faudra toucher quand elles arriveront, pour
qu'aucun passage ne reste sur l'ancien compte.

## Garde-fou déjà en place

`analyse_v3.py` refuse de produire les tableaux si une cellule entraînée porte un tag
absent de `ARCH_LABELS` :

```
cellules entrainees sans etiquette dans ARCH_LABELS, elles seraient ignorees
en silence : <tag>
```

Les trois tailles de SmolLM2 y sont déjà déclarées, donc le garde-fou ne se déclenchera
pas pour elles. Il protège le prochain ajout.

## Les treize passages à réviser

Le compte passe de sept encodeurs à dix, de 140 cellules à 200, et la famille de tests de
Holm de quatorze à vingt.

| # | Section | Ce qui est écrit | Ce que ça devient |
|---|---|---|---|
| 1 | Résumé | « on all seven encoders » | dix |
| 2 | Experimental Setup | « seven encoders × ten seeds × two corpus conditions, for 140 runs » | dix, 200 |
| 3 | Experimental Setup | « Encoders span three families, BERT, RoBERTa and DeBERTaV3 » | quatre familles, et SmolLM2 est un décodeur, pas un encodeur |
| 4 | Experimental Setup | « the grid is roughly 500 GPU-hours » | à recalculer, les cellules Ada sont bien plus rapides |
| 5 | Experimental Setup | « across the fourteen condition tests » | vingt |
| 6 | Results | « on all seven encoders, with d from 9.9 to 81.1 » | dix, bornes à recalculer |
| 7 | Results | « seven encoders trained without these rows » | dix |
| 8 | Limitations | « drops on three of seven encoders » | à recompter |
| 9 | Annexe B | « identical for all 140 runs » | 200, et la précision n'est plus identique partout |
| 10 | Annexe B | « what makes the seven encoders comparable » | dix |
| 11 | Annexe C | « three NVIDIA Pascal-generation cards » | plus trois RTX 6000 Ada |
| 12 | Annexe C | « Two of the seven encoders wedge the GTX 1080 Ti » | sept des dix, la contrainte ne concerne que le parc Pascal |
| 13 | Annexe C | « The 140 runs take roughly 500 GPU-hours » | à recalculer |

## La différence de protocole, à déclarer

Les sept encodeurs tournent en fp32 sur Pascal, les trois SmolLM2 en bf16 sur Ada. Le lot
effectif reste 32 partout et rien d'autre ne change, mais la précision n'est pas la même
et cela se dit plutôt que de se cacher. Une phrase dans l'Experimental Setup et une ligne
dans l'annexe B suffisent.

Le titre « Encoders span three families » devient faux pour une autre raison : SmolLM2 est
un décodeur. La phrase doit nommer la quatrième famille comme telle, puisque c'est
précisément ce qu'elle apporte au banc.

## Ordre de travail

1. Vérifier que les soixante cellules sont là et que `analyse_v3.py` ne lève pas.
2. Régénérer les quatre tableaux et la figure 2.
3. Recalculer les bornes citées dans le texte : les `d`, les plages de macro-F1 et de
   suites de contrôle, le compte d'encodeurs où MoNLI baisse.
4. Réviser les treize passages ci-dessus.
5. Recalculer le budget de calcul à partir des durées réelles des logs de caribou.
6. Relancer `/review-acl` : un banc qui passe de sept à dix entrées change ce qu'un
   relecteur attend de la section Résultats.
