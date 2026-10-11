## DACCORD, le seul corpus écrit en français

| Modèle | Entrée | AUC | Contradictions < 0 | Compatibles < 0 | Barre |
|---|---|---|---|---|---|
| `DeBERTa-v3-large NLI` | traduit | 96.71 | 82.33 | 4.05 | 1b |
| `XLM-R-large NLI` | français | 90.95 | 69.51 | 6.74 | 1a |
| `mDeBERTa-v3-base NLI` | français | 90.84 | 69.32 | 7.71 |  |
| `v3 AUG+LEX` | français | 84.70 | 49.90 | 1.54 |  |
| `v3 AUG+LEX` | traduit | 84.04 | 53.59 | 3.08 |  |

Barre 1a : meilleure tête multilingue prête à l'emploi qui lit le français. Barre 1b : meilleur translate-test.

## AUC par corpus

| Modèle | Entrée | DACCORD | SICK-fr | Breaking-fr | RTE3-FR | GQNLI-fr | FraCaS-fr |
|---|---|---|---|---|---|---|---|
| `DeBERTa-v3-large NLI` | anglais d'origine | - | 99.77 | 99.51 | - | 84.07 | 97.11 |
| `DeBERTa-v3-large NLI` | traduit | 96.71 | 99.15 | 97.08 | 95.73 | 83.10 | 95.92 |
| `XLM-R-large NLI` | anglais d'origine | - | 99.69 | 99.41 | - | 45.12 | 88.21 |
| `XLM-R-large NLI` | français | 90.95 | 99.24 | 98.17 | 84.07 | 42.59 | 90.80 |
| `mDeBERTa-v3-base NLI` | anglais d'origine | - | 99.30 | 91.13 | - | 50.76 | 89.56 |
| `mDeBERTa-v3-base NLI` | français | 90.84 | 98.74 | 88.92 | 86.60 | 54.22 | 91.26 |
| `v3 AUG+LEX` | anglais d'origine | - | 99.55 | 99.03 | - | 72.42 | 93.62 |
| `v3 AUG+LEX` | français | 84.70 | 97.46 | 96.41 | 88.76 | 68.81 | 93.32 |
| `v3 AUG+LEX` | traduit | 84.04 | 98.11 | 94.79 | 90.25 | 73.69 | 92.68 |

## Breaking-fr, part sous zéro par famille lexicale

| Modèle | Entrée | antonymes | co-hyponymes | nombres | synonymes |
|---|---|---|---|---|---|
| `DeBERTa-v3-large NLI` | anglais d'origine | 85.24 | 77.79 | 92.79 | 0.00 |
| `DeBERTa-v3-large NLI` | traduit | 77.36 | 71.48 | 91.62 | 2.68 |
| `XLM-R-large NLI` | anglais d'origine | 91.25 | 90.84 | 99.34 | 0.45 |
| `XLM-R-large NLI` | français | 79.93 | 72.49 | 95.63 | 0.22 |
| `mDeBERTa-v3-base NLI` | anglais d'origine | 64.52 | 52.52 | 84.27 | 0.00 |
| `mDeBERTa-v3-base NLI` | français | 60.20 | 37.86 | 81.72 | 0.34 |
| `v3 AUG+LEX` | anglais d'origine | 83.98 | 9.29 | 78.51 | 0.00 |
| `v3 AUG+LEX` | français | 82.56 | 14.39 | 93.01 | 0.67 |
| `v3 AUG+LEX` | traduit | 76.65 | 9.72 | 81.72 | 1.90 |

Synonymes sous zéro = fausses alarmes. Co-hyponymes : Breaking-NLI les étiquette contradiction, la v3 est conçue pour les garder au-dessus de zéro.
