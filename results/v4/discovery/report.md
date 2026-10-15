## DACCORD, le seul corpus écrit en français

| Modèle | Entrée | AUC | Contradictions < 0 | Compatibles < 0 | Barre |
|---|---|---|---|---|---|
| `DeBERTa-v3-large NLI` | traduit | 96.71 | 82.33 | 4.05 | 1b |
| `XLM-R-large NLI` | français | 90.95 | 69.51 | 6.74 | 1a |
| `mDeBERTa-v3-base NLI` | français | 90.84 | 69.32 | 7.71 |  |
| `v3 AUG+LEX` | français | 86.08 | 52.62 | 0.77 |  |
| `v3 AUG+LEX` | traduit | 85.97 | 56.31 | 2.31 |  |
| `XLM-R-large AUG+LEX (EN)` | français | 82.33 | 52.82 | 6.74 |  |
| `XLM-R-large AUG+LEX (FR traduit)` | français | 81.23 | 52.23 | 9.25 |  |
| `XLM-R-large AUG+LEX (EN+FR)` | français | 78.60 | 48.93 | 9.06 |  |

Barre 1a : meilleure tête multilingue prête à l'emploi qui lit le français. Barre 1b : meilleur translate-test.

## AUC par corpus

| Modèle | Entrée | DACCORD | SICK-fr | Breaking-fr | RTE3-FR | GQNLI-fr | FraCaS-fr | XNLI-fr |
|---|---|---|---|---|---|---|---|---|
| `DeBERTa-v3-large NLI` | anglais d'origine | - | 99.77 | 99.51 | - | 84.07 | 97.11 | 99.57 |
| `DeBERTa-v3-large NLI` | traduit | 96.71 | 99.15 | 97.08 | 95.73 | 83.10 | 95.92 | 97.19 |
| `XLM-R-large AUG+LEX (EN)` | anglais d'origine | - | 99.48 | 98.94 | - | 46.28 | 86.58 | 96.62 |
| `XLM-R-large AUG+LEX (EN)` | français | 82.33 | 98.97 | 98.28 | 83.85 | 45.39 | 87.61 | 93.70 |
| `XLM-R-large AUG+LEX (EN+FR)` | anglais d'origine | - | 99.45 | 97.69 | - | 50.49 | 87.26 | 95.56 |
| `XLM-R-large AUG+LEX (EN+FR)` | français | 78.60 | 98.55 | 91.02 | 83.01 | 46.97 | 83.36 | 91.28 |
| `XLM-R-large AUG+LEX (FR traduit)` | anglais d'origine | - | 99.32 | 98.60 | - | 44.77 | 88.12 | 97.51 |
| `XLM-R-large AUG+LEX (FR traduit)` | français | 81.23 | 98.60 | 96.76 | 84.26 | 44.96 | 87.73 | 94.29 |
| `XLM-R-large NLI` | anglais d'origine | - | 99.69 | 99.41 | - | 45.12 | 88.21 | 100.00 |
| `XLM-R-large NLI` | français | 90.95 | 99.24 | 98.17 | 84.07 | 42.59 | 90.80 | 99.96 |
| `mDeBERTa-v3-base NLI` | anglais d'origine | - | 99.30 | 91.13 | - | 50.76 | 89.56 | 98.31 |
| `mDeBERTa-v3-base NLI` | français | 90.84 | 98.74 | 88.92 | 86.60 | 54.22 | 91.26 | 96.59 |
| `v3 AUG+LEX` | anglais d'origine | - | 99.63 | 99.15 | - | 73.75 | 94.38 | 93.96 |
| `v3 AUG+LEX` | français | 86.08 | 97.88 | 97.26 | 89.33 | 71.95 | 92.98 | 80.19 |
| `v3 AUG+LEX` | traduit | 85.97 | 98.29 | 95.31 | 91.12 | 75.68 | 92.68 | 82.64 |

## Breaking-fr, part sous zéro par famille lexicale

| Modèle | Entrée | antonymes | co-hyponymes | nombres | synonymes |
|---|---|---|---|---|---|
| `DeBERTa-v3-large NLI` | anglais d'origine | 85.24 | 77.79 | 92.79 | 0.00 |
| `DeBERTa-v3-large NLI` | traduit | 77.36 | 71.48 | 91.62 | 2.68 |
| `XLM-R-large AUG+LEX (EN)` | anglais d'origine | 86.93 | 15.07 | 97.31 | 0.00 |
| `XLM-R-large AUG+LEX (EN)` | français | 81.19 | 16.86 | 95.56 | 0.22 |
| `XLM-R-large AUG+LEX (EN+FR)` | anglais d'origine | 84.69 | 5.48 | 88.49 | 0.00 |
| `XLM-R-large AUG+LEX (EN+FR)` | français | 81.19 | 5.50 | 90.75 | 0.34 |
| `XLM-R-large AUG+LEX (FR traduit)` | anglais d'origine | 89.07 | 12.04 | 95.41 | 0.00 |
| `XLM-R-large AUG+LEX (FR traduit)` | français | 83.49 | 8.68 | 93.95 | 0.34 |
| `XLM-R-large NLI` | anglais d'origine | 91.25 | 90.84 | 99.34 | 0.45 |
| `XLM-R-large NLI` | français | 79.93 | 72.49 | 95.63 | 0.22 |
| `mDeBERTa-v3-base NLI` | anglais d'origine | 64.52 | 52.52 | 84.27 | 0.00 |
| `mDeBERTa-v3-base NLI` | français | 60.20 | 37.86 | 81.72 | 0.34 |
| `v3 AUG+LEX` | anglais d'origine | 86.00 | 15.95 | 87.62 | 0.00 |
| `v3 AUG+LEX` | français | 85.95 | 22.24 | 95.56 | 0.67 |
| `v3 AUG+LEX` | traduit | 80.04 | 15.60 | 90.24 | 2.01 |

Synonymes sous zéro = fausses alarmes. Co-hyponymes : Breaking-NLI les étiquette contradiction, la v3 est conçue pour les garder au-dessus de zéro.
