## DACCORD, le seul corpus écrit en français

| Modèle | Entrée | AUC | Contradictions < 0 | Compatibles < 0 | Barre |
|---|---|---|---|---|---|
| `DeBERTa-v3-large NLI` | traduit | 96.71 | 82.33 | 4.05 | 1b |
| `XLM-R-large NLI` | français | 90.95 | 69.51 | 6.74 | 1a |
| `mDeBERTa-v3-base NLI` | français | 90.84 | 69.32 | 7.71 |  |
| `v3 AUG+LEX` | français | 86.08 | 52.62 | 0.77 |  |
| `v3 AUG+LEX` | traduit | 85.97 | 56.31 | 2.31 |  |
| `XLM-R-large AUG+LEX (EN)` | français | 83.05 | 51.84 | 6.74 |  |
| `XLM-R-large AUG+LEX (FR traduit)` | français | 80.85 | 53.20 | 10.21 |  |
| `mDeBERTa-v3-base AUG+LEX (EN)` | français | 79.22 | 47.38 | 5.78 |  |
| `XLM-R-large AUG+LEX (EN+FR)` | français | 78.90 | 49.51 | 8.86 |  |
| `mDeBERTa-v3-base AUG+LEX (FR traduit)` | français | 76.91 | 52.62 | 10.79 |  |
| `mDeBERTa-v3-base AUG+LEX (EN+FR)` | français | 76.70 | 51.07 | 11.56 |  |

Barre 1a : meilleure tête multilingue prête à l'emploi qui lit le français. Barre 1b : meilleur translate-test.

## AUC par corpus

| Modèle | Entrée | DACCORD | SICK-fr | Breaking-fr | RTE3-FR | GQNLI-fr | FraCaS-fr | XNLI-fr |
|---|---|---|---|---|---|---|---|---|
| `DeBERTa-v3-large NLI` | anglais d'origine | - | 99.77 | 99.51 | - | 84.07 | 97.11 | 99.57 |
| `DeBERTa-v3-large NLI` | traduit | 96.71 | 99.15 | 97.08 | 95.73 | 83.10 | 95.92 | 97.19 |
| `XLM-R-large AUG+LEX (EN)` | anglais d'origine | - | 99.62 | 99.10 | - | 46.43 | 87.36 | 97.21 |
| `XLM-R-large AUG+LEX (EN)` | français | 83.05 | 99.07 | 98.33 | 84.55 | 45.78 | 88.55 | 94.47 |
| `XLM-R-large AUG+LEX (EN+FR)` | anglais d'origine | - | 99.50 | 97.84 | - | 50.53 | 87.26 | 96.12 |
| `XLM-R-large AUG+LEX (EN+FR)` | français | 78.90 | 98.58 | 91.33 | 83.07 | 46.16 | 85.24 | 92.19 |
| `XLM-R-large AUG+LEX (FR traduit)` | anglais d'origine | - | 99.28 | 98.64 | - | 46.66 | 87.50 | 97.65 |
| `XLM-R-large AUG+LEX (FR traduit)` | français | 80.85 | 98.57 | 96.52 | 83.89 | 46.07 | 86.72 | 94.69 |
| `XLM-R-large NLI` | anglais d'origine | - | 99.69 | 99.41 | - | 45.12 | 88.21 | 100.00 |
| `XLM-R-large NLI` | français | 90.95 | 99.24 | 98.17 | 84.07 | 42.59 | 90.80 | 99.96 |
| `mDeBERTa-v3-base AUG+LEX (EN)` | anglais d'origine | - | 99.60 | 98.93 | - | 54.64 | 86.97 | 92.49 |
| `mDeBERTa-v3-base AUG+LEX (EN)` | français | 79.22 | 98.95 | 97.43 | 82.01 | 56.09 | 85.71 | 86.54 |
| `mDeBERTa-v3-base AUG+LEX (EN+FR)` | anglais d'origine | - | 99.52 | 98.72 | - | 54.62 | 86.03 | 90.50 |
| `mDeBERTa-v3-base AUG+LEX (EN+FR)` | français | 76.70 | 98.80 | 95.31 | 78.96 | 55.84 | 82.47 | 85.42 |
| `mDeBERTa-v3-base AUG+LEX (FR traduit)` | anglais d'origine | - | 99.02 | 96.67 | - | 54.94 | 85.71 | 92.71 |
| `mDeBERTa-v3-base AUG+LEX (FR traduit)` | français | 76.91 | 98.32 | 91.20 | 80.51 | 57.10 | 83.71 | 87.56 |
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
| `XLM-R-large AUG+LEX (EN)` | anglais d'origine | 87.10 | 14.34 | 96.80 | 0.00 |
| `XLM-R-large AUG+LEX (EN)` | français | 80.37 | 16.20 | 95.70 | 0.34 |
| `XLM-R-large AUG+LEX (EN+FR)` | anglais d'origine | 85.29 | 5.70 | 89.00 | 0.00 |
| `XLM-R-large AUG+LEX (EN+FR)` | français | 81.90 | 5.83 | 91.48 | 0.34 |
| `XLM-R-large AUG+LEX (FR traduit)` | anglais d'origine | 88.79 | 12.06 | 95.70 | 0.00 |
| `XLM-R-large AUG+LEX (FR traduit)` | français | 83.16 | 9.19 | 94.39 | 0.34 |
| `XLM-R-large NLI` | anglais d'origine | 91.25 | 90.84 | 99.34 | 0.45 |
| `XLM-R-large NLI` | français | 79.93 | 72.49 | 95.63 | 0.22 |
| `mDeBERTa-v3-base AUG+LEX (EN)` | anglais d'origine | 80.32 | 7.19 | 80.48 | 0.11 |
| `mDeBERTa-v3-base AUG+LEX (EN)` | français | 75.78 | 6.31 | 74.73 | 0.22 |
| `mDeBERTa-v3-base AUG+LEX (EN+FR)` | anglais d'origine | 81.03 | 5.22 | 80.12 | 0.00 |
| `mDeBERTa-v3-base AUG+LEX (EN+FR)` | français | 82.18 | 6.23 | 82.74 | 0.45 |
| `mDeBERTa-v3-base AUG+LEX (FR traduit)` | anglais d'origine | 80.32 | 12.54 | 79.83 | 0.00 |
| `mDeBERTa-v3-base AUG+LEX (FR traduit)` | français | 80.81 | 9.21 | 78.51 | 0.45 |
| `mDeBERTa-v3-base NLI` | anglais d'origine | 64.52 | 52.52 | 84.27 | 0.00 |
| `mDeBERTa-v3-base NLI` | français | 60.20 | 37.86 | 81.72 | 0.34 |
| `v3 AUG+LEX` | anglais d'origine | 86.00 | 15.95 | 87.62 | 0.00 |
| `v3 AUG+LEX` | français | 85.95 | 22.24 | 95.56 | 0.67 |
| `v3 AUG+LEX` | traduit | 80.04 | 15.60 | 90.24 | 2.01 |

Synonymes sous zéro = fausses alarmes. Co-hyponymes : Breaking-NLI les étiquette contradiction, la v3 est conçue pour les garder au-dessus de zéro.
