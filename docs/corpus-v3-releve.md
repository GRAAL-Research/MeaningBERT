# Releve de disponibilite, corpus de polarite (v3)

Genere par `src/diagnostics/releve_corpus_v3.py`. Ne pas editer a la main : un releve
ecrit a la main pourrit en silence, ce qui est la pire facon d'avoir tort.

| corpus | etat | lignes | etiquette | licence | role |
|---|---|---|---|---|---|
| `tals/vitaminc` | ouvrable | 488,904 | Value('string') | non declaree | 450k contrastive claim-evidence pairs; the largest minimal-pair corpus there is |
| `nikitam/ACES` | ouvrable | 36,476 | -- | non declaree | 36k examples over 68 phenomena: negation, antonyms, numbers, entities, argument order |
| `google-research-datasets/paws` | ouvrable | 65,401 | ['0', '1'] | non declaree | 108k pairs with high lexical overlap that are NOT paraphrases |
| `yangwang825/sick` | ouvrable | 9,840 | Value('int64') | non declaree | SICK, entailment half: the polarity label |
| `mteb/sickr-sts` | ouvrable | 9,927 | -- | non declaree | SICK, relatedness half: the continuous score, to join on the pair |
| `tasksource/monli` | ouvrable | 1,202 | Value('string') | non declaree | downward monotonicity under negation |
| `joey234/nan-nli` | ouvrable | 258 | Value('string') | non declaree | sub-clausal negation, targeted and small |
| `lasha-nlp/CONDAQA` | ouvrable | 14,182 | Value('string') | non declaree | negation scope in reading comprehension |
| `sentence-transformers/stsb` | ouvrable | 8,628 | -- | non declaree | continuous relatedness, to calibrate the magnitude head against |

## Echecs

