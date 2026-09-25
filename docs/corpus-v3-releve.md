# Releve de disponibilite, corpus de polarite (v3)

Genere par `src/diagnostics/releve_corpus_v3.py`. Ne pas editer a la main : un releve
ecrit a la main pourrit en silence, ce qui est la pire facon d'avoir tort.

| corpus | etat | lignes | etiquette | licence | role |
|---|---|---|---|---|---|
| `tals/vitaminc` | ouvrable | 488,904 | Value('string') | non declaree | 450k contrastive claim-evidence pairs; the largest minimal-pair corpus there is |
| `nightingal3/fig-qa` | ouvrable | 11,914 | Value('int64') | non declaree | control: figurative pairs, to see whether the harness reads an unrelated schema |
| `EdinburghNLP/ACES` | **indisponible** | -- | -- | -- | 36k examples over 68 phenomena: negation, antonyms, numbers, entities, argument order |
| `google-research-datasets/paws` | ouvrable | 65,401 | ['0', '1'] | non declaree | 108k pairs with high lexical overlap that are NOT paraphrases |
| `sentence-transformers/stsb` | ouvrable | 8,628 | -- | non declaree | continuous relatedness, the bridge between a magnitude head and a polarity head |
| `sick` | **indisponible** | -- | -- | -- | relatedness AND entailment on the same pairs; the natural bridge between the two heads |
| `pietrolesci/nan-nli` | **indisponible** | -- | -- | -- | sub-clausal negation, targeted and small |
| `sagnikrayc/monli` | **indisponible** | -- | -- | -- | downward monotonicity under negation |
| `lasha-nlp/CondaQA` | ouvrable | 0 | -- | non declaree | negation scope in reading comprehension |

## Echecs

- `EdinburghNLP/ACES` : DatasetNotFoundError: Dataset 'EdinburghNLP/ACES' doesn't exist on the Hub or cannot be accessed.
- `sick` : RuntimeError: Dataset scripts are no longer supported, but found sick.py
- `pietrolesci/nan-nli` : DatasetNotFoundError: Dataset 'pietrolesci/nan-nli' doesn't exist on the Hub or cannot be accessed.
- `sagnikrayc/monli` : DatasetNotFoundError: Dataset 'sagnikrayc/monli' doesn't exist on the Hub or cannot be accessed.
