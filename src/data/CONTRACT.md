# Contrat de donnees - CSMD v2

Tout loader de corpus produit **exactement** ce schema. Aucune colonne en plus, aucune
en moins. Un loader qui devie casse l'etape d'harmonisation en aval.

## Regle la plus importante

**Un loader ne calcule jamais `label`.** Il emet `label_raw` dans l'echelle native du
corpus source, plus `scale` qui identifie cette echelle. Le mapping vers l'echelle
0-100 commune est fait par un seul proprietaire, `src/data/harmonize.py`, apres que
tous les loaders sont livres.

Raison : quatre loaders ecrits en parallele produiraient quatre mappings implicites
incompatibles, et personne ne pourrait plus mesurer H2 (voir PRODUIT.md).

## Schema

| Colonne | Type | Obligatoire | Description |
|---|---|---|---|
| `item_id` | str | oui | Identifiant stable, format `<corpus>:<id natif>`. Unique dans le corpus. |
| `original` | str | oui | Phrase source, complexe. Non vide, strippee. |
| `simplification` | str | oui | Phrase candidate, simplifiee. Non vide, strippee. |
| `label_raw` | float | oui | Score de preservation du sens dans l'echelle NATIVE du corpus. |
| `label` | float | oui | Toujours `float("nan")` a la sortie du loader. Rempli par `harmonize.py`. |
| `scale` | str | oui | Identifiant d'echelle. Valeurs permises ci-dessous. |
| `n_annotators` | int | oui | Nombre d'annotateurs humains derriere `label_raw`. 0 si derive ou synthetique. |
| `label_std` | float | oui | Ecart-type inter-annotateurs. `float("nan")` si inconnu ou si `n_annotators < 2`. |
| `corpus` | str | oui | Identifiant du corpus source. Constante par loader. |
| `source` | str | oui | `original`, `identical`, `unrelated` ou `back_translated`. |
| `domain` | str | oui | `wiki`, `news`, `biomedical`, `scientific`, `mixed`. |
| `system` | str | oui | Systeme ayant produit la simplification. `human` si humaine, `""` si inconnu. |
| `split_hint` | str | oui | `train`, `dev`, `test` ou `""` si le corpus source n'impose pas de split. |
| `license` | str | oui | Licence du corpus source, identifiant SPDX quand il existe. |

## Valeurs permises pour `scale`

| Valeur | Signification | Borne basse | Borne haute | Sens |
|---|---|---|---|---|
| `da100` | Direct assessment continu 0-100 | 0 | 100 | plus haut = mieux preserve |
| `likert5` | Likert 1-5 | 1 | 5 | plus haut = mieux preserve |
| `likert7` | Likert 1-7 | 1 | 7 | plus haut = mieux preserve |
| `severity3` | Severite d'erreur 1-3 | 1 | 3 | **plus haut = PIRE**, a inverser |
| `likert3_signed` | Likert signe a 3 points | -1 | 1 | plus haut = mieux preserve |
| `binary` | Etiquette binaire | 0 | 1 | 1 = sens preserve |
| `error_count` | Nombre d'erreurs de distorsion | 0 | inf | **plus haut = PIRE**, a inverser |

Un loader qui a besoin d'une echelle absente de cette table ne l'invente pas : il ouvre
la question avant d'ecrire le mapping.

`likert3_signed` a ete ajoutee le 2026-09-19 exactement par ce chemin. Le loader PLABA a
rencontre l'echelle `{-1, 0, 1}` des jugements experts TREC, a constate qu'aucune entree ne
convenait, et s'est arrete en levant `UnsupportedScaleError` plutot que de reutiliser
`severity3`, qui a la meme cardinalite mais des bornes differentes et l'orientation
inverse. Reutiliser `severity3` aurait inverse le signal en silence.

## Interface Python

Chaque loader expose une fonction unique :

```python
def load() -> datasets.Dataset:
    """Retourne le corpus au schema du CONTRACT, sans aucune harmonisation."""
```

Le module vit dans `src/data/loaders/<corpus>.py`. Le nom du module est la valeur de la
colonne `corpus`.

## Validation

`src/data/schema.py` expose `validate(dataset)`. Un loader n'est pas livre tant que
`validate()` ne passe pas. La validation verifie :

- presence et type exact de chaque colonne ;
- `label` entierement NaN ;
- `scale` dans la table des valeurs permises ;
- `label_raw` dans les bornes de son `scale` ;
- `item_id` unique et prefixe par `corpus` ;
- `original` et `simplification` non vides apres strip ;
- `corpus`, `source`, `domain`, `split_hint` dans leurs vocabulaires.

## Rapport de livraison

Chaque loader livre aussi `src/data/loaders/<corpus>.report.json` :

```json
{
  "corpus": "...",
  "n_rows": 0,
  "n_unique_originals": 0,
  "scale": "...",
  "label_raw_min": 0.0,
  "label_raw_max": 0.0,
  "label_raw_mean": 0.0,
  "n_annotators_median": 0,
  "pct_with_std": 0.0,
  "domains": {"...": 0},
  "systems": {"...": 0},
  "split_hints": {"...": 0},
  "license": "...",
  "license_allows_redistribution": true,
  "source_url": "...",
  "retrieval_date": "YYYY-MM-DD",
  "notes": "decisions de mapping, lignes ecartees et pourquoi"
}
```

`license_allows_redistribution` est la reponse a H4 du PRODUIT.md. Si elle est `false`,
CSMD v2 ne peut distribuer que le loader, pas les lignes.

## Deduplication

Les loaders ne deduppent pas entre corpus. C'est le travail de `harmonize.py`, qui a la
vue d'ensemble. Un loader dedupe uniquement a l'interieur de son propre corpus, sur
`(original, simplification, system)`, et le note dans `notes`.

## Fuite de donnees

CSMD v1 contient deja des paires issues d'ASSET. SimpEval et SALSA en contiennent aussi.
Les loaders signalent leur origine amont dans `notes`, mais ne filtrent pas. Le controle
anti-fuite est centralise dans `harmonize.py` puis `validate_datasets.py`.
