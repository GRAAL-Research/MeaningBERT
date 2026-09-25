# Environnement d'entrainement v2

Cible : `renard`, GPU 1, Quadro P5000, Pascal, capacite de calcul **6.1**.

Tout ce qui suit decoule de ce 6.1. Ce n'est pas une preference, c'est une contrainte
materielle, et elle elimine trois optimisations que l'on prendrait par defaut ailleurs.

## Les trois choses que Pascal interdit

| Optimisation | Exigence | Verdict sur sm_61 |
|---|---|---|
| `bf16` | capacite >= 8.0 (Ampere) | **impossible** |
| `torch.compile` avec inductor | Triton exige une capacite >= 7.0 | **impossible** |
| FlashAttention | capacite >= 8.0 | **impossible** |

Le `fp16` merite une mention a part. Il est disponible, mais GP102 et GP104 ont un debit
fp16 de l'ordre du 1/64e du fp32 : ce n'est pas une acceleration, c'est un ralentissement
qui economise de la memoire. Comme les 16 Go de la P5000 suffisent a `deberta-v3-large`,
**le fp32 est le bon defaut**, et le fp16 n'est utile que si un OOM se presente.

## Le choix de version de torch, et pourquoi il est etroit

PyTorch retire progressivement Pascal :

- `cu128` a retire `sm_50` et `sm_60` des la 2.8, pour raison de taille de binaire.
- **`cu126` conserve Maxwell et Pascal**, et c'est le canal a utiliser.
- **PyTorch 2.15 retirera Pascal entierement**, CUDA 13.x ne le supportant plus.

La derniere version cp311 publiee sur le canal `cu126` est **2.14.0**. C'est donc
exactement le point optimal : la version la plus recente qui contient encore des noyaux
`sm_61`. Au-dela, plus rien ne tournera sur cette carte.

Le pilote 535.309.01 supporte nativement CUDA 12.2. Le runtime CUDA 12.6 embarque dans les
roues fonctionne dessus par compatibilite de version mineure a l'interieur de CUDA 12.x,
qui exige un pilote >= 525.60.13.

**Ce choix n'est pas suppose, il est verifie.** `src/diagnostics/verify_training_env.py`
echoue si `sm_61` est absent de `torch.cuda.get_arch_list()`.

## Installation

```
bash env/setup_renard.sh
```

Le script est idempotent. Il cree `~/.venvs/meaningbert-v2`, installe les versions
epinglees de `env/renard-requirements.txt`, puis lance la verification.

## Verification

```
ssh renard '~/.venvs/meaningbert-v2/bin/python -m diagnostics.verify_training_env --gpu 1'
```

Elle refuse de passer si l'une de ces conditions manque :

1. `sm_61` present dans les architectures compilees ;
2. un vrai aller-retour avant/arriere sur la GPU visee, pas seulement `cuda.is_available()` ;
3. la precision recommandee coherente avec la capacite detectee ;
4. un debit mesure, pour comparer apres chaque mise a jour du serveur.

## Reglages de performance retenus

| Reglage | Valeur | Raison |
|---|---|---|
| Precision | `fp32` | pas de bf16, fp16 contre-productif en calcul sur Pascal |
| `torch.compile` | desactive | Triton exige >= 7.0 |
| Backend d'attention | SDPA `mem_efficient` et `math` | flash exige >= 8.0 |
| `cudnn.benchmark` | `True` | tailles d'entree stables apres padding par lot |
| `CUDA_VISIBLE_DEVICES` | `1` | arbitrage David, P5000 uniquement |
| Workers du DataLoader | 6 | 12 coeurs, la moitie laissee au reste |
| `pin_memory` | `True` | transfert hote vers peripherique asynchrone |
| Accumulation de gradient | plutot qu'un gros lot | 16 Go, et le lot effectif reste comparable au sweep |
| Optimiseur 8 bits | **non par defaut** | economiserait 6 octets par parametre, mais ajoute une dependance fragile pour un probleme de memoire qu'on n'a pas |

## Ce qui change quand renard rebouge

Ne pas reecrire ce fichier de tete. Relancer la sonde :

```
PYTHONPATH=src python src/diagnostics/probe_training_host.py --host renard --gpu 1 \
    --json-out results/host-renard.json --markdown-out docs/serveur-releve-renard.md
```

Si la capacite de calcul passe au-dessus de 7.0, `torch.compile` redevient possible et ce
tableau est a refaire. Si elle passe au-dessus de 8.0, le `bf16` aussi, et le canal `cu126`
n'est plus une contrainte.
