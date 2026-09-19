# ModernBERT sur caribou

## Pourquoi pas sur renard

renard est une Quadro P5000, Pascal, capacite de calcul **6.1**. Trois choses sont
impossibles par materiel, pas par configuration :

| | Exige |
|---|---|
| bf16 | 8.0 |
| `torch.compile` via Triton | 7.0 |
| FlashAttention | 8.0 |

ModernBERT est construit autour de FlashAttention et de l'attention sans padding. Il
tourne sans, mais si lentement et avec un profil memoire si different qu'un resultat
Pascal ne dirait rien d'utile sur l'architecture. caribou (RTX 6000 Ada, capacite 8.9,
49 Go) a les trois.

## Preparation

```
# 1. Relever le materiel, comme pour renard
PYTHONPATH=src python src/diagnostics/probe_training_host.py --host caribou --gpu 0 \
    --json-out results/host-caribou.json --markdown-out docs/serveur-releve-caribou.md

# 2. Synchroniser code et corpus. Les variantes sont deja construites sur renard ;
#    les recopier evite de refaire la back-translation.
rsync -az --exclude '.venv' --exclude '__pycache__' ./ caribou:~/MeaningBERT/
rsync -az renard:~/MeaningBERT/data/v2/ caribou:~/MeaningBERT/data/v2/

# 3. Environnement. ATTENTION : ne pas reutiliser env/renard-lock.txt, qui epingle
#    torch 2.14.0+cu126 parce que c'est la derniere version portant des noyaux Pascal.
#    caribou est en 8.9 et n'a pas cette contrainte ; un canal plus recent y est
#    preferable. Repartir de env/renard-constraints.txt sans l'epingle torch.
ssh caribou 'python3 -m venv ~/.venvs/meaningbert-v2'
ssh caribou '~/.venvs/meaningbert-v2/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128'
ssh caribou '~/.venvs/meaningbert-v2/bin/pip install -r ~/MeaningBERT/env/renard-constraints.txt'

# 4. Verifier
ssh caribou 'cd ~/MeaningBERT && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
    ~/.venvs/meaningbert-v2/bin/python src/diagnostics/verify_training_env.py --gpu 0'
```

`transformers >= 4.48` est requis : ModernBERT n'existe pas avant. Le script le verifie et
s'arrete avec un message clair plutot que de partir sur une erreur de configuration
obscure.

## Lancement

```
ssh caribou 'cd ~/MeaningBERT && setsid nohup bash src/training/run_grid_caribou.sh \
    > results/grid-caribou.log 2>&1 < /dev/null &'
```

Le script refuse de demarrer si la capacite de calcul est sous 8.0, pour qu'on ne le lance
pas sur renard par inadvertance et qu'on ne compare pas un ModernBERT bride a un DeBERTa
a pleine vitesse.

## Ce qui est tenu identique a renard

Memes variantes de corpus, meme lot effectif de 32, meme tete de sortie sigmoide, meme
graine, meme budget d'epoques. **Seule la precision differe**, bf16 contre fp32, et c'est
precisement ce qu'on veut mesurer : l'apport d'une architecture recente sur un materiel
qui lui convient.

Cette difference de precision est une limite de la comparaison, pas un detail. Elle
s'ajoute a la difference d'architecture, et il faudra le dire dans le tableau final.

## Rapatriement

```
rsync -az caribou:~/MeaningBERT/results/grid/ ./results/grid/
PYTHONPATH=src python src/figures_generator/analyze_v2_experiment.py --runs-dir results/grid
```
