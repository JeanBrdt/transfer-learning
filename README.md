# transfer-forest

Pipeline d'expériences autour du transfert pour les forêts aléatoires. Le package `transfer_forest` contient les scénarios de données, les méthodes comparées et les contrats communs (contexte, construction des forêts).

## Installation

```bash
git clone https://gitlab.imt-atlantique.fr/c25trand/transfer-forest.git
cd transfer-forest
pip install -e .
```

L'option `-e` installe le package en mode éditable : les modifications du code sont prises en compte sans réinstaller.

Python 3.10 ou plus est requis. Pour faire tourner les notebooks (graphiques, Jupyter) ou les tests, ajouter les options correspondantes :

```bash
pip install -e ".[notebooks]"   # matplotlib et jupyter
pip install -e ".[dev]"         # pytest
```

### Dans un notebook (Colab, Jupyter)

Le package n'est pas sur PyPI, il faut donc l'installer dans l'environnement du notebook avant le moindre import. Le dépôt est **privé** : `git clone` a besoin d'un jeton d'accès en lecture, qu'on ne peut pas saisir dans Colab.

**Une fois par personne : créer un jeton.** Sur GitLab : avatar > Preferences > Access tokens, créer un jeton avec la seule case `read_repository`, et le copier tout de suite (il ne s'affiche qu'une fois). Ou utiliser le jeton d'équipe (« Deploy token », lecture seule) communiqué par le propriétaire du dépôt.

**Première cellule du notebook :**

```python
from getpass import getpass
token = getpass("Jeton GitLab : ")   # le jeton se colle dans la case qui s'ouvre
!git clone https://oauth2:{token}@gitlab.imt-atlantique.fr/c25trand/transfer-forest.git
%cd transfer-forest
!pip install -e ".[notebooks]"
```

Ne jamais écrire le jeton dans une cellule, dans un fichier ni dans un commit, et ne pas partager un notebook qui le contient.

Sur Colab, redémarrer la session après l'installation (Exécution > Redémarrer la session), puis relancer les cellules.

**Sans GitLab (si le serveur n'est pas joignable hors du réseau de l'école).** Compresser le dossier `transfer-forest` en `transfer-forest.zip` sur son ordinateur, l'envoyer dans Colab (icône de dossier à gauche, puis bouton d'envoi), puis :

```python
!unzip -q transfer-forest.zip
%cd transfer-forest
!pip install -e ".[notebooks]"
```

## Utilisation

```python
from transfer_forest.contracts import Context, make_forest
from transfer_forest.data import load_scenarios
from transfer_forest.methods import METHODS, make_method
```

Les méthodes disponibles sont dans `METHODS` : les références sans transfert (`Src`, `Tgt`, `Union`, `Union-P`, `Oracle`) et les méthodes de transfert (`SER`, `STRUT`, `SER*`, `STRUT*`).

Les notebooks d'expériences sont dans `notebooks/` :
- `experience_references_s1.ipynb` : les références sans transfert sur S1 ;
- `experience_transfert_s1.ipynb` : toutes les méthodes de `METHODS` face aux références sur S1. Les résultats sont enregistrés dans `results/s1_transfert.csv` : en relançant, seules les combinaisons manquantes sont calculées.

## Structure du dépôt

```
transfer-forest/
├── data/             # données des scénarios
├── notebooks/        # notebooks d'expériences
├── results/          # résultats enregistrés par les notebooks (ignoré par git, créé au besoin)
├── tests/            # tests (pytest)
├── transfer_forest/  # le package
└── pyproject.toml
```

## Ajouter une méthode

Écrire la classe dans `transfer_forest/methods/`, puis l'ajouter au dictionnaire `METHODS` dans `transfer_forest/methods/__init__.py`. Elle devient alors disponible via `make_method(name)`.

## Tests

```bash
pip install -e ".[dev]"
python -m pytest tests -q
```

## Problèmes fréquents

**`fatal: could not read Username for 'https://gitlab...'`** (dans Colab)
Le dépôt est privé et le clone s'est fait sans jeton. Suivre la procédure du jeton d'accès dans « Dans un notebook (Colab, Jupyter) ». Les erreurs `No such file or directory: 'transfer-forest'` et `does not appear to be a Python project` qui suivent viennent de ce clone raté.

**`ModuleNotFoundError: No module named 'transfer_forest'`**
Le package n'est pas installé dans l'environnement utilisé. Vérifier avec `pip show transfer_forest`, puis relancer `pip install -e .` depuis la racine du dépôt. Dans un notebook, redémarrer le noyau après l'installation.
