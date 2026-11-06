# transfer-forest

Pipeline d'expériences autour du transfert pour les forêts aléatoires. Le package `transfer_forest` contient les scénarios de données, les méthodes comparées et les contrats communs (contexte, construction des forêts).

## Installation

```bash
git clone https://gitlab.imt-atlantique.fr/c25trand/transfer-forest.git
cd transfer-forest
pip install -e .
```

L'option `-e` installe le package en mode éditable : les modifications du code sont prises en compte sans réinstaller.

### Dans un notebook (Colab, Jupyter)

Le package n'est pas sur PyPI, il faut donc l'installer dans l'environnement du notebook avant le moindre import. Première cellule :

```python
!git clone https://gitlab.imt-atlantique.fr/c25trand/transfer-forest.git
%cd transfer-forest
!pip install -e .
```

Sur Colab, redémarrer la session après l'installation (Exécution > Redémarrer la session), puis relancer les cellules.

## Utilisation

```python
from transfer_forest.contracts import Context, make_forest
from transfer_forest.data import load_scenarios
from transfer_forest.methods import METHODS, make_method
```

Les notebooks d'exemple sont dans `notebooks/`.

## Structure du dépôt

```
transfer-forest/
├── data/             # données des scénarios
├── notebooks/        # notebooks d'expériences
├── transfer_forest/  # le package
└── pyproject.toml
```

## Ajouter une méthode

Écrire la classe dans `transfer_forest/methods/`, puis l'ajouter au dictionnaire `METHODS` dans `transfer_forest/methods/__init__.py`. Elle devient alors disponible via `make_method(name)`.

## Problèmes fréquents

**`ModuleNotFoundError: No module named 'transfer_forest'`**
Le package n'est pas installé dans l'environnement utilisé. Vérifier avec `pip show transfer_forest`, puis relancer `pip install -e .` depuis la racine du dépôt. Dans un notebook, redémarrer le noyau après l'installation.
