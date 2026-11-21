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
pip install -e ".[notebooks]"   # jupyter
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

## Lancer l'expérience

Depuis la racine du dépôt :

```bash
python -m transfer_forest.run S1        # ou S5
```

Le moteur (`transfer_forest/run.py`) fait tout, de bout en bout : découpage test / réserve,
mesure du décalage, puis, pour chacune des 20 graines (en parallèle), la forêt source et
chaque méthode à chaque budget, notées sur le test ; enfin l'analyse et les figures.
Sorties dans `results/` :

| Fichier | Contenu |
|---|---|
| `S1.csv` | une ligne par (graine, budget, méthode) : notes, temps, version du code (940 lignes) |
| `S1_summary.csv` | par (budget, méthode) : gains G_Tgt et G_ref, IC à 95 %, taux de victoire, Wilcoxon (p brute et Holm), économie d'annotation, budget de croisement |
| `S1_shift.csv` | le décalage : AUC du classifieur de domaine (avec et sans la variable de coupe), écart des proportions, écart de domaine |
| `S1_scores.png`, `S1_gains.png` | note selon le budget ; gains avec leurs IC |

G_ref compare à la meilleure des références Src, Tgt et Union, comme le prévoit le
protocole. Union-P (source + cible surpondérée) est jugée comme une méthode de transfert,
ce qu'elle est : elle a ses propres gains G_Tgt et G_ref.

## Le pipeline

| Étape | Module | Ce qu'il fournit |
|---|---|---|
| contrat | `contracts.py` | formats communs : `Scenario`, `Context`, `Method`, `make_forest` |
| scénarios | `data/` | `load_scenarios("S1")` |
| décalage | `shift.py` | `characterize(scn, reserve)` |
| tirages | `splits.py` | `split_target`, `draws` (voir plus bas) |
| méthodes | `methods/` | `METHODS`, `make_method(name)` |
| notation | `evaluation.py` | `evaluate(y_test, P)` : error, ber, f1_macro, fnr_critical, log_loss |
| analyse | `scoring.py`, `plots.py` | gains, Wilcoxon-Holm, synthèse, figures |
| moteur | `run.py` | l'expérience complète |

Les méthodes disponibles sont dans `METHODS` : les références sans transfert (`Src`, `Tgt`,
`Union`, `Oracle`), le transfert par repondération `Union-P` et les méthodes de Segev et al.
(`Relab`, `SER`, `STRUT`, `MIX`, `SER*`, `STRUT*`).

Les notebooks d'expériences sont dans `notebooks/` :
- `experience_s5.ipynb` : le scénario S5 (C-MAPSS) et l'interprétation de ses résultats ;
- `experience_references_s1.ipynb` : les références sans transfert sur S1 ;
- `experience_transfert_s1.ipynb` : toutes les méthodes de `METHODS` face aux références sur S1. Les résultats sont enregistrés dans `results/s1_transfert.csv` : en relançant, seules les combinaisons manquantes sont calculées.

## Structure du dépôt

```
transfer-forest/
├── data/             # données des scénarios
├── notebooks/        # notebooks d'expériences
├── results/          # résultats du moteur et des notebooks (ignoré par git, créé au besoin)
├── tests/            # tests (pytest)
├── transfer_forest/  # le package
└── pyproject.toml
```

## Scénario S5 : C-MAPSS (`transfer_forest/data/s5_cmapss.py`)

Turboréacteurs simulés par la NASA, suivis cycle après cycle jusqu'à la panne. Un exemple est un
cycle d'un moteur : 3 réglages de vol et 21 capteurs. Tâche : prédire une **panne dans les
30 cycles** (classe `panne`, 15 % des cycles), notée en BER ; `fnr_critical` est la part des
pannes ratées.

- **Source** : FD001, 100 moteurs, un seul régime de vol (20 631 cycles).
- **Cible** : FD002, 260 moteurs, six régimes de vol (53 759 cycles). Même panne (usure du
  compresseur haute pression) ; ce sont les conditions de vol qui changent les mesures. Un seul
  des six régimes, 15 % des cycles, est celui de la source.
- **`groups_t`** : le moteur de chaque cycle cible. Le découpage met chaque moteur entier dans
  le test ou dans la réserve (130 moteurs chacun).
- **Données** : `data/cmapss/train_FD001.txt.gz` et `train_FD002.txt.gz`, tirés de l'archive
  CMAPSSData du NASA Prognostics Data Repository (« Turbofan Engine Degradation Simulation »).

## Découpage des données (`transfer_forest/splits.py`)

Implémente la partie 2 du protocole :

```
cible ──┬── test    (scn.test_frac = 50 %, stratifié, toujours le même)
        └── réserve
               └── pour chaque graine : un ordre de la réserve
                     ├── les n_T premiers : exemples cibles annotés du budget
                     └── le reste : non annoté
```

- **Test** : tiré une fois pour toutes (graine fixe `TEST_SEED`), stratifié sur les
  classes ; il ne sert qu'à l'évaluation. Si le scénario déclare des groupes (`groups_t` :
  un moteur, un utilisateur…), chaque groupe va entier dans le test ou dans la réserve.
- **Budgets** : nombre d'exemples **cibles** annotés n_T = round(p · n_S), pour
  p ∈ `BUDGETS` = {1, 2, 5, 10, 20} % (fraction de la taille de la source, comme
  Segev et al.), et au moins K pour avoir chaque classe. Sur S1 : 108, 216, 541,
  1082, 2164.
- **Tirages emboîtés** : pour chaque graine de `SEEDS` (20 graines), la réserve est
  mise dans un ordre aléatoire dont les K premiers éléments couvrent les K classes ;
  chaque budget prend les n_T premiers, donc D_T(1 %) ⊂ D_T(2 %) ⊂ …

Trois fonctions, qui renvoient des indices dans `scn.Xt` / `scn.yt` :

```python
from transfer_forest.splits import BUDGETS, SEEDS, budgets, reserve_order, split_target

reserve, test = split_target(scn)              # une fois par scénario
n_T = budgets(scn, len(reserve))               # {p: n_T}
for seed in SEEDS:
    order = reserve_order(scn.yt[reserve], seed)
    for p in BUDGETS:
        lab = reserve[order[:n_T[p]]]          # exemples cibles annotés
        unl = reserve[order[n_T[p]:]]          # reste de la réserve, sans étiquettes
```

Ce découpage est identique à celui, provisoire, du notebook
`experience_references_s1.ipynb` : mêmes indices de test, mêmes ordres, mêmes budgets.

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
