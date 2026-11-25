# transfer-forest

Pipeline d'expériences sur le **transfert de modèle pour les forêts aléatoires** : on apprend une forêt sur un domaine source, puis on la corrige avec un petit nombre d'exemples annotés du domaine cible, et on compare ces méthodes à des références sans transfert sur trois jeux de données réels.

Le package `transfer_forest` contient les scénarios de données, les méthodes comparées (dont celles de Segev et al., 2017), le protocole de découpage et d'évaluation, et le moteur qui enchaîne tout, du décalage entre domaines jusqu'aux figures.

| Scénario | Données | Source → cible | Tâche |
|---|---|---|---|
| `S1` | Letter Recognition (UCI n° 59) | lettres étroites → lettres étalées (coupe sur `x2bar`) | reconnaissance de 26 lettres |
| `S5` | C-MAPSS (NASA) | FD001 → FD002 | panne d'un turboréacteur dans les 30 cycles |
| `S6` | HHAR (UCI n° 344) | montres → téléphones | reconnaissance d'activité (6 classes) |

## Installation

```bash
git clone https://github.com/JeanBrdt/transfer-learning.git
cd transfer-learning
pip install -e .
```

L'option `-e` installe le package en mode éditable : les modifications du code sont prises en compte sans réinstaller. Python 3.10 ou plus est requis. Pour les notebooks ou les tests, ajouter les options correspondantes :

```bash
pip install -e ".[notebooks]"   # jupyter
pip install -e ".[dev]"         # pytest
```

### Dans un notebook (Colab, Jupyter)

Le package n'est pas sur PyPI : il faut l'installer dans l'environnement du notebook avant le moindre import.

```python
!git clone https://github.com/JeanBrdt/transfer-learning.git
%cd transfer-learning
!pip install -e ".[notebooks]"
```

Si le dépôt est privé, `git clone` a besoin d'un jeton d'accès (GitHub : Settings > Developer settings > Personal access tokens, droit `repo` en lecture) :

```python
from getpass import getpass
token = getpass("Jeton GitHub : ")   # se colle dans la case qui s'ouvre
!git clone https://{token}@github.com/JeanBrdt/transfer-learning.git
%cd transfer-learning
!pip install -e ".[notebooks]"
```

Ne jamais écrire le jeton dans une cellule, dans un fichier ni dans un commit, et ne pas partager un notebook qui le contient. Sur Colab, redémarrer la session après l'installation (Exécution > Redémarrer la session), puis relancer les cellules.

## Lancer l'expérience

Depuis la racine du dépôt :

```bash
python -m transfer_forest.run S1        # ou S5, S6
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

Scénarios disponibles : `S1` (Letter), `S5` (C-MAPSS) et `S6` (HHAR), décrits plus bas.

Les notebooks d'expériences sont dans `notebooks/` :
- `experience_s5.ipynb` : le scénario S5 (C-MAPSS) et l'interprétation de ses résultats ;
- `experience_s6.ipynb` : le scénario S6 (HHAR) : décalage, résultats, coût et bilan ;
- `experience_references_s1.ipynb` : les références sans transfert sur S1 ;
- `experience_transfert_s1.ipynb` : toutes les méthodes de `METHODS` face aux références sur S1. Les résultats sont enregistrés dans `results/s1_transfert.csv` : en relançant, seules les combinaisons manquantes sont calculées.

## Structure du dépôt

```
transfer-learning/
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

## Scénario S6 : HHAR (`transfer_forest/data/s6_hhar.py`)

Reconnaissance d'activité (6 classes : bike, sit, stairsdown, stairsup, stand, walk) à partir de
l'accéléromètre et du gyroscope de téléphones et de montres (UCI n° 344, 9 utilisateurs). Les
appareils ne se portent pas au même endroit et n'échantillonnent pas à la même fréquence
(50 à 200 Hz) : le décalage entre domaines est réel (AUC du classifieur de domaine ≈ 1, la forêt
source se trompe à 61 % sur la cible).

```bash
python -m transfer_forest.run S6
```

- **Fenêtres** : les mesures brutes sont résumées en fenêtres de 2 s sans recouvrement, par
  appareil et par utilisateur. 32 variables = 2 capteurs × 4 signaux (x, y, z, norme) × 4 résumés
  (moyenne, écart-type, min, max). On écarte les fenêtres sans activité (`null`), celles où l'activité
  change (moins de 90 % d'une même étiquette) et celles de moins de 40 échantillons par capteur.
  Les `samsungold` n'ont pas de gyroscope et sont écartés.
- **Source et cible** : montres → téléphones (7 521 → 35 175 fenêtres). Le sens inverse est
  impossible avec le protocole : le budget maximal (20 % de la source = 7 035) dépasserait la
  réserve de la cible (3 760). `make_scenario(windows, source="phone", target="watch")` le permet
  tout de même pour des essais hors protocole.
- **`groups_t`** : l'utilisateur de chaque fenêtre de la cible. Le découpage met chaque utilisateur
  entier dans le test ou dans la réserve (5 utilisateurs et 19 984 fenêtres dans le test, 4 et
  15 191 dans la réserve) : aucune méthode n'est notée sur la personne qu'elle a vue.
- **Données** : le résumé en fenêtres est dans `data/hhar/windows.csv.gz` (5 Mo, versionné), le
  moteur n'a besoin de rien d'autre. Pour le reconstruire (autre taille de fenêtre, autres
  variables), télécharger « Activity recognition exp.zip » sur
  https://archive.ics.uci.edu/dataset/344, le décompresser dans `data/hhar/`, puis :

```bash
python -m transfer_forest.data.s6_hhar    # 3 Go à lire, quelques minutes
```

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

**`fatal: could not read Username for 'https://github.com'`** (dans Colab)
Le dépôt est privé et le clone s'est fait sans jeton. Suivre la procédure du jeton d'accès dans « Dans un notebook (Colab, Jupyter) ». Les erreurs `No such file or directory: 'transfer-learning'` et `does not appear to be a Python project` qui suivent viennent de ce clone raté.

**`ModuleNotFoundError: No module named 'transfer_forest'`**
Le package n'est pas installé dans l'environnement utilisé. Vérifier avec `pip show transfer_forest`, puis relancer `pip install -e .` depuis la racine du dépôt. Dans un notebook, redémarrer le noyau après l'installation.

## Référence

Segev, N., Harel, M., Mannor, S., Crammer, K., El-Yaniv, R. (2017). *Learn on Source, Refine on Target: A Model Transfer Learning Framework with Random Forests.* IEEE Transactions on Pattern Analysis and Machine Intelligence.

Les jeux de données viennent du UCI Machine Learning Repository (Letter Recognition, HHAR) et du NASA Prognostics Data Repository (C-MAPSS).
