# transfer-forest

Transfer learning pour les forêts aléatoires (SER, STRUT, MIX — Segev et al., 2017).
Projet IMT Atlantique — MBDA.

## Organisation

```
data/
    letter_recognition.csv   données brutes du scénario S1
notebooks/
    experience_references_s1.ipynb   références sans transfert sur S1
transfer_forest/
    contracts.py          contrat commun : formats, forêt, Scenario, Context, Method
    splits.py             découpage de la cible et tirages emboîtés
    data/                 chargeurs de scénarios (annuaire par nom : S1, ...)
    methods/
        __init__.py       annuaire des méthodes (par nom officiel)
        references.py     références sans transfert : Src, Tgt, Union, Union-P, Oracle
tests/
    test_splits.py        tests du découpage
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
  classes ; il ne sert qu'à l'évaluation.
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

## Tests

```bash
python -m pytest -q
```
