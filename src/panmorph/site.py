"""Site-decodability diagnostic: can TCGA tissue-source-site be predicted from PRISM?

This is a *confound diagnostic* — it motivates confound 2 (the site shortcut) that the
gate must defeat. It is NOT part of the gate decision; the gate defeats the site
shortcut structurally via cross-cohort TSS-disjointness (see docs/methods-notes.md).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

from .data import Cohort
from .probe import make_probe


def site_decodability(
    cohort: Cohort, min_n: int = 8, n_splits: int = 5, seed: int = 0
) -> dict:
    """Balanced accuracy of predicting TSS from PRISM embeddings.

    Restricted to sites with >= ``min_n`` cases (rare sites are dropped). Compares
    against the multiclass chance level (1 / #sites) and the majority-class baseline.
    """
    sites = cohort.sites
    vc = pd.Series(sites).value_counts()
    keep = vc[vc >= min_n].index
    m = np.isin(sites, keep)
    X, s = cohort.X[m], sites[m]
    classes, counts = np.unique(s, return_counts=True)

    bas = []
    for tr, te in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, s):
        clf = make_probe().fit(X[tr], s[tr])  # multinomial logistic (lbfgs) for >2 classes
        bas.append(balanced_accuracy_score(s[te], clf.predict(X[te])))

    return dict(
        n=int(m.sum()),
        n_sites=len(classes),
        chance=1.0 / len(classes),
        majority=float(counts.max() / counts.sum()),
        bal_acc=float(np.mean(bas)),
        bal_acc_std=float(np.std(bas)),
    )
