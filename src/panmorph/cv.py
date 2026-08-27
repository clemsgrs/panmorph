"""Pooled out-of-fold cross-validation for the within-organ reference ceiling.

Decision #8/#9: the honest ceiling is the leave-site-out AUC (GroupKFold by TSS,
pooled OOF). Random-CV is kept only as the site-inflated contrast. We always pool
out-of-fold predictions and compute one AUC over the full cohort — never average
per-fold AUCs (a positive-light fold would yield a garbage per-fold AUC). The only
real failure mode is a *training* fold starved of positives, which the guard catches.
See README.md.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedKFold

from .probe import fit_predict


def pooled_oof(
    X: np.ndarray,
    y: np.ndarray,
    splitter,
    groups: np.ndarray | None = None,
    min_train_pos: int = 10,
) -> tuple[np.ndarray, list[int]]:
    """Return (pooled out-of-fold P(MSI-high), test-positives per fold).

    Every patient is held out exactly once. Raises if any training fold has fewer
    than ``min_train_pos`` positives.
    """
    y = np.asarray(y)
    p = np.full(len(y), np.nan)
    folds = splitter.split(X, y, groups) if groups is not None else splitter.split(X, y)
    test_pos = []
    for tr, te in folds:
        n_tr_pos = int(y[tr].sum())
        if n_tr_pos < min_train_pos:
            raise ValueError(
                f"training fold has {n_tr_pos} positives (< min_train_pos={min_train_pos})"
            )
        p[te] = fit_predict(X[tr], y[tr], X[te])
        test_pos.append(int(y[te].sum()))
    if np.isnan(p).any():
        raise AssertionError("some patient was never held out")
    return p, test_pos


def random_oof(X, y, n_splits: int = 5, seed: int = 0):
    """Site-inflated reference: plain stratified CV (sites can appear in train+test)."""
    return pooled_oof(X, y, StratifiedKFold(n_splits, shuffle=True, random_state=seed))


def site_out_oof(X, y, sites, n_splits: int = 5):
    """Honest ceiling: site-grouped CV — no TSS appears in both train and test."""
    return pooled_oof(X, y, GroupKFold(n_splits=n_splits), groups=sites)
