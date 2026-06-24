"""AUC, patient-level bootstrap CI, and the within-source permutation null.

The permutation null (decision #5) is the formal significance test for the gate: it
catches pipeline artifacts / leakage that a bare CI cannot, because a bug would
inflate the shuffled runs too.
"""
from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

from .probe import fit_predict


def bootstrap_auc_ci(
    y: np.ndarray, p: np.ndarray, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float, np.ndarray]:
    """Percentile bootstrap CI for AUC, resampling patients with replacement."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    p = np.asarray(p)
    idx = np.arange(len(y))
    stats = []
    for _ in range(n_boot):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[b])) < 2:
            continue
        stats.append(roc_auc_score(y[b], p[b]))
    stats = np.asarray(stats)
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi), stats


def permutation_null(
    Xs: np.ndarray,
    ys: np.ndarray,
    Xt: np.ndarray,
    yt: np.ndarray,
    n_perm: int = 1000,
    seed: int = 0,
    n_jobs: int = -1,
) -> tuple[float, float, np.ndarray]:
    """Within-source label-permutation null for a zero-shot transfer cell.

    Shuffle the *source* labels (preserving prevalence and feature distribution,
    destroying the feature->label link), refit the probe, score the *real*-labeled
    target. Returns (observed_auc, empirical_one_sided_p, null_aucs).

    Permutations are pre-generated from a seeded RNG so the result is deterministic
    regardless of parallel scheduling.
    """
    ys = np.asarray(ys)
    yt = np.asarray(yt)
    observed = roc_auc_score(yt, fit_predict(Xs, ys, Xt))

    rng = np.random.default_rng(seed)
    perms = [rng.permutation(ys) for _ in range(n_perm)]

    def one(y_perm: np.ndarray) -> float:
        if len(np.unique(y_perm)) < 2:
            return np.nan
        return roc_auc_score(yt, fit_predict(Xs, y_perm, Xt))

    null = np.asarray(Parallel(n_jobs=n_jobs)(delayed(one)(yp) for yp in perms))
    null = null[~np.isnan(null)]
    # +1 in num and denom: the observed statistic counts as one draw (never p=0).
    p = (1 + int(np.sum(null >= observed))) / (1 + len(null))
    return float(observed), float(p), null
