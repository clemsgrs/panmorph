"""AUC, stratified patient bootstrap, and the within-source permutation null.

The permutation null is the formal significance test for the gate: it catches
pipeline artifacts and leakage that a bare CI cannot, because a bug would inflate
the shuffled runs too.

Every bootstrap interval in the project uses the same label-stratified patient
resampling schedule, keyed by a seed and a cohort name, so the same cell reports
one interval wherever it appears.
"""
from __future__ import annotations

import hashlib

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

from .probe import fit_predict

BOOTSTRAP_REPLICATES = 2_000


def keyed_rng(seed: int, *key: str) -> np.random.Generator:
    """Return a generator whose state depends only on the seed and string key."""
    material = "\x1f".join((str(seed), *key)).encode()
    digest = int.from_bytes(hashlib.sha256(material).digest()[:8], "little")
    return np.random.default_rng(digest)


def stratified_bootstrap_indices(
    labels: np.ndarray,
    *,
    seed: int = 0,
    key: tuple[str, ...] = (),
    n_replicates: int = BOOTSTRAP_REPLICATES,
) -> np.ndarray:
    """Return a deterministic label-stratified patient bootstrap schedule."""
    labels = np.asarray(labels)
    classes = np.unique(labels)
    if len(classes) < 2:
        raise ValueError("patient bootstraps require at least two labels")
    rng = keyed_rng(seed, "bootstrap", *key)
    strata = tuple(np.flatnonzero(labels == label) for label in classes)
    return np.asarray(
        [
            np.concatenate(
                [rng.choice(indices, size=len(indices), replace=True) for indices in strata]
            )
            for _ in range(n_replicates)
        ],
        dtype=int,
    )


def bootstrap_auc(
    labels: np.ndarray, scores: np.ndarray, schedule: np.ndarray
) -> np.ndarray:
    """Evaluate a stratified bootstrap schedule with bounded vectorized comparisons."""
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    scheduled_labels = labels[schedule[0]]
    positive_columns = scheduled_labels == 1
    negative_columns = ~positive_columns
    aucs = np.empty(len(schedule), dtype=float)
    for start in range(0, len(schedule), 100):
        stop = min(start + 100, len(schedule))
        sampled = scores[schedule[start:stop]]
        positive = sampled[:, positive_columns, None]
        negative = sampled[:, None, negative_columns]
        aucs[start:stop] = np.mean(positive > negative, axis=(1, 2)) + 0.5 * np.mean(
            positive == negative, axis=(1, 2)
        )
    return aucs


def stratified_bootstrap_auc_ci(
    y: np.ndarray,
    p: np.ndarray,
    *,
    key: tuple[str, ...],
    n_boot: int = BOOTSTRAP_REPLICATES,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, np.ndarray]:
    """Percentile interval for AUC from the shared stratified patient bootstrap."""
    schedule = stratified_bootstrap_indices(y, seed=seed, key=key, n_replicates=n_boot)
    stats = bootstrap_auc(y, p, schedule)
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
