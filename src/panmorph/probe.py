"""The frozen-embedding linear probe with pre-registered, fixed hyperparameters.

Nothing is tuned, so nothing can leak across the source/target boundary, and the
interpretation stays parsimonious: a linear boundary in frozen PRISM space that
transfers across organs is strong evidence of shared signal.

Every fit runs with one BLAS thread. The L-BFGS solver is sensitive to the
reduction order inside multithreaded BLAS calls, so the same data can give AUCs
that differ at the 1e-3 level between machines with different core counts. A
single thread makes every number in the repository reproducible bit for bit.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


def make_probe() -> Pipeline:
    """Fixed-HP probe. StandardScaler lives inside the pipeline, so it is fit on the
    training split only (and, in zero-shot, on the source organ only)."""
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"),
            ),
        ]
    )


def fit_predict(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray) -> np.ndarray:
    """Fit a fresh probe on (Xtr, ytr); return P(MSI-high) for Xte."""
    with threadpool_limits(limits=1):
        return make_probe().fit(Xtr, ytr).predict_proba(Xte)[:, 1]
