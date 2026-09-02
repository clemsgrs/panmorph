"""The frozen-embedding linear probe with pre-registered, fixed hyperparameters.

Nothing is tuned, so nothing can leak across the source/target boundary, and the
interpretation stays parsimonious: a linear boundary in frozen PRISM space that
transfers across organs is strong evidence of shared signal.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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
    return make_probe().fit(Xtr, ytr).predict_proba(Xte)[:, 1]
