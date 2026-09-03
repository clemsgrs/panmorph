"""Cohort registry, named feature sets, and patient-level feature/label loading.

Each cohort has one slide-level embedding per patient (all of a patient's slides
already aggregated into a single .pt), so there is no same-patient slide leakage.
The committed label tables under ``data/`` are the verified inventory. A named
feature set records which extractor produced the embeddings, their width, and the
per-cohort directory; PRISM (1280-dim) is the default.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import torch

# Labels are committed in-repo (data/); features are too large and stay external.
ROOT = Path("/data/pathology/projects/clement/mutation-prediction")
FEATURES_ROOT = ROOT / "features"
FEATURES = FEATURES_ROOT / "prism"
LABELS = Path(__file__).resolve().parents[2] / "data"

# cohort -> (label csv, label column, feature directory). MSI gate cohorts only.
# BLCA-MSI and PRAD-MSI are excluded: ~3 positives each (dead). See README.md.
MSI_COHORTS: dict[str, tuple[Path, str, Path]] = {
    "COAD": (LABELS / "tcga-coad/dx+msi.csv", "msi_high", FEATURES / "lxbzb8rd/features"),
    "UCEC": (LABELS / "tcga-ucec/dx+msi.csv", "msi_high", FEATURES / "kooqa1ym/features"),
    "STAD": (LABELS / "tcga-stad/dx+msi.csv", "msi_high", FEATURES / "oowdp902/features"),
}


@dataclass(frozen=True)
class FeatureSet:
    """One extractor's slide-level embeddings: width and per-cohort directory."""

    name: str
    extractor: str
    width: int
    dirs: Mapping[str, Path]  # cohort -> directory of one <case_id>.pt per case
    # cohort -> short identity of that directory, recorded in run manifests.
    identities: Mapping[str, str]


# PRISM2 output variant -> embedding width. Files live under
# <features root>/prism2-<variant>/<cohort>/ (see experiments/extract_prism2.py).
PRISM2_WIDTHS: dict[str, int] = {"base": 2560, "diagnostic": 3072}


def prism2_feature_set(variant: str, features_root: Path = FEATURES_ROOT) -> FeatureSet:
    name = f"prism2-{variant}"
    return FeatureSet(
        name=name,
        extractor="PRISM2",
        width=PRISM2_WIDTHS[variant],
        dirs={cohort: features_root / name / cohort for cohort in MSI_COHORTS},
        identities={cohort: f"{name}/{cohort}" for cohort in MSI_COHORTS},
    )


DEFAULT_FEATURE_SET = "prism"
FEATURE_SETS: dict[str, FeatureSet] = {
    "prism": FeatureSet(
        name="prism",
        extractor="PRISM",
        width=1280,
        dirs={cohort: entry[2] for cohort, entry in MSI_COHORTS.items()},
        # PRISM directories are <hash>/features; the hash names the extraction run.
        identities={cohort: entry[2].parent.name for cohort, entry in MSI_COHORTS.items()},
    ),
    **{f"prism2-{variant}": prism2_feature_set(variant) for variant in PRISM2_WIDTHS},
}


def feature_set(name: str, feature_sets: Mapping[str, FeatureSet] = FEATURE_SETS) -> FeatureSet:
    """Look up a registered feature set; reject a name that is not registered."""
    try:
        return feature_sets[name]
    except KeyError:
        raise ValueError(
            f"unknown feature set {name!r}; registered: {sorted(feature_sets)}"
        ) from None


def tss(case_id: str) -> str:
    """TCGA-XX-YYYY -> XX, the tissue-source-site (contributing center) code."""
    return case_id.split("-")[1]


@dataclass(frozen=True)
class Cohort:
    """One organ's patient-level data."""

    name: str
    X: np.ndarray  # (n, d) float32 PRISM embeddings
    y: np.ndarray  # (n,) int in {0, 1}, MSI-high status
    sites: np.ndarray  # (n,) str, TSS code per patient
    case_ids: np.ndarray  # (n,) str

    @property
    def n(self) -> int:
        return len(self.y)

    @property
    def n_pos(self) -> int:
        return int(self.y.sum())

    @property
    def prevalence(self) -> float:
        return self.y.mean()

    @property
    def n_sites(self) -> int:
        return len(np.unique(self.sites))


def load_cohort(
    name: str,
    registry: dict = MSI_COHORTS,
    features: str = DEFAULT_FEATURE_SET,
    feature_sets: Mapping[str, FeatureSet] = FEATURE_SETS,
) -> Cohort:
    """Load one cohort's features + labels under a named feature set, one row per patient.

    Every loaded vector must have the feature set's registered width.
    """
    csv, col, _ = registry[name]
    selected = feature_set(features, feature_sets)
    fdir = selected.dirs[name]
    df = pd.read_csv(csv)[["case_id", col]].dropna()
    X, y, sites, ids = [], [], [], []
    missing = 0
    for cid, label in zip(df.case_id, df[col]):
        f = fdir / f"{cid}.pt"
        if not f.exists():
            missing += 1
            continue
        v = torch.load(f, map_location="cpu", weights_only=False).reshape(-1)
        if v.numel() != selected.width:
            raise ValueError(
                f"[{name}] case {cid}: feature set {selected.name!r} file {f} has "
                f"width {v.numel()}, expected {selected.width}"
            )
        X.append(v.float().numpy())
        y.append(int(label))
        sites.append(tss(cid))
        ids.append(cid)
    if missing:
        print(f"[{name}] WARNING: {missing} labeled cases had no feature file (skipped)")
    return Cohort(
        name=name,
        X=np.asarray(X, dtype=np.float32),
        y=np.asarray(y, dtype=int),
        sites=np.asarray(sites),
        case_ids=np.asarray(ids),
    )


def load_all(
    registry: dict = MSI_COHORTS, features: str = DEFAULT_FEATURE_SET
) -> dict[str, Cohort]:
    return {name: load_cohort(name, registry, features) for name in registry}


def shared_sites(cohorts: dict[str, Cohort]) -> dict[tuple[str, str], list[str]]:
    """Pairwise TSS overlap between cohorts. Empty everywhere => the gate's
    site-shortcut immunity holds by construction (see README.md)."""
    out = {}
    for a, b in itertools.combinations(cohorts, 2):
        out[(a, b)] = sorted(set(cohorts[a].sites) & set(cohorts[b].sites))
    return out
