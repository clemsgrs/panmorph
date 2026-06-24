"""Cohort registry and patient-level PRISM feature/label loading.

Each cohort has one 1280-dim PRISM embedding per patient (all of a patient's slides
already aggregated into a single .pt), so there is no same-patient slide leakage.
See docs/data.md for the verified inventory.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path("/data/pathology/projects/clement/mutation-prediction")
FEATURES = ROOT / "features" / "prism"
CSVS = ROOT / "csvs"

# cohort -> (label csv, label column, feature directory). MSI gate cohorts only.
# BLCA-MSI and PRAD-MSI are excluded: ~3 positives each (dead). See docs/data.md.
MSI_COHORTS: dict[str, tuple[Path, str, Path]] = {
    "COAD": (CSVS / "tcga-coad/dx+msi.csv", "msi_high", FEATURES / "lxbzb8rd/features"),
    "UCEC": (CSVS / "tcga-ucec/dx+msi.csv", "msi_high", FEATURES / "kooqa1ym/features"),
    "STAD": (CSVS / "tcga-stad/dx+msi.csv", "msi_high", FEATURES / "oowdp902/features"),
}


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


def load_cohort(name: str, registry: dict = MSI_COHORTS) -> Cohort:
    """Load one cohort's PRISM features + labels, one row per patient."""
    csv, col, fdir = registry[name]
    df = pd.read_csv(csv)[["case_id", col]].dropna()
    X, y, sites, ids = [], [], [], []
    missing = 0
    for cid, label in zip(df.case_id, df[col]):
        f = fdir / f"{cid}.pt"
        if not f.exists():
            missing += 1
            continue
        v = torch.load(f, map_location="cpu", weights_only=False)
        X.append(v.reshape(-1).float().numpy())
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


def load_all(registry: dict = MSI_COHORTS) -> dict[str, Cohort]:
    return {name: load_cohort(name, registry) for name in registry}


def shared_sites(cohorts: dict[str, Cohort]) -> dict[tuple[str, str], list[str]]:
    """Pairwise TSS overlap between cohorts. Empty everywhere => the gate's
    site-shortcut immunity holds by construction (see docs/methods-notes.md)."""
    out = {}
    for a, b in itertools.combinations(cohorts, 2):
        out[(a, b)] = sorted(set(cohorts[a].sites) & set(cohorts[b].sites))
    return out
