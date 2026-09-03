"""Cross-organ MSI transfer gate as a callable module function.

Three components (see README.md):
  1. Within-organ ceiling  : leave-site-out pooled OOF AUC (honest) + random-CV (contrast)
  2. Zero-shot matrix       : source organ(s) -> never-seen target; pooled target AUC + bootstrap CI
  3. Permutation null       : within-source label shuffle -> empirical p per cell

Verdict is anchored on the confirmatory COAD<->STAD pair:
  pass cell  := bootstrap-CI lower bound > 0.60  AND  permutation p < 0.05
  STRONG     := both COAD->STAD and STAD->COAD pass
  ASYMMETRIC := one direction passes
  FAIL       := COAD<->STAD does not clear the bar -> kill the cross-organ story
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .cv import random_oof, site_out_oof
from .data import DEFAULT_FEATURE_SET, Cohort
from .metrics import permutation_null, stratified_bootstrap_auc_ci
from .probe import fit_predict

CI_LO_PASS = 0.60
P_PASS = 0.05
CONFIRMATORY = {("COAD", "STAD"), ("STAD", "COAD")}
ROOT = Path(__file__).resolve().parents[2]

Logger = Callable[[str], None]


def results_dir(features: str, root: Path = ROOT) -> Path:
    """Where the gate table for one feature set lives.

    The default set writes to ``results/``, as it always has; a named set writes
    to ``results/<features>/`` so its table never overwrites the committed one.
    """
    results = root / "results"
    return results if features == DEFAULT_FEATURE_SET else results / features


def _silent(_: str) -> None:
    pass


def banner(s: str) -> str:
    return "\n" + "=" * 78 + f"\n{s}\n" + "=" * 78


def cell_passes(ci_lo: float, perm_p: float) -> bool:
    """The gate pass rule: bootstrap lower bound above 0.60 and permutation p below 0.05."""
    return bool((ci_lo > CI_LO_PASS) and (perm_p < P_PASS))


@dataclass(frozen=True)
class GateResult:
    """Everything the gate script writes or prints, in one object."""

    table: pd.DataFrame
    verdicts: dict[tuple[str, str], bool]
    verdict: str
    confirmatory: list[dict]


def run_ceiling(cohorts: Mapping[str, Cohort], n_boot: int, seed: int, log: Logger = _silent):
    """Within-organ reference ceiling: random-CV (inflated) and site-out (honest)."""
    log(banner("WITHIN-ORGAN CEILING  (random-CV = site-inflated, site-out = honest)"))
    rows = []
    for name, c in cohorts.items():
        p_rand, _ = random_oof(c.X, c.y, seed=seed)
        p_site, tpos = site_out_oof(c.X, c.y, c.sites)
        a_rand = roc_auc_score(c.y, p_rand)
        a_site = roc_auc_score(c.y, p_site)
        lo_s, hi_s, _ = stratified_bootstrap_auc_ci(
            c.y, p_site, key=(c.name,), n_boot=n_boot, seed=seed
        )
        log(
            f"  {name}: random-CV={a_rand:.3f}  site-out={a_site:.3f} "
            f"[{lo_s:.3f},{hi_s:.3f}]  gap={a_rand - a_site:+.3f}  "
            f"| site-out test-pos/fold={tpos}"
        )
        rows.append(
            dict(component="ceiling", source=f"{name} (within)", target=name,
                 auc=a_site, ci_lo=lo_s, ci_hi=hi_s, perm_p=np.nan,
                 auc_random_cv=a_rand, n_target=c.n, pos_target=c.n_pos, role="ceiling")
        )
    return rows


def run_zeroshot(
    cohorts: Mapping[str, Cohort],
    n_perm: int,
    n_boot: int,
    seed: int,
    n_jobs: int,
    log: Logger = _silent,
):
    """Zero-shot matrix: single-source + combined-source -> each target organ."""
    log(banner("ZERO-SHOT TRANSFER  (source -> never-seen target)"))
    organs = list(cohorts)
    rows = []
    for tgt in organs:
        c = cohorts[tgt]
        Xt, yt = c.X, c.y
        log(f"\n  target = {tgt}  (n={c.n}, MSI+={c.n_pos}, prev={c.prevalence:.0%})")
        # single-source cells
        for src in organs:
            if src == tgt:
                continue
            cs = cohorts[src]
            obs, pval, _ = permutation_null(
                cs.X, cs.y, Xt, yt, n_perm=n_perm, seed=seed, n_jobs=n_jobs
            )
            lo, hi, _ = stratified_bootstrap_auc_ci(
                yt, fit_predict(cs.X, cs.y, Xt), key=(tgt,), n_boot=n_boot, seed=seed
            )
            role = "confirmatory" if (src, tgt) in CONFIRMATORY else "exploratory"
            passed = cell_passes(lo, pval)
            flag = "  <== confirmatory" if role == "confirmatory" else ""
            mark = "PASS" if passed else "----"
            log(f"    {src:>5s} -> {tgt}:  AUC={obs:.3f} [{lo:.3f},{hi:.3f}]  "
                f"perm_p={pval:.4f}  [{mark}]{flag}")
            rows.append(
                dict(component="zeroshot", source=src, target=tgt, auc=obs,
                     ci_lo=lo, ci_hi=hi, perm_p=pval, auc_random_cv=np.nan,
                     n_target=c.n, pos_target=c.n_pos, role=role, passed=passed)
            )
        # combined-source cell (exploratory: re-mixes the sample-size axis)
        others = [o for o in organs if o != tgt]
        Xs = np.concatenate([cohorts[o].X for o in others])
        ys = np.concatenate([cohorts[o].y for o in others])
        obs, pval, _ = permutation_null(Xs, ys, Xt, yt, n_perm=n_perm, seed=seed, n_jobs=n_jobs)
        lo, hi, _ = stratified_bootstrap_auc_ci(
            yt, fit_predict(Xs, ys, Xt), key=(tgt,), n_boot=n_boot, seed=seed
        )
        label = "+".join(others)
        log(f"    {label} -> {tgt}:  AUC={obs:.3f} [{lo:.3f},{hi:.3f}]  "
            f"perm_p={pval:.4f}  (combined, exploratory)")
        rows.append(
            dict(component="zeroshot", source=f"{label} (combined)", target=tgt,
                 auc=obs, ci_lo=lo, ci_hi=hi, perm_p=pval, auc_random_cv=np.nan,
                 n_target=c.n, pos_target=c.n_pos, role="exploratory", passed=None)
        )
    return rows


def verdict(zs_rows: list[dict]) -> tuple[str, list[dict]]:
    conf = [r for r in zs_rows if r["role"] == "confirmatory"]
    n_pass = sum(bool(r["passed"]) for r in conf)
    if n_pass == 2:
        v = "STRONG PASS  — cross-organ MSI transfer is real (both COAD<->STAD directions)."
    elif n_pass == 1:
        v = "ASYMMETRIC  — one COAD<->STAD direction passes; inspect before claiming."
    else:
        v = "FAIL  — COAD<->STAD does not clear the bar; the cross-organ story does not hold."
    return v, conf


def run_gate(
    cohorts: Mapping[str, Cohort],
    *,
    n_perm: int = 1000,
    n_boot: int = 2000,
    seed: int = 0,
    n_jobs: int = -1,
    log: Logger = _silent,
) -> GateResult:
    """Run the full gate on the given cohorts.

    Returns the results table (ceiling rows then zero-shot rows, in the column order
    the gate script writes), the per-cell pass verdict for every single-source
    zero-shot cell keyed by (source, target), and the overall verdict string.
    ``log`` receives every line the gate script prints while it runs.
    """
    ceil_rows = run_ceiling(cohorts, n_boot, seed, log)
    zs_rows = run_zeroshot(cohorts, n_perm, n_boot, seed, n_jobs, log)
    table = pd.DataFrame(ceil_rows + zs_rows)
    verdicts = {
        (r["source"], r["target"]): r["passed"]
        for r in zs_rows
        if r["passed"] is not None
    }
    v, conf = verdict(zs_rows)
    return GateResult(table=table, verdicts=verdicts, verdict=v, confirmatory=conf)
