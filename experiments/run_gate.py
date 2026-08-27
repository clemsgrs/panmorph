"""Run the cross-organ MSI transfer GATE.

Three components (see README.md):
  1. Within-organ ceiling  : leave-site-out pooled OOF AUC (honest) + random-CV (contrast)
  2. Zero-shot matrix       : source organ(s) -> never-seen target; pooled target AUC + bootstrap CI
  3. Permutation null       : within-source label shuffle -> empirical p per cell

Verdict is anchored on the confirmatory COAD<->STAD pair:
  pass cell  := bootstrap-CI lower bound > 0.60  AND  permutation p < 0.05
  STRONG     := both COAD->STAD and STAD->COAD pass
  PARTIAL    := GI<->GI passes, UCEC transfer weak/asymmetric (the atlas thesis)
  FAIL       := COAD<->STAD does not clear the bar -> kill the cross-organ story

Run:
  python experiments/run_gate.py            # full: 1000 perms, 2000 bootstraps
  python experiments/run_gate.py --quick    # smoke test: 100 perms, 500 bootstraps
"""
from __future__ import annotations

import argparse
import sys
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from panmorph.cv import random_oof, site_out_oof  # noqa: E402
from panmorph.data import load_all, shared_sites  # noqa: E402
from panmorph.metrics import bootstrap_auc_ci, permutation_null  # noqa: E402
from panmorph.probe import fit_predict  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

CI_LO_PASS = 0.60
P_PASS = 0.05
CONFIRMATORY = {("COAD", "STAD"), ("STAD", "COAD")}


def banner(s: str) -> None:
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78)


def run_ceiling(cohorts, n_boot, seed):
    """Within-organ reference ceiling: random-CV (inflated) and site-out (honest)."""
    banner("WITHIN-ORGAN CEILING  (random-CV = site-inflated, site-out = honest)")
    rows = []
    for name, c in cohorts.items():
        p_rand, _ = random_oof(c.X, c.y, seed=seed)
        p_site, tpos = site_out_oof(c.X, c.y, c.sites)
        a_rand = roc_auc_score(c.y, p_rand)
        a_site = roc_auc_score(c.y, p_site)
        lo_s, hi_s, _ = bootstrap_auc_ci(c.y, p_site, n_boot=n_boot, seed=seed)
        print(
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


def run_zeroshot(cohorts, n_perm, n_boot, seed, n_jobs):
    """Zero-shot matrix: single-source + combined-source -> each target organ."""
    banner("ZERO-SHOT TRANSFER  (source -> never-seen target)")
    organs = list(cohorts)
    rows = []
    for tgt in organs:
        c = cohorts[tgt]
        Xt, yt = c.X, c.y
        print(f"\n  target = {tgt}  (n={c.n}, MSI+={c.n_pos}, prev={c.prevalence:.0%})")
        # single-source cells
        for src in organs:
            if src == tgt:
                continue
            cs = cohorts[src]
            obs, pval, _ = permutation_null(
                cs.X, cs.y, Xt, yt, n_perm=n_perm, seed=seed, n_jobs=n_jobs
            )
            lo, hi, _ = bootstrap_auc_ci(yt, fit_predict(cs.X, cs.y, Xt),
                                         n_boot=n_boot, seed=seed)
            role = "confirmatory" if (src, tgt) in CONFIRMATORY else "exploratory"
            passed = (lo > CI_LO_PASS) and (pval < P_PASS)
            flag = "  <== confirmatory" if role == "confirmatory" else ""
            mark = "PASS" if passed else "----"
            print(f"    {src:>5s} -> {tgt}:  AUC={obs:.3f} [{lo:.3f},{hi:.3f}]  "
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
        lo, hi, _ = bootstrap_auc_ci(yt, fit_predict(Xs, ys, Xt), n_boot=n_boot, seed=seed)
        label = "+".join(others)
        print(f"    {label} -> {tgt}:  AUC={obs:.3f} [{lo:.3f},{hi:.3f}]  "
              f"perm_p={pval:.4f}  (combined, exploratory)")
        rows.append(
            dict(component="zeroshot", source=f"{label} (combined)", target=tgt,
                 auc=obs, ci_lo=lo, ci_hi=hi, perm_p=pval, auc_random_cv=np.nan,
                 n_target=c.n, pos_target=c.n_pos, role="exploratory", passed=None)
        )
    return rows


def verdict(zs_rows) -> tuple[str, list[dict]]:
    conf = [r for r in zs_rows if r["role"] == "confirmatory"]
    n_pass = sum(bool(r["passed"]) for r in conf)
    if n_pass == 2:
        v = "STRONG PASS  — cross-organ MSI transfer is real (both COAD<->STAD directions)."
    elif n_pass == 1:
        v = "ASYMMETRIC  — one COAD<->STAD direction passes; inspect before claiming."
    else:
        v = "FAIL  — COAD<->STAD does not clear the bar; the cross-organ story does not hold."
    return v, conf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--quick", action="store_true", help="100 perms / 500 bootstraps")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "results")
    args = ap.parse_args()
    if args.quick:
        args.n_perm, args.n_boot = 100, 500

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"GATE run  | n_perm={args.n_perm}  n_boot={args.n_boot}  seed={args.seed}")

    cohorts = load_all()
    banner("DATA")
    for name, c in cohorts.items():
        print(f"  {name}: n={c.n}  MSI+={c.n_pos} ({c.prevalence:.0%})  sites={c.n_sites}")
    ov = shared_sites(cohorts)
    bad = {k: v for k, v in ov.items() if v}
    print(f"  TSS disjointness: {'OK (no shared sites)' if not bad else f'SHARED: {bad}'}")

    ceil_rows = run_ceiling(cohorts, args.n_boot, args.seed)
    zs_rows = run_zeroshot(cohorts, args.n_perm, args.n_boot, args.seed, args.n_jobs)

    df = pd.DataFrame(ceil_rows + zs_rows)
    csv = args.out / "gate_results.csv"
    df.to_csv(csv, index=False)

    v, conf = verdict(zs_rows)
    banner("VERDICT")
    for r in conf:
        m = "PASS" if r["passed"] else "fail"
        print(f"  {r['source']} -> {r['target']}: AUC={r['auc']:.3f} "
              f"[{r['ci_lo']:.3f},{r['ci_hi']:.3f}] perm_p={r['perm_p']:.4f}  [{m}]")
    print(f"\n  >>> {v}")
    print(f"\nSaved: {csv}")


if __name__ == "__main__":
    main()
