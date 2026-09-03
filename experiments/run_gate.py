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

The computation lives in ``panmorph.gate.run_gate``; this script only loads the
cohorts, prints the banners, and writes the table.

Run:
  python experiments/run_gate.py            # full: 1000 perms, 2000 bootstraps
  python experiments/run_gate.py --quick    # smoke test: 100 perms, 500 bootstraps
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from panmorph.data import load_all, shared_sites  # noqa: E402
from panmorph.gate import banner, run_gate  # noqa: E402


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
    print(banner("DATA"))
    for name, c in cohorts.items():
        print(f"  {name}: n={c.n}  MSI+={c.n_pos} ({c.prevalence:.0%})  sites={c.n_sites}")
    ov = shared_sites(cohorts)
    bad = {k: v for k, v in ov.items() if v}
    print(f"  TSS disjointness: {'OK (no shared sites)' if not bad else f'SHARED: {bad}'}")

    result = run_gate(
        cohorts, n_perm=args.n_perm, n_boot=args.n_boot, seed=args.seed,
        n_jobs=args.n_jobs, log=print,
    )

    csv = args.out / "gate_results.csv"
    result.table.to_csv(csv, index=False)

    print(banner("VERDICT"))
    for r in result.confirmatory:
        m = "PASS" if r["passed"] else "fail"
        print(f"  {r['source']} -> {r['target']}: AUC={r['auc']:.3f} "
              f"[{r['ci_lo']:.3f},{r['ci_hi']:.3f}] perm_p={r['perm_p']:.4f}  [{m}]")
    print(f"\n  >>> {result.verdict}")
    print(f"\nSaved: {csv}")


if __name__ == "__main__":
    main()
