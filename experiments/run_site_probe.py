"""Site-decodability diagnostic (deck slide "Two ways to be fooled", confound 2).

Is TCGA tissue-source-site predictable from frozen PRISM embeddings? If yes (far
above chance), site is a shortcut the gate must defeat. Not part of the gate decision.

Run:  python experiments/run_site_probe.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from panmorph.data import load_all  # noqa: E402
from panmorph.site import site_decodability  # noqa: E402


def main() -> None:
    cohorts = load_all()
    print(f"{'organ':6s} {'site bal-acc':>16s} {'chance':>8s} {'majority':>9s} "
          f"{'sites>=8':>9s} {'n':>5s}")
    for name, c in cohorts.items():
        d = site_decodability(c)
        print(f"{name:6s} {d['bal_acc']:>10.3f}±{d['bal_acc_std']:.3f} "
              f"{d['chance']:>8.3f} {d['majority']:>9.3f} {d['n_sites']:>9d} {d['n']:>5d}")


if __name__ == "__main__":
    main()
