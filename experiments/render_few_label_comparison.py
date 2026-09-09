"""Compare the few-label AUC gain of several feature sets.

Each argument names one feature set and its complete few-label bundle as ``name=dir``.
The script writes one figure with one gain series per feature set and prints the
Markdown tables of the two gastrointestinal directions for the README.

Run:
  python experiments/render_few_label_comparison.py prism=results/few-label \\
      prism2-base=results/prism2-base/few-label prism2-diagnostic=results/prism2-diagnostic/few-label \\
      --out results/few_label_lift_by_feature_set
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from panmorph.few_label_comparison import (  # noqa: E402
    build_lift_comparison, render_figure, render_markdown,
)

ROOT = Path(__file__).resolve().parent.parent


def named_bundle(arg: str) -> tuple[str, Path]:
    name, sep, path = arg.partition("=")
    if not sep or not name or not path:
        raise argparse.ArgumentTypeError(f"expected name=dir, got {arg!r}")
    return name, Path(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundles", nargs="+", type=named_bundle, metavar="NAME=DIR",
                        help="a feature-set name and its few-label bundle directory")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "few_label_lift_by_feature_set",
                        help="figure stem; .png and .pdf are written next to it")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summaries = {
        name: pd.read_csv(path / "few_label_summaries.csv") for name, path in args.bundles
    }
    comparison = build_lift_comparison(summaries)
    render_figure(comparison, args.out)
    print(render_markdown(comparison), end="")


if __name__ == "__main__":
    main()
