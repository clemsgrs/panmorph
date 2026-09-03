"""Render the verdict matrix of several feature sets as Markdown for the README.

Each argument names one feature set and its gate table as ``name=path``. The first
name is not special: the flags compare each feature set with ``--reference``
(default ``prism``), which must be one of the names.

Run:
  python experiments/render_verdict_matrix.py prism=results/gate_results.csv \\
      prism2-base=results/prism2-base/gate_results.csv prism2-diagnostic=results/prism2-diagnostic/gate_results.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from panmorph.verdict_matrix import REFERENCE, build_verdict_matrix, render_markdown  # noqa: E402


def named_table(arg: str) -> tuple[str, Path]:
    name, sep, path = arg.partition("=")
    if not sep or not name or not path:
        raise argparse.ArgumentTypeError(f"expected name=path, got {arg!r}")
    return name, Path(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tables", nargs="+", type=named_table, metavar="NAME=PATH",
                        help="a feature-set name and its gate_results.csv")
    parser.add_argument("--reference", default=REFERENCE,
                        help="the feature set the pass and fail flags compare with")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    tables = {name: pd.read_csv(path) for name, path in args.tables}
    matrix = build_verdict_matrix(tables, reference=args.reference)
    print(render_markdown(matrix, reference=args.reference))


if __name__ == "__main__":
    main()
