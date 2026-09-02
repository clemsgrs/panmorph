"""Run the few-label transfer analysis and write its validated result bundle."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from panmorph.few_label_runner import (  # noqa: E402
    DEFAULT_WORKERS,
    CompleteBundleError,
    run_few_label_bundle,
)

ROOT = Path(__file__).resolve().parent.parent


def build_parser(root: Path = ROOT) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=root / "results" / "few-label")
    parser.add_argument("--profile", choices=("quick", "full"), default="full")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        run_few_label_bundle(args.out, profile=args.profile, workers=args.workers)
    except CompleteBundleError as existing:
        print(f"Nothing to do: {existing}")
        return
    label = "REPORTABLE" if args.profile == "full" else "NON-REPORTABLE"
    print(f"Saved validated few-label {args.profile} bundle ({label}) to {args.out}")


if __name__ == "__main__":
    main()
