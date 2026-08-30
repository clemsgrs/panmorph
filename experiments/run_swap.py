"""Run the exploratory budget-matched COAD↔STAD swap on completed E1."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from panmorph.e1_runner import DEFAULT_WORKERS  # noqa: E402
from panmorph.swap_runner import run_swap_bundle, validate_swap_bundle  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "e1")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--validate-only", action="store_true",
        help="validate an existing completed E1+swap result bundle",
    )
    args = parser.parse_args()
    if args.validate_only:
        validate_swap_bundle(args.out)
        print(f"Validated exploratory swap bundle at {args.out}")
        return
    run_swap_bundle(args.out, workers=args.workers)
    print(f"Saved validated exploratory swap results to {args.out}")


if __name__ == "__main__":
    main()
