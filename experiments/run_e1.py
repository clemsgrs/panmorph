"""Run the registered E1 paired OOF matrix and write auditable records."""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict, fields
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from panmorph.data import load_all  # noqa: E402
from panmorph.e1 import (  # noqa: E402
    TraceResult,
    run_e1_matrix,
    validate_phase1_anchors,
)

ROOT = Path(__file__).resolve().parent.parent


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    with path.open(newline="") as handle:
        return tuple(csv.DictReader(handle))


def _write_records(path: Path, records: Iterable[object]) -> None:
    records = iter(records)
    first = next(records, None)
    if first is None:
        raise ValueError(f"refusing to write empty E1 records: {path.name}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=[field.name for field in fields(first)]
        )
        writer.writeheader()
        writer.writerow(asdict(first))
        for record in records:
            writer.writerow(asdict(record))


def write_reportable_results(
    result: TraceResult,
    phase1_path: Path,
    out: Path,
) -> None:
    """Validate Phase-1 anchors before emitting any reportable E1 file."""
    validate_phase1_anchors(result.aucs, _read_rows(phase1_path))
    out.mkdir(parents=True, exist_ok=True)
    _write_records(out / "e1_draws.csv", result.draws)
    _write_records(out / "e1_predictions.csv", result.predictions)
    _write_records(out / "e1_aucs.csv", result.aucs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    parser.add_argument(
        "--phase1", type=Path, default=ROOT / "results" / "gate_results.csv"
    )
    args = parser.parse_args()

    result = run_e1_matrix(load_all())
    write_reportable_results(result, args.phase1, args.out)
    print(
        f"Saved {len(result.aucs)} AUCs, {len(result.predictions)} predictions, "
        f"and {len(result.draws)} training rows to {args.out}"
    )


if __name__ == "__main__":
    main()
