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
    DrawRecord,
    PredictionRecord,
    TraceResult,
    run_e1_matrix,
    validate_phase1_anchors,
)

ROOT = Path(__file__).resolve().parent.parent


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    with path.open(newline="") as handle:
        return tuple(csv.DictReader(handle))


def _write_records(
    path: Path,
    records: Iterable[object],
    record_type: type,
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=[field.name for field in fields(record_type)]
        )
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def _write_results(path: Path, result: TraceResult) -> None:
    fieldnames = [
        "experiment",
        "source",
        "target",
        "base",
        "arm",
        "k",
        "draw_seed",
        "auc",
        "rank_auc",
        "rank_gap",
        "rank_diverged",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in result.aucs:
            base = (
                "target-only"
                if record.arm == "cold"
                else "pooled" if "+" in record.source else "single"
            )
            writer.writerow(
                {
                    "experiment": "E1",
                    "source": record.source,
                    "target": record.target,
                    "base": base,
                    "arm": record.arm,
                    "k": record.k,
                    "draw_seed": record.draw_seed,
                    "auc": record.raw_auc,
                    "rank_auc": record.rank_auc,
                    "rank_gap": record.rank_gap,
                    "rank_diverged": record.rank_diverged,
                }
            )


def write_reportable_results(
    result: TraceResult,
    phase1_path: Path,
    out: Path,
) -> None:
    """Validate Phase-1 anchors before emitting any reportable E1 file."""
    validate_phase1_anchors(result.aucs, _read_rows(phase1_path))
    out.mkdir(parents=True, exist_ok=True)
    _write_records(out / "e1_draws.csv", result.draws, DrawRecord)
    _write_records(
        out / "e1_predictions.csv", result.predictions, PredictionRecord
    )
    _write_results(out / "e1_results.csv", result)


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
