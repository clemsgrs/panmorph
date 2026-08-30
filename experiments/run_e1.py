"""Run the registered E1 paired OOF matrix and write auditable records."""
from __future__ import annotations

import argparse
import ast
import csv
import sys
from dataclasses import asdict, fields
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from panmorph.data import load_all  # noqa: E402
from panmorph.e1 import (  # noqa: E402
    ConfirmatoryResult,
    DrawRecord,
    E1Inference,
    PredictionRecord,
    TraceResult,
    count_source_cases,
    estimate_e1_matrix,
    run_confirmatory_test,
    run_e1_matrix,
    validate_phase1_anchors,
)
from panmorph.e1_runner import (  # noqa: E402
    DEFAULT_WORKERS,
    migrate_e1_bundle_v1_to_v2,
    run_e1_bundle,
    validate_e1_bundle,
)

ROOT = Path(__file__).resolve().parent.parent


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    with path.open(newline="") as handle:
        return tuple(csv.DictReader(handle))


def read_prediction_records(path: Path) -> tuple[PredictionRecord, ...]:
    """Load the complete stored issue-#6 OOF prediction artifact."""
    records = []
    for row in _read_rows(path):
        records.append(
            PredictionRecord(
                draw_seed=None if row["draw_seed"] == "" else int(row["draw_seed"]),
                k="all" if row["k"] == "all" else int(row["k"]),
                fold=int(row["fold"]),
                held_out_sites=tuple(ast.literal_eval(row["held_out_sites"])),
                arm=row["arm"],
                source=row["source"],
                target=row["target"],
                case_id=row["case_id"],
                label=int(row["label"]),
                score=float(row["score"]),
            )
        )
    return tuple(records)


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


def write_registered_inference(
    inference: E1Inference,
    confirmatory: ConfirmatoryResult,
    out: Path,
) -> None:
    """Write registered descriptive estimates and the sole permutation test."""
    estimate_fields = [
        "source", "target", "base", "k", "n_draws",
        "warm_auc", "warm_ci_lower", "warm_ci_upper",
        "cold_auc", "cold_ci_lower", "cold_ci_upper",
        "lift", "lift_ci_lower", "lift_ci_upper",
        "rank_warm_auc", "rank_cold_auc", "rank_lift", "rank_diverged",
        "confirmatory", "permutation_p",
    ]
    with (out / "e1_estimates.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=estimate_fields)
        writer.writeheader()
        for cell in inference.cells:
            writer.writerow(
                {
                    "source": cell.source,
                    "target": cell.target,
                    "base": cell.base,
                    "k": cell.k,
                    "n_draws": cell.n_draws,
                    "warm_auc": cell.warm.point,
                    "warm_ci_lower": cell.warm.lower,
                    "warm_ci_upper": cell.warm.upper,
                    "cold_auc": cell.cold.point,
                    "cold_ci_lower": cell.cold.lower,
                    "cold_ci_upper": cell.cold.upper,
                    "lift": cell.lift.point,
                    "lift_ci_lower": cell.lift.lower,
                    "lift_ci_upper": cell.lift.upper,
                    "rank_warm_auc": cell.rank_warm,
                    "rank_cold_auc": cell.rank_cold,
                    "rank_lift": cell.rank_lift,
                    "rank_diverged": cell.rank_diverged,
                    "confirmatory": cell.confirmatory,
                    "permutation_p": confirmatory.p_value if cell.confirmatory else "",
                }
            )

    equivalence_fields = [
        "source", "target", "base", "local_positive_equivalence",
        "local_ci_lower", "local_ci_upper", "local_point_censored",
        "local_ci_lower_censored", "local_ci_upper_censored",
        "average_source_case_equivalence", "average_ci_lower", "average_ci_upper",
        "average_point_censored", "average_ci_lower_censored",
        "average_ci_upper_censored",
    ]
    with (out / "e1_equivalence.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=equivalence_fields)
        writer.writeheader()
        for cell in inference.equivalences:
            local = cell.local_positive
            average = local.average_source_case
            writer.writerow(
                {
                    "source": cell.source,
                    "target": cell.target,
                    "base": cell.base,
                    "local_positive_equivalence": local.point.value,
                    "local_ci_lower": local.interval.lower,
                    "local_ci_upper": local.interval.upper,
                    "local_point_censored": local.point.censored,
                    "local_ci_lower_censored": local.interval.lower_censored,
                    "local_ci_upper_censored": local.interval.upper_censored,
                    "average_source_case_equivalence": average.point.value,
                    "average_ci_lower": average.interval.lower,
                    "average_ci_upper": average.interval.upper,
                    "average_point_censored": average.point.censored,
                    "average_ci_lower_censored": average.interval.lower_censored,
                    "average_ci_upper_censored": average.interval.upper_censored,
                }
            )

    with (out / "e1_permutation_null.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("permutation", "null_mean_lift"))
        writer.writeheader()
        for permutation, null_lift in enumerate(confirmatory.null_lifts):
            writer.writerow({"permutation": permutation, "null_mean_lift": null_lift})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "e1")
    parser.add_argument("--profile", choices=("quick", "full"), default="full")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--migrate-v1", action="store_true",
        help="normalize an existing complete v1 bundle in place without recomputation",
    )
    args = parser.parse_args()
    if args.migrate_v1:
        migrate_e1_bundle_v1_to_v2(args.out)
        validate_e1_bundle(args.out)
        print(f"Migrated and validated E1 bundle v2 at {args.out}")
        return
    run_e1_bundle(args.out, profile=args.profile, workers=args.workers)
    label = "REPORTABLE" if args.profile == "full" else "NON-REPORTABLE"
    print(f"Saved validated E1 {args.profile} bundle ({label}) to {args.out}")


if __name__ == "__main__":
    main()
