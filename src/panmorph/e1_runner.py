"""Auditable, resumable execution boundary for the registered E1 experiment."""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Iterable, Literal, Mapping

import matplotlib
import numpy as np
from joblib import Parallel, delayed, parallel_config

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .data import Cohort, MSI_COHORTS, load_cohort
from .e1 import (
    AucRecord,
    ConfirmatoryResult,
    DrawRecord,
    E1Inference,
    PredictionRecord,
    Rung,
    TraceResult,
    _pool_cohorts,
    _restore_pooled_provenance,
    confirmatory_observed,
    empirical_superiority_p,
    evaluate_confirmatory_null,
    estimate_e1_matrix,
    preflight_rungs,
    sample_rung,
    source_label_permutations,
    summarize_predictions,
    trace_paired_cell,
    validate_phase1_anchors,
)

BUNDLE_SCHEMA_VERSION = "panmorph.e1.bundle/v1"
DEFAULT_WORKERS = 8
ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class E1Profile:
    name: Literal["quick", "full"]
    reportable: bool
    cohorts: tuple[str, ...]
    pairs: tuple[tuple[str, str], ...] | None
    rungs: tuple[Rung, ...]
    draw_ids: tuple[int, ...]
    bootstrap_replicates: int
    permutations: int
    permutation_chunk_size: int


PROFILES = {
    "quick": E1Profile(
        "quick", False, ("COAD", "STAD"), (("COAD", "STAD"),),
        (0, 10, "all"), (0,), 50, 9, 3,
    ),
    "full": E1Profile(
        "full", True, ("COAD", "UCEC", "STAD"), None,
        (0, 3, 5, 10, 25, 40, "all"), tuple(range(20)), 2_000, 999, 50,
    ),
}

DRAW_FIELDS = tuple(field.name for field in fields(DrawRecord))
PREDICTION_FIELDS = tuple(field.name for field in fields(PredictionRecord))
AUC_FIELDS = (
    "experiment", "source", "target", "base", "arm", "k", "draw_seed",
    "auc", "rank_auc", "rank_gap", "rank_diverged",
)
SUMMARY_FIELDS = (
    "source", "target", "base", "k", "n_draws", "warm_auc",
    "warm_ci_lower", "warm_ci_upper", "cold_auc", "cold_ci_lower",
    "cold_ci_upper", "lift", "lift_ci_lower", "lift_ci_upper",
    "rank_warm_auc", "rank_cold_auc", "rank_lift", "rank_diverged",
    "confirmatory", "permutation_p",
)
EQUIVALENCE_FIELDS = (
    "source", "target", "base", "local_positive_equivalence",
    "local_ci_lower", "local_ci_upper", "local_point_censored",
    "local_ci_lower_censored", "local_ci_upper_censored",
    "average_source_case_equivalence", "average_ci_lower", "average_ci_upper",
    "average_point_censored", "average_ci_lower_censored",
    "average_ci_upper_censored",
)
NULL_FIELDS = ("permutation", "null_mean_lift")


def _sha256(parts: Iterable[bytes]) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part)
    return digest.hexdigest()


def _code_identity() -> dict[str, str]:
    paths = tuple(sorted((ROOT / "src/panmorph").glob("*.py"))) + (
        ROOT / "experiments/run_e1.py",
    )
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    return {"commit": commit, "sha256": _sha256(path.read_bytes() for path in paths)}


def _cohort_identity(cohort: Cohort) -> dict[str, object]:
    return {
        "cases": cohort.n,
        "positives": cohort.n_pos,
        "sha256": _sha256(
            (
                np.ascontiguousarray(cohort.X).tobytes(),
                np.ascontiguousarray(cohort.y).tobytes(),
                "\0".join(map(str, cohort.sites)).encode(),
                "\0".join(map(str, cohort.case_ids)).encode(),
            )
        ),
    }


def _manifest(profile: E1Profile, cohorts: Mapping[str, Cohort], workers: int) -> dict:
    feature_hashes = {
        name: MSI_COHORTS[name][2].parent.name
        for name in profile.cohorts if name in MSI_COHORTS
    }
    return {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "profile": profile.name,
        "reportable": profile.reportable,
        "status": "running",
        "code": _code_identity(),
        "features": {"extractor": "PRISM", "hashes": feature_hashes},
        "cohorts": {name: _cohort_identity(cohorts[name]) for name in profile.cohorts},
        "model": {
            "type": "StandardScaler+LogisticRegression",
            "grid": {"C": [1.0], "class_weight": ["balanced"], "max_iter": [2000]},
        },
        "splitter": {"type": "GroupKFold", "groups": "tissue_source_site", "folds": 5},
        "configuration": {
            "pairs": profile.pairs or "registered_full_matrix",
            "rungs": list(profile.rungs),
            "draw_ids": list(profile.draw_ids),
            "bootstrap_replicates": profile.bootstrap_replicates,
            "permutations": profile.permutations,
            "permutation_chunk_size": profile.permutation_chunk_size,
            "seeds": {"draws": list(profile.draw_ids), "bootstrap": 0, "permutation": 0},
        },
        "resources": {"workers": workers, "threads_per_worker": 1},
        "elapsed_seconds": None,
    }


def _identity(manifest: Mapping[str, object]) -> dict[str, object]:
    identity = {
        key: manifest[key]
        for key in (
            "schema_version", "profile", "reportable", "code", "features", "cohorts",
            "model", "splitter", "configuration",
        )
    }
    return json.loads(json.dumps(identity, sort_keys=True))


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: Iterable[Mapping]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _read_csv(path: Path, expected: tuple[str, ...]) -> tuple[dict[str, str], ...]:
    if not path.is_file():
        raise ValueError(f"missing bundle artifact: {path.name}")
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != expected:
            raise ValueError(
                f"invalid schema for {path.name}: expected {expected}, got {reader.fieldnames}"
            )
        return tuple(reader)


def _rung(value: str) -> Rung:
    return "all" if value == "all" else int(value)


def _optional_int(value: str) -> int | None:
    return None if value == "" else int(value)


def _draw_record(row: Mapping[str, str]) -> DrawRecord:
    return DrawRecord(
        _optional_int(row["draw_seed"]), _rung(row["k"]), int(row["fold"]),
        tuple(ast.literal_eval(row["held_out_sites"])), row["arm"], row["source"],
        row["target"], row["origin"], row["cohort"], row["case_id"],
        int(row["label"]), row["site"],
    )


def _prediction_record(row: Mapping[str, str]) -> PredictionRecord:
    return PredictionRecord(
        _optional_int(row["draw_seed"]), _rung(row["k"]), int(row["fold"]),
        tuple(ast.literal_eval(row["held_out_sites"])), row["arm"], row["source"],
        row["target"], row["case_id"], int(row["label"]), float(row["score"]),
    )


def _auc_record(row: Mapping[str, str]) -> AucRecord:
    return AucRecord(
        _optional_int(row["draw_seed"]), _rung(row["k"]), row["arm"], row["source"],
        row["target"], float(row["raw_auc"]), float(row["rank_auc"]),
        float(row["rank_gap"]), row["rank_diverged"] == "True",
    )


def _checkpoint_name(source: str, target: str, k: Rung, draw: int | None, arm: str) -> str:
    material = json.dumps([source, target, k, draw, arm], separators=(",", ":"))
    return hashlib.sha256(material.encode()).hexdigest()[:16]


def _load_cell(
    path: Path,
    *,
    source: str,
    source_cohort: Cohort,
    target: Cohort,
    k: Rung,
    draw_seed: int | None,
    arm: str,
) -> TraceResult | None:
    if not path.is_dir():
        return None
    try:
        draws = tuple(_draw_record(row) for row in _read_csv(path / "draws.csv", DRAW_FIELDS))
        predictions = tuple(
            _prediction_record(row)
            for row in _read_csv(path / "predictions.csv", PREDICTION_FIELDS)
        )
        aucs = tuple(_auc_record(row) for row in _read_csv(path / "aucs.csv", tuple(f.name for f in fields(AucRecord))))
    except (ValueError, KeyError, TypeError, SyntaxError):
        return None
    expected_key = (draw_seed, k, arm, source, target.name)
    by_fold = {
        fold: [row for row in predictions if row.fold == fold]
        for fold in range(5)
    }
    draws_complete = True
    expected_draw_count = 0
    for fold, fold_predictions in by_fold.items():
        if not fold_predictions:
            draws_complete = False
            break
        held_out_sites = fold_predictions[0].held_out_sites
        expected_local = {
            row.case_id for row in sample_rung(target, held_out_sites, k, draw_seed)
        }
        actual_local = [
            row.case_id for row in draws
            if row.fold == fold and row.origin == "target"
        ]
        expected_source = (
            set(map(str, source_cohort.case_ids)) if arm == "warm" else set()
        )
        expected_draw_count += len(expected_local) + len(expected_source)
        actual_source = [
            row.case_id for row in draws
            if row.fold == fold and row.origin == "source"
        ]
        if (
            len(actual_local) != len(expected_local)
            or set(actual_local) != expected_local
            or len(actual_source) != len(expected_source)
            or set(actual_source) != expected_source
        ):
            draws_complete = False
            break
    draws_complete = draws_complete and len(draws) == expected_draw_count
    if (
        len(predictions) != target.n
        or {row.case_id for row in predictions} != set(map(str, target.case_ids))
        or any(
            (row.draw_seed, row.k, row.arm, row.source, row.target) != expected_key
            or not np.isfinite(row.score)
            for row in predictions
        )
        or not draws_complete
        or any(
            (row.draw_seed, row.k, row.arm, row.source, row.target) != expected_key
            for row in draws
        )
        or len(aucs) != 1
        or (
            aucs[0].draw_seed, aucs[0].k, aucs[0].arm,
            aucs[0].source, aucs[0].target,
        ) != expected_key
        or not all(
            np.isfinite(value)
            for value in (aucs[0].raw_auc, aucs[0].rank_auc, aucs[0].rank_gap)
        )
    ):
        return None
    return TraceResult(draws, predictions, aucs)


def _save_cell(path: Path, result: TraceResult) -> None:
    path.mkdir(parents=True, exist_ok=True)
    _write_csv(path / "draws.csv", DRAW_FIELDS, (asdict(row) for row in result.draws))
    _write_csv(
        path / "predictions.csv", PREDICTION_FIELDS,
        (asdict(row) for row in result.predictions),
    )
    auc_fields = tuple(f.name for f in fields(AucRecord))
    _write_csv(path / "aucs.csv", auc_fields, (asdict(row) for row in result.aucs))


def _execution_specs(profile: E1Profile, cohorts: Mapping[str, Cohort]):
    if profile.pairs is not None:
        bases = [(cohorts[source], cohorts[target], (cohorts[source],)) for source, target in profile.pairs]
    else:
        bases = []
        for target_name in sorted(profile.cohorts):
            foreign = tuple(cohorts[name] for name in cohorts if name != target_name)
            bases.extend((source, cohorts[target_name], (source,)) for source in foreign)
            bases.append((_pool_cohorts(foreign), cohorts[target_name], foreign))
    specs = []
    cold_seen = set()
    for source, target, members in bases:
        for k in profile.rungs:
            draws = (None,) if k in (0, "all") else profile.draw_ids
            for draw in draws:
                cold_key = (target.name, k, draw)
                if cold_key not in cold_seen:
                    specs.append((source, target, members, k, draw, "cold"))
                    cold_seen.add(cold_key)
                specs.append((source, target, members, k, draw, "warm"))
    return specs


def _execute_cell(spec) -> TraceResult:
    source, target, members, k, draw, arm = spec
    result = trace_paired_cell(source, target, k, draw, arms=(arm,))
    if arm == "cold":
        return TraceResult(
            tuple(replace(row, source="target-only") for row in result.draws),
            tuple(replace(row, source="target-only") for row in result.predictions),
            tuple(replace(row, source="target-only") for row in result.aucs),
        )
    return _restore_pooled_provenance(result, members) if len(members) > 1 else result


def _run_cells(out: Path, profile: E1Profile, cohorts: Mapping[str, Cohort], workers: int) -> TraceResult:
    checkpoint_root = out / "checkpoints" / "cells"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    results: dict[int, TraceResult] = {}
    missing = []
    specs = _execution_specs(profile, cohorts)
    for index, spec in enumerate(specs):
        source, target, _, k, draw, arm = spec
        path = checkpoint_root / _checkpoint_name(source.name, target.name, k, draw, arm)
        cached = _load_cell(
            path,
            source="target-only" if arm == "cold" else source.name,
            source_cohort=source,
            target=target,
            k=k,
            draw_seed=draw,
            arm=arm,
        )
        if cached is None:
            missing.append((index, spec, path))
        else:
            results[index] = cached
    phase1_endpoints = [
        item for item in missing if item[1][3] == 0 and item[1][5] == "warm"
    ]
    worker_cells = [item for item in missing if item not in phase1_endpoints]
    for index, spec, path in phase1_endpoints:
        result = _execute_cell(spec)
        _save_cell(path, result)
        results[index] = result
    with parallel_config(backend="loky", inner_max_num_threads=1):
        computed = Parallel(n_jobs=workers)(
            delayed(_execute_cell)(item[1]) for item in worker_cells
        )
    for (index, _, path), result in zip(worker_cells, computed):
        _save_cell(path, result)
        results[index] = result
    ordered = [results[index] for index in range(len(specs))]
    return TraceResult(
        tuple(row for result in ordered for row in result.draws),
        tuple(row for result in ordered for row in result.predictions),
        tuple(row for result in ordered for row in result.aucs),
    )


def _run_permutations(out, profile, cohorts, predictions, workers) -> ConfirmatoryResult:
    observed, cold_by_draw = confirmatory_observed(predictions, profile.draw_ids)
    schedules = source_label_permutations(
        cohorts["COAD"].y, "COAD", seed=0, n_permutations=profile.permutations
    )
    checkpoint_root = out / "checkpoints" / "permutations"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    null = np.empty(profile.permutations, dtype=float)
    for start in range(0, profile.permutations, profile.permutation_chunk_size):
        stop = min(start + profile.permutation_chunk_size, profile.permutations)
        path = checkpoint_root / f"{start:04d}-{stop:04d}.csv"
        cached = None
        if path.exists():
            try:
                rows = _read_csv(path, NULL_FIELDS)
                if [int(row["permutation"]) for row in rows] == list(range(start, stop)):
                    cached = np.asarray([float(row["null_mean_lift"]) for row in rows])
                    if not np.all(np.isfinite(cached)):
                        cached = None
            except (ValueError, KeyError):
                pass
        if cached is None:
            with parallel_config(backend="loky", inner_max_num_threads=1):
                cached = evaluate_confirmatory_null(
                    cohorts["COAD"], cohorts["STAD"], schedules[start:stop],
                    cold_by_draw, draw_ids=profile.draw_ids, n_jobs=workers,
                )
            _write_csv(path, NULL_FIELDS, (
                {"permutation": index, "null_mean_lift": value}
                for index, value in zip(range(start, stop), cached)
            ))
        null[start:stop] = cached
    p_value = empirical_superiority_p(observed, null)
    return ConfirmatoryResult(observed, p_value, p_value < 0.05, len(null), null)


def _base(record: AucRecord) -> str:
    return "target-only" if record.arm == "cold" else "pooled" if "+" in record.source else "single"


def _write_raw_tables(out: Path, result: TraceResult) -> None:
    _write_csv(out / "e1_draws.csv", DRAW_FIELDS, (asdict(row) for row in result.draws))
    _write_csv(out / "e1_predictions.csv", PREDICTION_FIELDS, (asdict(row) for row in result.predictions))


def _write_derived_tables(
    out: Path,
    predictions: tuple[PredictionRecord, ...],
    inference: E1Inference,
    confirmatory: ConfirmatoryResult,
) -> None:
    aucs = summarize_predictions(predictions)
    _write_csv(out / "e1_aucs.csv", AUC_FIELDS, (
        {
            "experiment": "E1", "source": row.source, "target": row.target,
            "base": _base(row), "arm": row.arm, "k": row.k,
            "draw_seed": row.draw_seed, "auc": row.raw_auc,
            "rank_auc": row.rank_auc, "rank_gap": row.rank_gap,
            "rank_diverged": row.rank_diverged,
        }
        for row in aucs
    ))
    _write_csv(out / "e1_summaries.csv", SUMMARY_FIELDS, (
        {
            "source": cell.source, "target": cell.target, "base": cell.base, "k": cell.k,
            "n_draws": cell.n_draws, "warm_auc": cell.warm.point,
            "warm_ci_lower": cell.warm.lower, "warm_ci_upper": cell.warm.upper,
            "cold_auc": cell.cold.point, "cold_ci_lower": cell.cold.lower,
            "cold_ci_upper": cell.cold.upper, "lift": cell.lift.point,
            "lift_ci_lower": cell.lift.lower, "lift_ci_upper": cell.lift.upper,
            "rank_warm_auc": cell.rank_warm, "rank_cold_auc": cell.rank_cold,
            "rank_lift": cell.rank_lift, "rank_diverged": cell.rank_diverged,
            "confirmatory": cell.confirmatory,
            "permutation_p": confirmatory.p_value if cell.confirmatory else "",
        }
        for cell in inference.cells
    ))
    _write_csv(out / "e1_equivalence.csv", EQUIVALENCE_FIELDS, (
        {
            "source": cell.source, "target": cell.target, "base": cell.base,
            "local_positive_equivalence": cell.local_positive.point.value,
            "local_ci_lower": cell.local_positive.interval.lower,
            "local_ci_upper": cell.local_positive.interval.upper,
            "local_point_censored": cell.local_positive.point.censored,
            "local_ci_lower_censored": cell.local_positive.interval.lower_censored,
            "local_ci_upper_censored": cell.local_positive.interval.upper_censored,
            "average_source_case_equivalence": cell.local_positive.average_source_case.point.value,
            "average_ci_lower": cell.local_positive.average_source_case.interval.lower,
            "average_ci_upper": cell.local_positive.average_source_case.interval.upper,
            "average_point_censored": cell.local_positive.average_source_case.point.censored,
            "average_ci_lower_censored": cell.local_positive.average_source_case.interval.lower_censored,
            "average_ci_upper_censored": cell.local_positive.average_source_case.interval.upper_censored,
        }
        for cell in inference.equivalences
    ))
    _write_csv(out / "e1_confirmatory_null.csv", NULL_FIELDS, (
        {"permutation": index, "null_mean_lift": value}
        for index, value in enumerate(confirmatory.null_lifts)
    ))


def _derive_reports(out: Path) -> tuple[tuple[PredictionRecord, ...], E1Inference, ConfirmatoryResult]:
    manifest = json.loads((out / "manifest.json").read_text())
    predictions = tuple(
        _prediction_record(row)
        for row in _read_csv(out / "e1_predictions.csv", PREDICTION_FIELDS)
    )
    draw_ids = tuple(int(value) for value in manifest["configuration"]["draw_ids"])
    source_names = {row.source for row in predictions if row.arm == "warm"}
    source_counts = {
        source: sum(int(manifest["cohorts"][member]["cases"]) for member in source.split("+"))
        for source in source_names
    }
    inference = estimate_e1_matrix(
        predictions,
        source_counts,
        draw_ids=draw_ids,
        n_bootstraps=int(manifest["configuration"]["bootstrap_replicates"]),
    )
    null_rows = _read_csv(out / "e1_confirmatory_null.csv", NULL_FIELDS)
    null = np.asarray([float(row["null_mean_lift"]) for row in null_rows])
    observed, _ = confirmatory_observed(predictions, draw_ids)
    p_value = empirical_superiority_p(observed, null)
    confirmatory = ConfirmatoryResult(
        observed, p_value, p_value < 0.05, len(null), null
    )
    return predictions, inference, confirmatory


def rebuild_e1_reports(out: Path) -> None:
    """Rebuild all derived CSV reports from stored predictions and null values."""
    predictions, inference, confirmatory = _derive_reports(out)
    _write_derived_tables(out, predictions, inference, confirmatory)


def validate_e1_bundle(out: Path) -> None:
    """Validate public schemas and material joins in a completed E1 bundle."""
    manifest_path = out / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("missing bundle artifact: manifest.json")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        raise ValueError("unsupported E1 bundle schema version")
    draws = _read_csv(out / "e1_draws.csv", DRAW_FIELDS)
    predictions = _read_csv(out / "e1_predictions.csv", PREDICTION_FIELDS)
    aucs = _read_csv(out / "e1_aucs.csv", AUC_FIELDS)
    summaries = _read_csv(out / "e1_summaries.csv", SUMMARY_FIELDS)
    _read_csv(out / "e1_equivalence.csv", EQUIVALENCE_FIELDS)
    null = _read_csv(out / "e1_confirmatory_null.csv", NULL_FIELDS)
    prediction_keys = {
        (row["source"], row["target"], row["arm"], row["k"], row["draw_seed"])
        for row in predictions
    }
    auc_keys = {
        (row["source"], row["target"], row["arm"], row["k"], row["draw_seed"])
        for row in aucs
    }
    if prediction_keys != auc_keys:
        raise ValueError("prediction-to-AUC join is incomplete")
    draw_keys = {
        (row["source"], row["target"], row["arm"], row["k"], row["draw_seed"])
        for row in draws
    }
    if not draw_keys <= prediction_keys:
        raise ValueError("draw-to-prediction join is incomplete")
    prediction_patients: dict[tuple[str, str, str, str, str], set[str]] = {}
    for row in predictions:
        key = (row["source"], row["target"], row["arm"], row["k"], row["draw_seed"])
        patients = prediction_patients.setdefault(key, set())
        if row["case_id"] in patients:
            raise ValueError("prediction grain contains a duplicate patient")
        patients.add(row["case_id"])
    warm_cells = {(row["source"], row["target"], row["k"]) for row in aucs if row["arm"] == "warm"}
    summary_cells = {(row["source"], row["target"], row["k"]) for row in summaries}
    if warm_cells != summary_cells:
        raise ValueError("AUC-to-summary join is incomplete")
    expected_permutations = int(manifest["configuration"]["permutations"])
    if [int(row["permutation"]) for row in null] != list(range(expected_permutations)):
        raise ValueError("confirmatory null is incomplete or out of order")
    if not draws:
        raise ValueError("draw audit table is empty")
    predictions_records, inference, confirmatory = _derive_reports(out)
    with tempfile.TemporaryDirectory(prefix="panmorph-e1-validate-") as temporary:
        expected = Path(temporary)
        _write_derived_tables(
            expected, predictions_records, inference, confirmatory
        )
        for name in (
            "e1_aucs.csv", "e1_summaries.csv", "e1_equivalence.csv",
            "e1_confirmatory_null.csv",
        ):
            if (out / name).read_bytes() != (expected / name).read_bytes():
                raise ValueError(f"stored {name} is not reproducible from bundle inputs")


def render_e1_figures(out: Path) -> None:
    """Render value and lift figures using only a validated on-disk bundle."""
    validate_e1_bundle(out)
    rows = _read_csv(out / "e1_summaries.csv", SUMMARY_FIELDS)
    numeric = lambda value: 1e9 if value == "all" else float(value)
    ordered = sorted(rows, key=lambda row: (row["target"], row["source"], numeric(row["k"])))
    labels = [f'{row["source"]}→{row["target"]}\nk={row["k"]}' for row in ordered]
    divergent = [row["rank_diverged"] == "True" for row in ordered]
    x = np.arange(len(ordered))
    for kind, columns, ylabel, stem in (
        ("value", ("warm_auc", "cold_auc"), "Raw pooled OOF AUC", "e1_value"),
        ("lift", ("lift",), "Raw pooled OOF AUC lift", "e1_lift"),
    ):
        fig, ax = plt.subplots(figsize=(max(7, len(ordered) * 0.8), 4.8))
        if kind == "value":
            ax.plot(x, [float(row[columns[0]]) for row in ordered], "o-", label="warm")
            ax.plot(x, [float(row[columns[1]]) for row in ordered], "o-", label="cold")
            ax.legend()
        else:
            ax.axhline(0, color="0.5", linewidth=0.8)
            ax.errorbar(
                x, [float(row["lift"]) for row in ordered],
                yerr=[
                    [float(row["lift"]) - float(row["lift_ci_lower"]) for row in ordered],
                    [float(row["lift_ci_upper"]) - float(row["lift"]) for row in ordered],
                ], fmt="o",
            )
        for index, flag in enumerate(divergent):
            if flag:
                ax.annotate("rank divergence", (index, ax.get_ylim()[1]), rotation=90,
                            va="top", ha="center", fontsize=7, color="darkred")
        ax.set_xticks(x, labels, rotation=45, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(f"E1 {kind} ({'REPORTABLE' if json.loads((out / 'manifest.json').read_text())['reportable'] else 'QUICK — NON-REPORTABLE'})")
        fig.tight_layout()
        fig.savefig(out / f"{stem}.png", dpi=300)
        fig.savefig(out / f"{stem}.pdf")
        plt.close(fig)


def run_e1_bundle(
    out: Path,
    *,
    profile: Literal["quick", "full"] = "full",
    cohorts: Mapping[str, Cohort] | None = None,
    workers: int = DEFAULT_WORKERS,
) -> Path:
    """Execute or safely resume E1 and return its validated bundle directory."""
    if workers < 1:
        raise ValueError("workers must be positive")
    selected = PROFILES[profile]
    loaded = (
        {name: load_cohort(name) for name in selected.cohorts}
        if cohorts is None else dict(cohorts)
    )
    missing = set(selected.cohorts) - loaded.keys()
    if missing:
        raise ValueError(f"missing configured cohorts: {sorted(missing)}")
    loaded = {name: loaded[name] for name in selected.cohorts}
    for target in loaded.values():
        preflight_rungs(target, selected.rungs)
    planned = _manifest(selected, loaded, workers)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if _identity(existing) != _identity(planned):
            raise ValueError("incompatible E1 partial results: manifest identity differs")
    started = time.monotonic()
    _write_json(manifest_path, planned)
    result = _run_cells(out, selected, loaded, workers)
    if selected.reportable:
        with (ROOT / "results/gate_results.csv").open(newline="") as handle:
            validate_phase1_anchors(result.aucs, tuple(csv.DictReader(handle)))
    _write_raw_tables(out, result)
    confirmatory = _run_permutations(out, selected, loaded, result.predictions, workers)
    _write_csv(out / "e1_confirmatory_null.csv", NULL_FIELDS, (
        {"permutation": index, "null_mean_lift": value}
        for index, value in enumerate(confirmatory.null_lifts)
    ))
    rebuild_e1_reports(out)
    render_e1_figures(out)
    completed = dict(planned)
    completed["status"] = "complete"
    completed["elapsed_seconds"] = time.monotonic() - started
    _write_json(manifest_path, completed)
    return out
