"""Execution and artifact boundary for the exploratory budget-matched swap."""
from __future__ import annotations

import ast
import json
from collections import defaultdict
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Mapping

import matplotlib
import numpy as np
from joblib import Parallel, delayed, parallel_config
from sklearn.isotonic import IsotonicRegression

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .data import Cohort, load_cohort
from .e1 import (
    BOOTSTRAP_REPLICATES,
    CensoredInterval,
    IntervalEstimate,
    PredictionRecord,
    Rung,
    _bootstrap_auc,
    _ordered_draw_scores,
    _percentile_ranks,
    estimate_e1_cell,
    rank_auc_diverged,
    stratified_bootstrap_indices,
    summarize_predictions,
)
from .e1_runner import (
    COHORT_CASE_FIELDS,
    DEFAULT_WORKERS,
    FOLD_FIELDS,
    _bundle_prediction_records,
    _cohort_identity,
    _read_csv,
    _write_csv,
    _write_json,
    validate_e1_bundle,
)
from .swap import (
    SWAP_BUDGETS,
    SWAP_DRAW_IDS,
    SWAP_TARGET_SHARES,
    ConditionalEquivalence,
    SwapAucRecord,
    SwapDrawRecord,
    SwapPredictionRecord,
    TargetReferencePoint,
    build_target_reference,
    conditional_equivalence_interval,
    conditional_source_case_equivalence,
    e1_case_coordinate,
    mixture_case_counts,
    prevalence_matched_counts,
    trace_swap_cell,
)

SWAP_ARTIFACTS = (
    "swap_draws.csv", "swap_predictions.csv", "swap_aucs.csv",
    "swap_summaries.csv", "swap_reference.csv", "swap_equivalence.csv",
    "swap_auc.png", "swap_auc.pdf", "swap_equivalence.png",
    "swap_equivalence.pdf",
)
SWAP_DRAW_FIELDS = tuple(field.name for field in fields(SwapDrawRecord))
SWAP_PREDICTION_FIELDS = tuple(field.name for field in fields(SwapPredictionRecord))
SWAP_AUC_FIELDS = tuple(field.name for field in fields(SwapAucRecord))
SWAP_SUMMARY_FIELDS = (
    "budget", "target_share", "source_cases", "target_cases", "n_draws",
    "raw_auc", "raw_ci_lower", "raw_ci_upper", "rank_auc", "rank_ci_lower",
    "rank_ci_upper", "rank_gap", "rank_diverged",
)
SWAP_REFERENCE_FIELDS = ("cases", "origin", "raw_auc", "monotone_auc")
SWAP_EQUIVALENCE_FIELDS = (
    "budget", "target_share", "source_cases", "target_cases", "defined",
    "average_source_case_equivalence", "censored", "censor_at",
    "ci_lower", "ci_upper", "ci_lower_censored", "ci_upper_censored",
)


@dataclass(frozen=True)
class SwapCellEstimate:
    budget: int
    target_share: int
    source_cases: int
    target_cases: int
    n_draws: int
    raw: IntervalEstimate
    rank: IntervalEstimate
    rank_gap: float
    rank_diverged: bool
    bootstrap_raw: np.ndarray
    bootstrap_rank: np.ndarray


@dataclass(frozen=True)
class SwapEquivalenceEstimate:
    budget: int
    target_share: int
    source_cases: int
    target_cases: int
    point: ConditionalEquivalence
    interval: CensoredInterval | None


@dataclass(frozen=True)
class SwapBundleResult:
    draws: tuple[SwapDrawRecord, ...]
    predictions: tuple[SwapPredictionRecord, ...]
    aucs: tuple[SwapAucRecord, ...]


def estimate_swap_cells(
    predictions: tuple[SwapPredictionRecord, ...],
    aucs: tuple[SwapAucRecord, ...],
    *,
    draw_ids: tuple[int, ...],
    n_bootstraps: int,
) -> tuple[SwapCellEstimate, ...]:
    estimates = []
    keys = sorted({(row.budget, row.target_share) for row in aucs})
    for budget, target_share in keys:
        cell_predictions = [
            row for row in predictions
            if (row.budget, row.target_share) == (budget, target_share)
        ]
        cell_aucs = [
            row for row in aucs
            if (row.budget, row.target_share) == (budget, target_share)
        ]
        if {row.draw_seed for row in cell_aucs} != set(draw_ids):
            raise ValueError("swap cells require the complete fixed draw schedule")
        case_ids = tuple(sorted({row.case_id for row in cell_predictions}))
        score_draws = []
        rank_score_draws = []
        labels = None
        for draw in draw_ids:
            generic = [
                PredictionRecord(
                    row.draw_seed, row.budget, row.fold, row.held_out_sites,
                    "warm", "COAD+STAD", "STAD", row.case_id, row.label,
                    row.score,
                )
                for row in cell_predictions if row.draw_seed == draw
            ]
            draw_labels, scores = _ordered_draw_scores(generic, case_ids)
            if labels is not None and not np.array_equal(labels, draw_labels):
                raise ValueError("swap patient labels must be fixed across draws")
            labels = draw_labels
            score_draws.append(scores)
            fold_by_case = {row.case_id: row.fold for row in generic}
            ordered_folds = np.asarray([fold_by_case[case_id] for case_id in case_ids])
            ranked = np.empty(len(scores), dtype=float)
            for fold in np.unique(ordered_folds):
                in_fold = ordered_folds == fold
                ranked[in_fold] = _percentile_ranks(scores[in_fold])
            rank_score_draws.append(ranked)
        assert labels is not None
        schedule = stratified_bootstrap_indices(
            labels, seed=0, key=("STAD",), n_replicates=n_bootstraps
        )
        bootstrap = np.mean(
            [_bootstrap_auc(labels, scores, schedule) for scores in score_draws], axis=0
        )
        bootstrap_rank = np.mean(
            [_bootstrap_auc(labels, scores, schedule) for scores in rank_score_draws], axis=0
        )
        point = float(np.mean([row.raw_auc for row in cell_aucs]))
        lower, upper = np.percentile(bootstrap, (2.5, 97.5))
        rank = float(np.mean([row.rank_auc for row in cell_aucs]))
        rank_lower, rank_upper = np.percentile(bootstrap_rank, (2.5, 97.5))
        source_cases, target_cases = mixture_case_counts(budget, target_share)
        estimates.append(
            SwapCellEstimate(
                budget, target_share, source_cases, target_cases, len(cell_aucs),
                IntervalEstimate(point, float(lower), float(upper)),
                IntervalEstimate(rank, float(rank_lower), float(rank_upper)),
                abs(point - rank), rank_auc_diverged(point, rank), bootstrap,
                bootstrap_rank,
            )
        )
    return tuple(estimates)


def _all_case_coordinate(predictions: tuple[PredictionRecord, ...]) -> float:
    rows = [
        row for row in predictions
        if (row.source, row.target, row.k, row.arm)
        == ("target-only", "STAD", "all", "cold")
    ]
    if not rows:
        raise ValueError("completed E1 must include the STAD cold all endpoint")
    return float(np.mean([
        len(rows) - sum(row.fold == fold for row in rows)
        for fold in sorted({row.fold for row in rows})
    ]))


def _reference_and_equivalence(
    e1_predictions: tuple[PredictionRecord, ...],
    estimates: tuple[SwapCellEstimate, ...],
    *,
    target_prevalence: float,
    e1_rungs: tuple[Rung, ...],
    draw_ids: tuple[int, ...],
    n_bootstraps: int,
) -> tuple[
    tuple[TargetReferencePoint, ...], np.ndarray,
    tuple[SwapEquivalenceEstimate, ...],
]:
    e1_cells = {
        rung: estimate_e1_cell(
            e1_predictions, "COAD", "STAD", rung, draw_ids=draw_ids,
            n_bootstraps=n_bootstraps,
        )
        for rung in e1_rungs
    }
    target_only = {cell.budget: cell for cell in estimates if cell.target_share == 100}
    points = build_target_reference(
        e1_cold={rung: cell.cold.point for rung, cell in e1_cells.items()},
        target_prevalence=target_prevalence,
        all_case_coordinate=_all_case_coordinate(e1_predictions),
        swap_target_only={budget: cell.raw.point for budget, cell in target_only.items()},
    )
    bootstrap_by_key: dict[tuple[float, str], np.ndarray] = {}
    for rung, cell in e1_cells.items():
        cases = (
            _all_case_coordinate(e1_predictions)
            if rung == "all"
            else e1_case_coordinate(rung, target_prevalence)
        )
        bootstrap_by_key[(cases, "e1")] = cell.bootstrap_cold
    for budget, cell in target_only.items():
        bootstrap_by_key[(float(budget), "swap")] = cell.bootstrap_raw
    point_curve = {point.cases: point.auc for point in points}
    bootstrap_curves = [
        {
            point.cases: float(bootstrap_by_key[(point.cases, point.origin)][index])
            for point in points
        }
        for index in range(n_bootstraps)
    ]
    equivalences = []
    for cell in estimates:
        point = conditional_source_case_equivalence(
            cell.raw.point, cell.target_cases, cell.source_cases, point_curve
        )
        interval = None
        if point.defined:
            bootstrap = tuple(
                conditional_source_case_equivalence(
                    float(cell.bootstrap_raw[index]), cell.target_cases,
                    cell.source_cases, bootstrap_curves[index],
                )
                for index in range(n_bootstraps)
            )
            interval = conditional_equivalence_interval(bootstrap)
        equivalences.append(
            SwapEquivalenceEstimate(
                cell.budget, cell.target_share, cell.source_cases,
                cell.target_cases, point, interval,
            )
        )
    raw = np.asarray([point.auc for point in points])
    monotone = IsotonicRegression(increasing=True).fit_transform(
        np.asarray([point.cases for point in points]), raw,
        sample_weight=np.ones(len(points)),
    )
    return points, monotone, tuple(equivalences)


def _render_swap_figures(
    out: Path,
    estimates: tuple[SwapCellEstimate, ...],
    equivalences: tuple[SwapEquivalenceEstimate, ...],
) -> None:
    label = "EXPLORATORY"
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for budget in sorted({cell.budget for cell in estimates}):
        cells = [cell for cell in estimates if cell.budget == budget]
        ax.plot(
            [cell.target_share for cell in cells], [cell.raw.point for cell in cells],
            marker="o", label=f"N={budget}",
        )
        ax.fill_between(
            [cell.target_share for cell in cells],
            [cell.raw.lower for cell in cells], [cell.raw.upper for cell in cells],
            alpha=0.15,
        )
    ax.set(xlabel="STAD share (%)", ylabel="Pooled STAD OOF AUC",
           title=f"Budget-matched COAD→STAD swap ({label})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "swap_auc.png", dpi=180)
    fig.savefig(out / "swap_auc.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for budget in sorted({cell.budget for cell in equivalences}):
        cells = [cell for cell in equivalences if cell.budget == budget and cell.point.defined]
        y = [cell.point.censor_at if cell.point.censored else cell.point.value for cell in cells]
        ax.plot([cell.target_share for cell in cells], y, marker="o", label=f"N={budget}")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(xlabel="STAD share (%)", ylabel="Equivalent STAD cases / COAD case",
           title=f"Conditional average source-case equivalence ({label})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "swap_equivalence.png", dpi=180)
    fig.savefig(out / "swap_equivalence.pdf")
    plt.close(fig)


def _write_swap_tables(
    out: Path,
    result: SwapBundleResult,
    estimates: tuple[SwapCellEstimate, ...],
    points: tuple[TargetReferencePoint, ...],
    monotone: np.ndarray,
    equivalences: tuple[SwapEquivalenceEstimate, ...],
) -> None:
    _write_csv(out / "swap_draws.csv", SWAP_DRAW_FIELDS, map(asdict, result.draws))
    _write_csv(
        out / "swap_predictions.csv", SWAP_PREDICTION_FIELDS,
        ({**asdict(row), "held_out_sites": repr(row.held_out_sites)} for row in result.predictions),
    )
    _write_csv(out / "swap_aucs.csv", SWAP_AUC_FIELDS, map(asdict, result.aucs))
    _write_csv(out / "swap_summaries.csv", SWAP_SUMMARY_FIELDS, (
        {
            "budget": cell.budget, "target_share": cell.target_share,
            "source_cases": cell.source_cases, "target_cases": cell.target_cases,
            "n_draws": cell.n_draws, "raw_auc": cell.raw.point,
            "raw_ci_lower": cell.raw.lower, "raw_ci_upper": cell.raw.upper,
            "rank_auc": cell.rank.point, "rank_ci_lower": cell.rank.lower,
            "rank_ci_upper": cell.rank.upper, "rank_gap": cell.rank_gap,
            "rank_diverged": cell.rank_diverged,
        }
        for cell in estimates
    ))
    _write_csv(out / "swap_reference.csv", SWAP_REFERENCE_FIELDS, (
        {"cases": point.cases, "origin": point.origin, "raw_auc": point.auc,
         "monotone_auc": fitted}
        for point, fitted in zip(points, monotone)
    ))
    _write_csv(out / "swap_equivalence.csv", SWAP_EQUIVALENCE_FIELDS, (
        {
            "budget": cell.budget, "target_share": cell.target_share,
            "source_cases": cell.source_cases, "target_cases": cell.target_cases,
            "defined": cell.point.defined,
            "average_source_case_equivalence": cell.point.value,
            "censored": cell.point.censored, "censor_at": cell.point.censor_at,
            "ci_lower": None if cell.interval is None else cell.interval.lower,
            "ci_upper": None if cell.interval is None else cell.interval.upper,
            "ci_lower_censored": None if cell.interval is None else cell.interval.lower_censored,
            "ci_upper_censored": None if cell.interval is None else cell.interval.upper_censored,
        }
        for cell in equivalences
    ))


def run_swap_bundle(
    out: Path,
    *,
    cohorts: Mapping[str, Cohort] | None = None,
    budgets: tuple[int, ...] = SWAP_BUDGETS,
    target_shares: tuple[int, ...] = SWAP_TARGET_SHARES,
    draw_ids: tuple[int, ...] = SWAP_DRAW_IDS,
    n_bootstraps: int = BOOTSTRAP_REPLICATES,
    workers: int = DEFAULT_WORKERS,
) -> None:
    """Add the exploratory swap records and reports to a completed E1 bundle."""
    validate_e1_bundle(out, require_complete=True)
    loaded = dict(cohorts) if cohorts is not None else {
        name: load_cohort(name) for name in ("COAD", "STAD")
    }
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for name in ("COAD", "STAD"):
        if _cohort_identity(loaded[name]) != manifest["cohorts"][name]:
            raise ValueError(f"swap {name} data does not match the completed E1 bundle")
    configuration = {
        "status": "running", "source": "COAD", "target": "STAD",
        "budgets": list(budgets), "target_shares": list(target_shares),
        "draw_ids": list(draw_ids), "bootstrap_replicates": n_bootstraps,
        "confirmatory": False, "permutations": 0, "exploratory": True,
    }
    manifest.setdefault("downstream", {})["swap"] = configuration
    _write_json(manifest_path, manifest)
    specs = [
        (budget, share, draw)
        for budget in budgets for share in target_shares for draw in draw_ids
    ]
    with parallel_config(backend="loky", inner_max_num_threads=1):
        cells = Parallel(n_jobs=workers)(
            delayed(trace_swap_cell)(
                loaded["COAD"], loaded["STAD"], budget=budget,
                target_share=share, draw_seed=draw,
            )
            for budget, share, draw in specs
        )
    result = SwapBundleResult(
        tuple(row for cell in cells for row in cell.draws),
        tuple(row for cell in cells for row in cell.predictions),
        tuple(cell.auc for cell in cells),
    )
    estimates = estimate_swap_cells(
        result.predictions, result.aucs, draw_ids=draw_ids,
        n_bootstraps=n_bootstraps,
    )
    e1_predictions = _bundle_prediction_records(out)
    e1_rungs = tuple(
        "all" if rung == "all" else int(rung)
        for rung in manifest["configuration"]["rungs"]
    )
    points, monotone, equivalences = _reference_and_equivalence(
        e1_predictions, estimates,
        target_prevalence=loaded["STAD"].prevalence,
        e1_rungs=e1_rungs,
        draw_ids=tuple(int(draw) for draw in manifest["configuration"]["draw_ids"]),
        n_bootstraps=n_bootstraps,
    )
    _write_swap_tables(out, result, estimates, points, monotone, equivalences)
    _render_swap_figures(out, estimates, equivalences)
    configuration["status"] = "complete"
    _write_json(manifest_path, manifest)
    _validate_swap_bundle(out, validate_parent=False)


def _swap_prediction(row: Mapping[str, str]) -> SwapPredictionRecord:
    return SwapPredictionRecord(
        int(row["budget"]), int(row["target_share"]), int(row["draw_seed"]),
        int(row["fold"]), tuple(ast.literal_eval(row["held_out_sites"])),
        row["case_id"], int(row["label"]), float(row["score"]),
    )


def _swap_auc(row: Mapping[str, str]) -> SwapAucRecord:
    return SwapAucRecord(
        int(row["budget"]), int(row["target_share"]), int(row["draw_seed"]),
        float(row["raw_auc"]), float(row["rank_auc"]), float(row["rank_gap"]),
        row["rank_diverged"] == "True",
    )


def _same_float(actual: str, expected: float | None) -> bool:
    if expected is None:
        return actual == ""
    try:
        return bool(np.isclose(float(actual), expected, rtol=0, atol=1e-12))
    except ValueError:
        return False


def _validate_derived_swap_tables(
    out: Path,
    manifest: Mapping[str, object],
    prediction_rows: tuple[dict[str, str], ...],
    auc_rows: tuple[dict[str, str], ...],
    summary_rows: tuple[dict[str, str], ...],
    reference_rows: tuple[dict[str, str], ...],
    equivalence_rows: tuple[dict[str, str], ...],
) -> None:
    config = manifest["downstream"]["swap"]
    predictions = tuple(_swap_prediction(row) for row in prediction_rows)
    stored_aucs = tuple(_swap_auc(row) for row in auc_rows)
    derived_aucs = []
    for stored in stored_aucs:
        rows = [
            row for row in predictions
            if (row.budget, row.target_share, row.draw_seed)
            == (stored.budget, stored.target_share, stored.draw_seed)
        ]
        generic = tuple(
            PredictionRecord(
                row.draw_seed, row.budget, row.fold, row.held_out_sites, "warm",
                f"swap-{row.target_share}", "STAD", row.case_id, row.label,
                row.score,
            )
            for row in rows
        )
        (summary,) = summarize_predictions(generic)
        derived = SwapAucRecord(
            stored.budget, stored.target_share, stored.draw_seed,
            summary.raw_auc, summary.rank_auc, summary.rank_gap,
            summary.rank_diverged,
        )
        if (
            not _same_float(str(stored.raw_auc), derived.raw_auc)
            or not _same_float(str(stored.rank_auc), derived.rank_auc)
            or not _same_float(str(stored.rank_gap), derived.rank_gap)
            or stored.rank_diverged != derived.rank_diverged
        ):
            raise ValueError("swap AUC table is not reproducible from predictions")
        derived_aucs.append(derived)
    estimates = estimate_swap_cells(
        predictions, tuple(derived_aucs),
        draw_ids=tuple(int(value) for value in config["draw_ids"]),
        n_bootstraps=int(config["bootstrap_replicates"]),
    )
    expected_summaries = {
        (str(cell.budget), str(cell.target_share)): {
            "source_cases": str(cell.source_cases), "target_cases": str(cell.target_cases),
            "n_draws": str(cell.n_draws), "raw_auc": cell.raw.point,
            "raw_ci_lower": cell.raw.lower, "raw_ci_upper": cell.raw.upper,
            "rank_auc": cell.rank.point, "rank_ci_lower": cell.rank.lower,
            "rank_ci_upper": cell.rank.upper, "rank_gap": cell.rank_gap,
            "rank_diverged": str(cell.rank_diverged),
        }
        for cell in estimates
    }
    for row in summary_rows:
        expected = expected_summaries[(row["budget"], row["target_share"])]
        if any(
            row[field] != value if isinstance(value, str) else not _same_float(row[field], value)
            for field, value in expected.items()
        ):
            raise ValueError("swap summary table is not reproducible from predictions")

    e1_predictions = _bundle_prediction_records(out)
    e1_rungs = tuple(
        "all" if rung == "all" else int(rung)
        for rung in manifest["configuration"]["rungs"]
    )
    points, monotone, equivalences = _reference_and_equivalence(
        e1_predictions, estimates,
        target_prevalence=(
            int(manifest["cohorts"]["STAD"]["positives"])
            / int(manifest["cohorts"]["STAD"]["cases"])
        ),
        e1_rungs=e1_rungs,
        draw_ids=tuple(int(value) for value in manifest["configuration"]["draw_ids"]),
        n_bootstraps=int(config["bootstrap_replicates"]),
    )
    if len(reference_rows) != len(points) or any(
        row["origin"] != point.origin
        or not _same_float(row["cases"], point.cases)
        or not _same_float(row["raw_auc"], point.auc)
        or not _same_float(row["monotone_auc"], float(fitted))
        for row, point, fitted in zip(reference_rows, points, monotone)
    ):
        raise ValueError("swap reference table is not reproducible from predictions")
    expected_equivalence = {
        (str(cell.budget), str(cell.target_share)): cell for cell in equivalences
    }
    for row in equivalence_rows:
        cell = expected_equivalence[(row["budget"], row["target_share"])]
        interval = cell.interval
        if (
            row["defined"] != str(cell.point.defined)
            or row["censored"] != str(cell.point.censored)
            or not _same_float(row["average_source_case_equivalence"], cell.point.value)
            or not _same_float(row["censor_at"], cell.point.censor_at)
            or not _same_float(row["ci_lower"], None if interval is None else interval.lower)
            or not _same_float(row["ci_upper"], None if interval is None else interval.upper)
            or row["ci_lower_censored"]
            != ("" if interval is None else str(interval.lower_censored))
            or row["ci_upper_censored"]
            != ("" if interval is None else str(interval.upper_censored))
        ):
            raise ValueError("swap equivalence table is not reproducible from predictions")


def _validate_swap_bundle(out: Path, *, validate_parent: bool) -> None:
    if validate_parent:
        validate_e1_bundle(out, require_complete=True)
    manifest = json.loads((out / "manifest.json").read_text())
    config = manifest.get("downstream", {}).get("swap")
    if not config or config.get("status") != "complete":
        raise ValueError("swap bundle is not complete")
    if (
        config.get("exploratory") is not True
        or config.get("confirmatory") is not False
        or config.get("permutations") != 0
    ):
        raise ValueError("swap bundle must remain exploratory without permutations")
    missing = [name for name in SWAP_ARTIFACTS if not (out / name).is_file()]
    if missing:
        raise ValueError(f"missing swap bundle artifacts: {missing}")
    draws = _read_csv(out / "swap_draws.csv", SWAP_DRAW_FIELDS)
    predictions = _read_csv(out / "swap_predictions.csv", SWAP_PREDICTION_FIELDS)
    aucs = _read_csv(out / "swap_aucs.csv", SWAP_AUC_FIELDS)
    summaries = _read_csv(out / "swap_summaries.csv", SWAP_SUMMARY_FIELDS)
    reference = _read_csv(out / "swap_reference.csv", SWAP_REFERENCE_FIELDS)
    equivalence = _read_csv(out / "swap_equivalence.csv", SWAP_EQUIVALENCE_FIELDS)
    cohort_rows = _read_csv(out / "e1_cohort_cases.csv", COHORT_CASE_FIELDS)
    fold_rows = _read_csv(out / "e1_folds.csv", FOLD_FIELDS)
    expected = {
        (str(budget), str(share), str(draw))
        for budget in config["budgets"] for share in config["target_shares"]
        for draw in config["draw_ids"]
    }
    auc_keys = {(row["budget"], row["target_share"], row["draw_seed"]) for row in aucs}
    if auc_keys != expected or len(aucs) != len(expected):
        raise ValueError("swap AUC cells do not match the configured grid")
    expected_cells = {(key[0], key[1]) for key in expected}
    if (
        {(row["budget"], row["target_share"]) for row in summaries} != expected_cells
        or len(summaries) != len(expected_cells)
        or {(row["budget"], row["target_share"]) for row in equivalence} != expected_cells
        or len(equivalence) != len(expected_cells)
    ):
        raise ValueError("swap summaries do not match the configured grid")
    target_cases = int(manifest["cohorts"]["STAD"]["cases"])
    cohort_cases = {
        (row["cohort"], row["case_id"]): (row["label"], row["site"])
        for row in cohort_rows if row["cohort"] in ("COAD", "STAD")
    }
    held_out_by_fold = {
        int(row["fold"]): tuple(ast.literal_eval(row["held_out_sites"]))
        for row in fold_rows if row["target"] == "STAD"
    }
    by_cell: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in predictions:
        by_cell[(row["budget"], row["target_share"], row["draw_seed"])].append(row)
    for key in expected:
        rows = by_cell[key]
        if len(rows) != target_cases or len({row["case_id"] for row in rows}) != target_cases:
            raise ValueError("swap prediction cell lacks complete target OOF coverage")
        for row in rows:
            metadata = cohort_cases.get(("STAD", row["case_id"]))
            held_out = tuple(ast.literal_eval(row["held_out_sites"]))
            if (
                metadata is None or row["label"] != metadata[0]
                or held_out != held_out_by_fold.get(int(row["fold"]))
                or metadata[1] not in held_out
                or not np.isfinite(float(row["score"]))
            ):
                raise ValueError("swap predictions violate the E1 STAD fold audit")
    by_draw_fold: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in draws:
        by_draw_fold[(row["budget"], row["target_share"], row["draw_seed"], row["fold"])].append(row)
    for budget, share, draw in expected:
        for fold in range(int(manifest["splitter"]["folds"])):
            rows = by_draw_fold[(budget, share, draw, str(fold))]
            if len(rows) != int(budget):
                raise ValueError("swap draw does not hold the total assay budget fixed")
            n_source, n_target = mixture_case_counts(int(budget), int(share))
            portions = {"source": n_source, "target": n_target}
            if any(sum(row["origin"] == origin for row in rows) != count
                   for origin, count in portions.items()):
                raise ValueError("swap draw does not match its configured mixture")
            held_out = held_out_by_fold[fold]
            seen = set()
            for row in rows:
                key = (row["origin"], row["case_id"])
                cohort = "COAD" if row["origin"] == "source" else "STAD"
                metadata = cohort_cases.get((cohort, row["case_id"]))
                if key in seen or metadata != (row["label"], row["site"]):
                    raise ValueError("swap draw membership is not a unique cohort case")
                if row["origin"] == "target" and row["site"] in held_out:
                    raise ValueError("swap target draw includes a held-out site")
                seen.add(key)
            for origin, count in portions.items():
                cohort = "COAD" if origin == "source" else "STAD"
                identity = manifest["cohorts"][cohort]
                expected_positive, _ = prevalence_matched_counts(
                    count, int(identity["positives"]) / int(identity["cases"])
                )
                if sum(row["origin"] == origin and row["label"] == "1" for row in rows) != expected_positive:
                    raise ValueError("swap draw does not match full-cohort prevalence")
    prefix_groups: dict[tuple[str, str, str], list[tuple[int, set[str]]]] = defaultdict(list)
    for (budget, share, draw, fold), rows in by_draw_fold.items():
        for origin in ("source", "target"):
            cases = {row["case_id"] for row in rows if row["origin"] == origin}
            prefix_groups[(draw, fold, origin)].append((len(cases), cases))
    for prefixes in prefix_groups.values():
        ordered = sorted(prefixes, key=lambda value: value[0])
        for (_, lower), (_, upper) in zip(ordered, ordered[1:]):
            if not lower <= upper:
                raise ValueError("swap draws are not nested stratified prefixes")
    if any(
        (row["target_share"] == "100") != (row["defined"] == "False")
        for row in equivalence
    ):
        raise ValueError("only target-only swap equivalence may be undefined")
    fitted = [float(row["monotone_auc"]) for row in reference]
    if any(lower > upper for lower, upper in zip(fitted, fitted[1:])):
        raise ValueError("swap target-only reference is not monotone")
    for name in ("swap_auc.png", "swap_equivalence.png"):
        if not (out / name).read_bytes().startswith(b"\x89PNG"):
            raise ValueError(f"invalid swap figure: {name}")
    for name in ("swap_auc.pdf", "swap_equivalence.pdf"):
        if not (out / name).read_bytes().startswith(b"%PDF"):
            raise ValueError(f"invalid swap figure: {name}")
    _validate_derived_swap_tables(
        out, manifest, predictions, aucs, summaries, reference, equivalence
    )


def validate_swap_bundle(out: Path) -> None:
    """Reject incomplete or internally inconsistent E1+swap artifacts."""
    _validate_swap_bundle(out, validate_parent=True)
