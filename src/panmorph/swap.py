"""Budget-matched COAD/STAD swap experiment on the E1 OOF scaffold."""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Iterable, Literal, Mapping

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold

from .data import Cohort
from .e1 import (
    CensoredInterval,
    PredictionRecord,
    Rung,
    SampledCase,
    _keyed_rng,
    summarize_predictions,
)
from .probe import fit_predict

SWAP_BUDGETS = (50, 100, 200)
SWAP_TARGET_SHARES = tuple(range(0, 101, 10))
SWAP_DRAW_IDS = tuple(range(20))


def prevalence_matched_counts(n_cases: int, prevalence: float) -> tuple[int, int]:
    """Return positive/negative counts matching a cohort prevalence by rounding."""
    n_positive = round(n_cases * prevalence)
    return n_positive, n_cases - n_positive


def mixture_case_counts(budget: int, target_share: int) -> tuple[int, int]:
    """Return source and target cases for a budget and integer target percentage."""
    n_target = round(budget * target_share / 100)
    return budget - n_target, n_target


def e1_case_coordinate(positive_cases: int, prevalence: float) -> float:
    """Convert an E1 positive-count rung to its prevalence-matched assay count."""
    return float(positive_cases + round(positive_cases * (1 - prevalence) / prevalence))


@dataclass(frozen=True)
class SwapDrawRecord:
    budget: int
    target_share: int
    draw_seed: int
    fold: int
    origin: Literal["source", "target"]
    cohort: str
    case_id: str
    label: int
    site: str


@dataclass(frozen=True)
class SwapPredictionRecord:
    budget: int
    target_share: int
    draw_seed: int
    fold: int
    held_out_sites: tuple[str, ...]
    case_id: str
    label: int
    score: float


@dataclass(frozen=True)
class SwapAucRecord:
    budget: int
    target_share: int
    draw_seed: int
    raw_auc: float
    rank_auc: float
    rank_gap: float
    rank_diverged: bool


@dataclass(frozen=True)
class SwapTraceResult:
    draws: tuple[SwapDrawRecord, ...]
    predictions: tuple[SwapPredictionRecord, ...]
    auc: SwapAucRecord


@dataclass(frozen=True)
class ConditionalEquivalence:
    value: float | None
    defined: bool
    censored: bool
    censor_at: float | None


def conditional_equivalence_interval(
    values: tuple[ConditionalEquivalence, ...],
) -> CensoredInterval:
    """Return a percentile interval while retaining right-censored bounds."""
    if not values or any(not value.defined for value in values):
        raise ValueError("conditional equivalence intervals require defined values")
    ordered = sorted(
        values,
        key=lambda value: float("inf") if value.censored else float(value.value),
    )

    def bound(quantile: float) -> ConditionalEquivalence:
        return ordered[max(0, ceil(quantile * len(ordered)) - 1)]

    lower = bound(0.025)
    upper = bound(0.975)
    return CensoredInterval(
        lower=float(lower.censor_at if lower.censored else lower.value),
        upper=float(upper.censor_at if upper.censored else upper.value),
        lower_censored=lower.censored,
        upper_censored=upper.censored,
    )


@dataclass(frozen=True)
class TargetReferencePoint:
    cases: float
    origin: Literal["e1", "swap"]
    auc: float


def build_target_reference(
    *,
    e1_cold: Mapping[Rung, float],
    target_prevalence: float,
    all_case_coordinate: float,
    swap_target_only: Mapping[int, float],
) -> tuple[TargetReferencePoint, ...]:
    """Combine all registered target-only observations on a target-case axis."""
    if not 0 < target_prevalence <= 1:
        raise ValueError("target_prevalence must be in (0, 1]")
    if 0 not in e1_cold or "all" not in e1_cold:
        raise ValueError("E1 cold points must include zero and all")
    points = []
    for rung, auc in e1_cold.items():
        if rung == "all":
            cases = float(all_case_coordinate)
        else:
            cases = e1_case_coordinate(rung, target_prevalence)
        points.append(TargetReferencePoint(cases, "e1", float(auc)))
    points.extend(
        TargetReferencePoint(float(budget), "swap", float(auc))
        for budget, auc in swap_target_only.items()
    )
    return tuple(sorted(points, key=lambda point: (point.cases, point.origin)))


def conditional_source_case_equivalence(
    mixture_auc: float,
    target_cases: int,
    source_cases: int,
    target_curve: dict[float, float],
) -> ConditionalEquivalence:
    """Invert the monotone target-only curve and average over source cases."""
    if source_cases == 0:
        return ConditionalEquivalence(None, False, False, None)
    if source_cases < 0 or target_cases < 0:
        raise ValueError("case counts must be non-negative")
    if not target_curve or 0.0 not in target_curve:
        raise ValueError("the target-only curve must include zero cases")
    ordered = sorted((float(cases), float(auc)) for cases, auc in target_curve.items())
    x = np.asarray([row[0] for row in ordered])
    if len(np.unique(x)) != len(x):
        raise ValueError("target-only case coordinates must be unique")
    fitted = IsotonicRegression(increasing=True).fit_transform(
        x, np.asarray([row[1] for row in ordered]), sample_weight=np.ones(len(x))
    )
    if mixture_auc > fitted[-1]:
        return ConditionalEquivalence(
            None, True, True, float((x[-1] - target_cases) / source_cases)
        )
    crossing = x[0]
    for index, upper_y in enumerate(fitted):
        if upper_y < mixture_auc:
            continue
        if index:
            lower_x, upper_x = x[index - 1], x[index]
            lower_y = fitted[index - 1]
            crossing = (
                lower_x
                if upper_y == lower_y
                else lower_x
                + (upper_x - lower_x) * (mixture_auc - lower_y) / (upper_y - lower_y)
            )
        break
    return ConditionalEquivalence(
        float((crossing - target_cases) / source_cases), True, False, None
    )


def sample_prevalence_prefix(
    cohort: Cohort,
    held_out_sites: Iterable[str],
    n_cases: int,
    *,
    draw_seed: int,
    fold: int,
) -> tuple[SampledCase, ...]:
    """Take a nested stratified prefix at the cohort's full prevalence."""
    if n_cases < 0:
        raise ValueError("n_cases must be non-negative")
    eligible = ~np.isin(cohort.sites, tuple(held_out_sites))
    n_positive, n_negative = prevalence_matched_counts(n_cases, cohort.prevalence)
    fold_key = str(fold)
    positive = np.flatnonzero(eligible & (cohort.y == 1))
    negative = np.flatnonzero(eligible & (cohort.y == 0))
    if n_positive > len(positive) or n_negative > len(negative):
        raise ValueError(
            f"{cohort.name} portion of {n_cases} cases is infeasible in fold {fold}"
        )
    positive_order = _keyed_rng(
        draw_seed, "swap", cohort.name, fold_key, "positive"
    ).permutation(positive)
    negative_order = _keyed_rng(
        draw_seed, "swap", cohort.name, fold_key, "negative"
    ).permutation(negative)
    chosen = np.concatenate(
        (positive_order[:n_positive], negative_order[:n_negative])
    )
    return tuple(
        SampledCase(
            case_id=str(cohort.case_ids[index]),
            label=int(cohort.y[index]),
            site=str(cohort.sites[index]),
        )
        for index in chosen
    )


def trace_swap_cell(
    source: Cohort,
    target: Cohort,
    *,
    budget: int,
    target_share: int,
    draw_seed: int,
) -> SwapTraceResult:
    """Fit one budget/mixture draw and predict every STAD case out of fold."""
    if budget <= 0:
        raise ValueError("budget must be positive")
    if not 0 <= target_share <= 100:
        raise ValueError("target_share must be between 0 and 100")
    n_source, n_target = mixture_case_counts(budget, target_share)
    indices = {
        source.name: {str(case_id): index for index, case_id in enumerate(source.case_ids)},
        target.name: {str(case_id): index for index, case_id in enumerate(target.case_ids)},
    }
    draws: list[SwapDrawRecord] = []
    predictions: list[SwapPredictionRecord] = []
    folds = GroupKFold(n_splits=5).split(target.X, target.y, target.sites)
    for fold, (_, test_indices) in enumerate(folds):
        held_out_sites = tuple(
            sorted(str(site) for site in np.unique(target.sites[test_indices]))
        )
        portions = (
            ("source", source, sample_prevalence_prefix(
                source, (), n_source, draw_seed=draw_seed, fold=fold
            )),
            ("target", target, sample_prevalence_prefix(
                target, held_out_sites, n_target, draw_seed=draw_seed, fold=fold
            )),
        )
        training_indices = [
            np.asarray([indices[cohort.name][case.case_id] for case in cases], dtype=int)
            for _, cohort, cases in portions
        ]
        training_X = np.concatenate(
            [cohort.X[chosen] for (_, cohort, _), chosen in zip(portions, training_indices)]
        )
        training_y = np.concatenate(
            [cohort.y[chosen] for (_, cohort, _), chosen in zip(portions, training_indices)]
        )
        scores = fit_predict(training_X, training_y, target.X[test_indices])
        for origin, cohort, cases in portions:
            draws.extend(
                SwapDrawRecord(
                    budget, target_share, draw_seed, fold, origin, cohort.name,
                    case.case_id, case.label, case.site,
                )
                for case in cases
            )
        predictions.extend(
            SwapPredictionRecord(
                budget, target_share, draw_seed, fold, held_out_sites,
                str(target.case_ids[index]), int(target.y[index]), float(score),
            )
            for index, score in zip(test_indices, scores)
        )
    generic = tuple(
        PredictionRecord(
            row.draw_seed, row.budget, row.fold, row.held_out_sites, "warm",
            f"{source.name}+{target.name}", target.name, row.case_id,
            row.label, row.score,
        )
        for row in predictions
    )
    (auc,) = summarize_predictions(generic)
    return SwapTraceResult(
        tuple(draws), tuple(predictions),
        SwapAucRecord(
            budget, target_share, draw_seed, auc.raw_auc, auc.rank_auc,
            auc.rank_gap, auc.rank_diverged,
        ),
    )
