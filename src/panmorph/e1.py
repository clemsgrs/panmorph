"""Site-clean paired warm/cold tracer for the phase-2 E1 experiment."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from decimal import Decimal
from typing import Iterable, Literal, Mapping

import numpy as np
from joblib import Parallel, delayed
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from .data import Cohort
from .probe import fit_predict

Arm = Literal["warm", "cold"]
Origin = Literal["source", "target"]
Base = Literal["single", "pooled"]
Rung = int | Literal["all"]
E1_RUNGS: tuple[Rung, ...] = (0, 3, 5, 10, 25, 40, "all")
E1_DRAW_IDS: tuple[int, ...] = tuple(range(20))
BOOTSTRAP_REPLICATES = 2_000
PERMUTATION_COUNT = 999
CONFIRMATORY_CELL = ("COAD", "STAD", "single", 10)
CONFIRMATORY_ALPHA = 0.05


@dataclass(frozen=True)
class SampledCase:
    """One target patient selected for a local-training rung."""

    case_id: str
    label: int
    site: str


@dataclass(frozen=True)
class DrawRecord:
    """One auditable row used to fit an arm for one held-out fold."""

    draw_seed: int | None
    k: Rung
    fold: int
    held_out_sites: tuple[str, ...]
    arm: Arm
    source: str
    target: str
    origin: Origin
    cohort: str
    case_id: str
    label: int
    site: str


@dataclass(frozen=True)
class PredictionRecord:
    """One out-of-fold patient prediction."""

    draw_seed: int | None
    k: Rung
    fold: int
    held_out_sites: tuple[str, ...]
    arm: Arm
    source: str
    target: str
    case_id: str
    label: int
    score: float


@dataclass(frozen=True)
class AucRecord:
    """Pooled out-of-fold AUC for one paired arm."""

    draw_seed: int | None
    k: Rung
    arm: Arm
    source: str
    target: str
    raw_auc: float
    rank_auc: float
    rank_gap: float
    rank_diverged: bool


@dataclass(frozen=True)
class TraceResult:
    draws: tuple[DrawRecord, ...]
    predictions: tuple[PredictionRecord, ...]
    aucs: tuple[AucRecord, ...]


@dataclass(frozen=True)
class IntervalEstimate:
    """A point estimate and its two-sided 95% percentile interval."""

    point: float
    lower: float
    upper: float


@dataclass(frozen=True)
class CellEstimate:
    """Registered paired inference for one warm/cold E1 cell."""

    source: str
    target: str
    base: Base
    k: Rung
    n_draws: int
    warm: IntervalEstimate
    cold: IntervalEstimate
    lift: IntervalEstimate
    rank_warm: float
    rank_cold: float
    rank_lift: float
    rank_diverged: bool
    bootstrap_warm: np.ndarray
    bootstrap_cold: np.ndarray
    bootstrap_lift: np.ndarray
    confirmatory: bool


@dataclass(frozen=True)
class EquivalenceEstimate:
    """Local-positive equivalence, retaining right censoring at ``all``."""

    value: float | None
    censored: bool
    censor_at: float | None


@dataclass(frozen=True)
class CensoredInterval:
    """A percentile interval whose endpoints may be right-censored."""

    lower: float
    upper: float
    lower_censored: bool
    upper_censored: bool


@dataclass(frozen=True)
class EquivalenceDistribution:
    point: EquivalenceEstimate
    interval: CensoredInterval
    bootstrap: tuple[EquivalenceEstimate, ...]


@dataclass(frozen=True)
class EquivalenceSummary(EquivalenceDistribution):
    """Local-positive and per-source-case average equivalence."""

    average_source_case: EquivalenceDistribution


@dataclass(frozen=True)
class ConfirmatoryResult:
    """The sole registered COAD-to-STAD superiority test."""

    observed_lift: float
    p_value: float
    significant: bool
    n_permutations: int
    null_lifts: np.ndarray


@dataclass(frozen=True)
class EquivalenceCellSummary:
    source: str
    target: str
    base: Base
    local_positive: EquivalenceSummary


@dataclass(frozen=True)
class E1Inference:
    cells: tuple[CellEstimate, ...]
    equivalences: tuple[EquivalenceCellSummary, ...]


def rank_auc_diverged(raw_auc: float, rank_auc: float) -> bool:
    """Return whether the pre-specified sensitivity gap is greater than 0.01."""
    gap = abs(Decimal(str(raw_auc)) - Decimal(str(rank_auc)))
    return gap > Decimal("0.01")


def _percentile_ranks(scores: np.ndarray) -> np.ndarray:
    """One-indexed percentile ranks, assigning tied values their average rank."""
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    start = 0
    while start < len(scores):
        stop = start + 1
        while stop < len(scores) and scores[order[stop]] == scores[order[start]]:
            stop += 1
        ranks[order[start:stop]] = ((start + 1) + stop) / 2 / len(scores)
        start = stop
    return ranks


def _binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute binary AUC by its Mann-Whitney rank identity."""
    positive = labels == 1
    n_positive = int(np.sum(positive))
    n_negative = len(labels) - n_positive
    if n_positive == 0 or n_negative == 0:
        raise ValueError("AUC requires both labels")
    ranks = _percentile_ranks(scores) * len(scores)
    u = float(np.sum(ranks[positive])) - n_positive * (n_positive + 1) / 2
    return u / (n_positive * n_negative)


def _bootstrap_auc(
    labels: np.ndarray, scores: np.ndarray, schedule: np.ndarray
) -> np.ndarray:
    """Evaluate a stratified bootstrap schedule with bounded vectorized comparisons."""
    scheduled_labels = labels[schedule[0]]
    positive_columns = scheduled_labels == 1
    negative_columns = ~positive_columns
    aucs = np.empty(len(schedule), dtype=float)
    for start in range(0, len(schedule), 100):
        stop = min(start + 100, len(schedule))
        sampled = scores[schedule[start:stop]]
        positive = sampled[:, positive_columns, None]
        negative = sampled[:, None, negative_columns]
        aucs[start:stop] = np.mean(positive > negative, axis=(1, 2)) + 0.5 * np.mean(
            positive == negative, axis=(1, 2)
        )
    return aucs


def summarize_predictions(
    predictions: tuple[PredictionRecord, ...] | list[PredictionRecord],
) -> tuple[AucRecord, ...]:
    """Compute raw and within-fold-ranked AUCs from concatenated OOF predictions."""
    keys = list(
        dict.fromkeys((p.draw_seed, p.k, p.arm, p.source, p.target) for p in predictions)
    )
    summaries = []
    for draw_seed, k, arm, source, target in keys:
        arm_predictions = [
            p
            for p in predictions
            if (p.draw_seed, p.k, p.arm, p.source, p.target)
            == (draw_seed, k, arm, source, target)
        ]
        labels = np.asarray([p.label for p in arm_predictions])
        scores = np.asarray([p.score for p in arm_predictions])
        ranked = np.empty(len(scores), dtype=float)
        folds = np.asarray([p.fold for p in arm_predictions])
        for fold in np.unique(folds):
            in_fold = folds == fold
            ranked[in_fold] = _percentile_ranks(scores[in_fold])
        raw_auc = float(roc_auc_score(labels, scores))
        rank_auc = float(roc_auc_score(labels, ranked))
        gap = abs(raw_auc - rank_auc)
        summaries.append(
            AucRecord(
                draw_seed=draw_seed,
                k=k,
                arm=arm,
                source=source,
                target=target,
                raw_auc=raw_auc,
                rank_auc=rank_auc,
                rank_gap=gap,
                rank_diverged=rank_auc_diverged(raw_auc, rank_auc),
            )
        )
    return tuple(summaries)


def _keyed_rng(draw_seed: int, *key: str) -> np.random.Generator:
    material = "\x1f".join((str(draw_seed), *key)).encode()
    seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "little")
    return np.random.default_rng(seed)


def stratified_bootstrap_indices(
    labels: np.ndarray,
    *,
    seed: int = 0,
    key: tuple[str, ...] = (),
) -> np.ndarray:
    """Return a deterministic label-stratified patient bootstrap schedule."""
    labels = np.asarray(labels)
    classes = np.unique(labels)
    if len(classes) < 2:
        raise ValueError("patient bootstraps require at least two labels")
    rng = _keyed_rng(seed, "bootstrap", *key)
    strata = tuple(np.flatnonzero(labels == label) for label in classes)
    return np.asarray(
        [
            np.concatenate(
                [rng.choice(indices, size=len(indices), replace=True) for indices in strata]
            )
            for _ in range(BOOTSTRAP_REPLICATES)
        ],
        dtype=int,
    )


def is_confirmatory_cell(source: str, target: str, k: Rung) -> bool:
    """Return whether this is the sole registered confirmatory E1 cell."""
    return (source, target, _source_base(source), k) == CONFIRMATORY_CELL


def _source_members(source: str) -> tuple[str, ...]:
    return tuple(source.split("+"))


def _source_base(source: str) -> Base:
    return "pooled" if len(_source_members(source)) > 1 else "single"


def count_source_cases(
    cohorts: Mapping[str, Cohort], sources: Iterable[str]
) -> dict[str, int]:
    """Count source patients for single and pooled E1 bases."""
    return {
        source: sum(cohorts[name].n for name in _source_members(source))
        for source in sources
    }


def source_label_permutations(
    labels: np.ndarray,
    source: str,
    *,
    seed: int = 0,
) -> np.ndarray:
    """Generate keyed source-label permutations without changing prevalence."""
    labels = np.asarray(labels)
    rng = _keyed_rng(seed, "permutation", source)
    return np.asarray([rng.permutation(labels) for _ in range(PERMUTATION_COUNT)])


def empirical_superiority_p(observed: float, null: np.ndarray) -> float:
    """One-sided empirical p-value with the registered plus-one correction."""
    null = np.asarray(null)
    return float((1 + np.sum(null >= observed)) / (1 + len(null)))


def run_confirmatory_test(
    source: Cohort,
    target: Cohort,
    observed_predictions: tuple[PredictionRecord, ...] | list[PredictionRecord],
    *,
    seed: int = 0,
    n_jobs: int = -1,
) -> ConfirmatoryResult:
    """Run the one registered 999-permutation mean-lift superiority test."""
    if not is_confirmatory_cell(source.name, target.name, 10):
        raise ValueError("only single-source COAD -> STAD at k=10 is confirmatory")
    observed_warm = summarize_predictions(
        [
            record
            for record in observed_predictions
            if (record.source, record.target, record.k, record.arm)
            == ("COAD", "STAD", 10, "warm")
        ]
    )
    observed_cold = summarize_predictions(
        [
            record
            for record in observed_predictions
            if (record.source, record.target, record.k, record.arm)
            == ("target-only", "STAD", 10, "cold")
        ]
    )
    warm_by_draw = {record.draw_seed: record.raw_auc for record in observed_warm}
    cold_by_draw = {record.draw_seed: record.raw_auc for record in observed_cold}
    if tuple(sorted(warm_by_draw)) != E1_DRAW_IDS or warm_by_draw.keys() != cold_by_draw.keys():
        raise ValueError("confirmatory predictions require all twenty fixed paired draws")
    observed_lift = float(
        np.mean([warm_by_draw[draw] - cold_by_draw[draw] for draw in E1_DRAW_IDS])
    )

    permutations = source_label_permutations(source.y, source.name, seed=seed)

    def null_lift(shuffled_labels: np.ndarray) -> float:
        shuffled_source = replace(source, y=shuffled_labels)
        warm_aucs = []
        for draw in E1_DRAW_IDS:
            result = trace_paired_cell(
                shuffled_source, target, 10, draw, arms=("warm",)
            )
            warm_aucs.append(result.aucs[0].raw_auc)
        return float(
            np.mean(
                [
                    warm_auc - cold_by_draw[draw]
                    for draw, warm_auc in zip(E1_DRAW_IDS, warm_aucs)
                ]
            )
        )

    null_lifts = np.asarray(
        Parallel(n_jobs=n_jobs)(delayed(null_lift)(labels) for labels in permutations)
    )
    p_value = empirical_superiority_p(observed_lift, null_lifts)
    return ConfirmatoryResult(
        observed_lift=observed_lift,
        p_value=p_value,
        significant=p_value < CONFIRMATORY_ALPHA,
        n_permutations=len(null_lifts),
        null_lifts=null_lifts,
    )


def _ordered_draw_scores(
    records: list[PredictionRecord],
    case_ids: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    by_case = {record.case_id: record for record in records}
    if len(by_case) != len(records) or set(by_case) != set(case_ids):
        raise ValueError("each arm and draw must contain the same patients exactly once")
    labels = np.asarray([by_case[case_id].label for case_id in case_ids], dtype=int)
    scores = np.asarray([by_case[case_id].score for case_id in case_ids], dtype=float)
    return labels, scores


def _interval(point: float, bootstrap: np.ndarray) -> IntervalEstimate:
    lower, upper = np.percentile(bootstrap, (2.5, 97.5))
    return IntervalEstimate(point, float(lower), float(upper))


def estimate_e1_cell(
    predictions: tuple[PredictionRecord, ...] | list[PredictionRecord],
    source: str,
    target: str,
    k: Rung,
    *,
    seed: int = 0,
) -> CellEstimate:
    """Estimate mean arm AUCs and paired lift from stored OOF predictions."""
    warm = [
        record
        for record in predictions
        if (record.source, record.target, record.k, record.arm)
        == (source, target, k, "warm")
    ]
    cold = [
        record
        for record in predictions
        if (record.source, record.target, record.k, record.arm)
        == ("target-only", target, k, "cold")
    ]
    if not warm or not cold:
        raise ValueError(f"missing paired predictions for {source} -> {target} at k={k}")
    warm_draws = tuple(sorted({record.draw_seed for record in warm}, key=str))
    cold_draws = tuple(sorted({record.draw_seed for record in cold}, key=str))
    expected_draws: tuple[int | None, ...] = (
        (None,) if k in (0, "all") else E1_DRAW_IDS
    )
    if warm_draws != cold_draws or set(warm_draws) != set(expected_draws):
        raise ValueError("warm and cold arms must use the complete fixed draw schedule")

    case_ids = tuple(sorted({record.case_id for record in warm}))
    warm_scores: list[np.ndarray] = []
    cold_scores: list[np.ndarray] = []
    labels: np.ndarray | None = None
    for draw in warm_draws:
        warm_labels, draw_warm = _ordered_draw_scores(
            [record for record in warm if record.draw_seed == draw], case_ids
        )
        cold_labels, draw_cold = _ordered_draw_scores(
            [record for record in cold if record.draw_seed == draw], case_ids
        )
        if not np.array_equal(warm_labels, cold_labels):
            raise ValueError("paired warm and cold patients must have identical labels")
        if labels is not None and not np.array_equal(labels, warm_labels):
            raise ValueError("patient labels must be fixed across draws")
        labels = warm_labels
        warm_scores.append(draw_warm)
        cold_scores.append(draw_cold)
    assert labels is not None

    point_warm = float(np.mean([_binary_auc(labels, scores) for scores in warm_scores]))
    point_cold = float(np.mean([_binary_auc(labels, scores) for scores in cold_scores]))
    schedule = stratified_bootstrap_indices(
        labels, seed=seed, key=(target,)
    )
    bootstrap_warm = np.mean(
        [_bootstrap_auc(labels, scores, schedule) for scores in warm_scores], axis=0
    )
    bootstrap_cold = np.mean(
        [_bootstrap_auc(labels, scores, schedule) for scores in cold_scores], axis=0
    )
    bootstrap_lift = bootstrap_warm - bootstrap_cold

    warm_rank = float(np.mean([record.rank_auc for record in summarize_predictions(warm)]))
    cold_rank = float(np.mean([record.rank_auc for record in summarize_predictions(cold)]))
    rank_lift = warm_rank - cold_rank
    point_lift = point_warm - point_cold
    return CellEstimate(
        source=source,
        target=target,
        base=_source_base(source),
        k=k,
        n_draws=len(warm_draws),
        warm=_interval(point_warm, bootstrap_warm),
        cold=_interval(point_cold, bootstrap_cold),
        lift=_interval(point_lift, bootstrap_lift),
        rank_warm=warm_rank,
        rank_cold=cold_rank,
        rank_lift=rank_lift,
        rank_diverged=any(
            (
                rank_auc_diverged(point_warm, warm_rank),
                rank_auc_diverged(point_cold, cold_rank),
                rank_auc_diverged(point_lift, rank_lift),
            )
        ),
        bootstrap_warm=bootstrap_warm,
        bootstrap_cold=bootstrap_cold,
        bootstrap_lift=bootstrap_lift,
        confirmatory=is_confirmatory_cell(source, target, k),
    )


def local_positive_equivalence(
    foreign_only_auc: float,
    cold_curve: Mapping[Rung, float],
    *,
    all_coordinate: float,
) -> EquivalenceEstimate:
    """Find the first linear crossing of the equal-weight isotonic cold curve."""
    if foreign_only_auc <= 0.5:
        return EquivalenceEstimate(0.0, False, None)
    if "all" not in cold_curve:
        raise ValueError("the cold curve must include the all endpoint")
    numeric = sorted((float(k), float(value)) for k, value in cold_curve.items() if k != "all")
    if not numeric:
        raise ValueError("the cold curve must include at least one numeric rung")
    if all_coordinate <= numeric[-1][0]:
        raise ValueError("the all coordinate must lie beyond every numeric rung")
    x = np.asarray([point[0] for point in numeric] + [float(all_coordinate)])
    y = np.asarray([point[1] for point in numeric] + [float(cold_curve["all"])])
    fitted = IsotonicRegression(increasing=True).fit_transform(
        x, y, sample_weight=np.ones(len(x))
    )

    for index, value in enumerate(fitted):
        if value < foreign_only_auc:
            continue
        if index == 0:
            crossing = x[0]
        else:
            lower_x, upper_x = x[index - 1], x[index]
            lower_y, upper_y = fitted[index - 1], value
            crossing = (
                lower_x
                if upper_y == lower_y
                else lower_x
                + (upper_x - lower_x)
                * (foreign_only_auc - lower_y)
                / (upper_y - lower_y)
            )
        return EquivalenceEstimate(float(crossing), False, None)
    return EquivalenceEstimate(None, True, float(all_coordinate))


def _censored_percentile(
    values: tuple[EquivalenceEstimate, ...], quantile: float
) -> EquivalenceEstimate:
    ordered = sorted(
        values,
        key=lambda value: float("inf") if value.censored else float(value.value),
    )
    index = max(0, int(np.ceil(quantile * len(ordered))) - 1)
    return ordered[index]


def _equivalence_distribution(
    point: EquivalenceEstimate,
    bootstrap: tuple[EquivalenceEstimate, ...],
) -> EquivalenceDistribution:
    lower = _censored_percentile(bootstrap, 0.025)
    upper = _censored_percentile(bootstrap, 0.975)
    return EquivalenceDistribution(
        point=point,
        interval=CensoredInterval(
            lower=float(lower.censor_at if lower.censored else lower.value),
            upper=float(upper.censor_at if upper.censored else upper.value),
            lower_censored=lower.censored,
            upper_censored=upper.censored,
        ),
        bootstrap=bootstrap,
    )


def _scale_equivalence(
    estimate: EquivalenceEstimate, denominator: int
) -> EquivalenceEstimate:
    if estimate.censored:
        return EquivalenceEstimate(None, True, float(estimate.censor_at) / denominator)
    return EquivalenceEstimate(float(estimate.value) / denominator, False, None)


def summarize_equivalence_bootstrap(
    *,
    foreign_only_auc: float,
    cold_curve: Mapping[Rung, float],
    foreign_bootstrap: np.ndarray,
    cold_bootstrap: Mapping[Rung, np.ndarray],
    all_coordinate: float,
    source_case_count: int,
) -> EquivalenceSummary:
    """Recompute equivalence per bootstrap and report its censored interval."""
    foreign_bootstrap = np.asarray(foreign_bootstrap)
    if len(foreign_bootstrap) != BOOTSTRAP_REPLICATES:
        raise ValueError(f"equivalence requires exactly {BOOTSTRAP_REPLICATES} bootstraps")
    if source_case_count <= 0:
        raise ValueError("source_case_count must be positive")
    if set(cold_curve) != set(cold_bootstrap):
        raise ValueError("point and bootstrap cold curves must have identical rungs")
    if any(len(np.asarray(values)) != BOOTSTRAP_REPLICATES for values in cold_bootstrap.values()):
        raise ValueError(f"equivalence requires exactly {BOOTSTRAP_REPLICATES} bootstraps")

    point = local_positive_equivalence(
        foreign_only_auc, cold_curve, all_coordinate=all_coordinate
    )
    bootstrap = tuple(
        local_positive_equivalence(
            float(foreign_bootstrap[index]),
            {k: float(values[index]) for k, values in cold_bootstrap.items()},
            all_coordinate=all_coordinate,
        )
        for index in range(BOOTSTRAP_REPLICATES)
    )
    average_point = _scale_equivalence(point, source_case_count)
    average_bootstrap = tuple(
        _scale_equivalence(value, source_case_count) for value in bootstrap
    )
    average = _equivalence_distribution(average_point, average_bootstrap)
    distribution = _equivalence_distribution(point, bootstrap)
    return EquivalenceSummary(
        point=distribution.point,
        interval=distribution.interval,
        bootstrap=distribution.bootstrap,
        average_source_case=average,
    )


def _all_positive_coordinate(
    predictions: tuple[PredictionRecord, ...] | list[PredictionRecord], target: str
) -> float:
    rows = [
        record
        for record in predictions
        if (record.source, record.target, record.k, record.arm)
        == ("target-only", target, "all", "cold")
    ]
    by_case = {record.case_id: record for record in rows}
    if len(by_case) != len(rows) or not rows:
        raise ValueError(f"{target} requires one complete cold all prediction cohort")
    total_positive = sum(record.label == 1 for record in rows)
    folds = {record.fold for record in rows}
    return float(
        np.mean(
            [
                total_positive
                - sum(record.label == 1 and record.fold == fold for record in rows)
                for fold in folds
            ]
        )
    )


def estimate_e1_matrix(
    predictions: tuple[PredictionRecord, ...] | list[PredictionRecord],
    source_case_counts: Mapping[str, int],
    *,
    seed: int = 0,
) -> E1Inference:
    """Turn the complete stored E1 predictions into registered estimates."""
    keys = {
        (record.source, record.target, record.k)
        for record in predictions
        if record.arm == "warm"
    }
    cells = tuple(
        estimate_e1_cell(predictions, source, target, k, seed=seed)
        for source, target, k in sorted(keys, key=lambda key: (key[1], key[0], str(key[2])))
    )
    pairs = sorted({(cell.source, cell.target) for cell in cells})
    equivalences = []
    for source, target in pairs:
        pair_cells = [cell for cell in cells if (cell.source, cell.target) == (source, target)]
        by_rung = {cell.k: cell for cell in pair_cells}
        if 0 not in by_rung or "all" not in by_rung:
            raise ValueError(f"{source} -> {target} equivalence requires 0 and all endpoints")
        if source not in source_case_counts:
            raise ValueError(f"missing source case count for {source}")
        summary = summarize_equivalence_bootstrap(
            foreign_only_auc=by_rung[0].warm.point,
            cold_curve={k: cell.cold.point for k, cell in by_rung.items()},
            foreign_bootstrap=by_rung[0].bootstrap_warm,
            cold_bootstrap={k: cell.bootstrap_cold for k, cell in by_rung.items()},
            all_coordinate=_all_positive_coordinate(predictions, target),
            source_case_count=source_case_counts[source],
        )
        equivalences.append(
            EquivalenceCellSummary(source, target, by_rung[0].base, summary)
        )
    return E1Inference(cells, tuple(equivalences))


def _numeric_rung_pool(
    cohort: Cohort,
    held_out_sites: tuple[str, ...],
    k: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    eligible = ~np.isin(cohort.sites, held_out_sites)
    positive = np.flatnonzero(eligible & (cohort.y == 1))
    negative = np.flatnonzero(eligible & (cohort.y == 0))
    required_negative = round(k * (1.0 - cohort.prevalence) / cohort.prevalence)
    return positive, negative, required_negative


def sample_rung(
    cohort: Cohort,
    held_out_sites: tuple[str, ...],
    k: Rung,
    draw_seed: int | None,
) -> tuple[SampledCase, ...]:
    """Select ``k`` positives and prevalence-matched negatives outside test sites."""
    eligible = ~np.isin(cohort.sites, held_out_sites)
    if k == "all":
        chosen = np.flatnonzero(eligible)
        return tuple(
            SampledCase(
                case_id=str(cohort.case_ids[index]),
                label=int(cohort.y[index]),
                site=str(cohort.sites[index]),
            )
            for index in chosen
        )

    if k < 0:
        raise ValueError("k must be non-negative")
    if k == 0:
        return ()

    positive, negative, n_negative = _numeric_rung_pool(
        cohort, held_out_sites, k
    )
    if k > len(positive) or n_negative > len(negative):
        raise ValueError("rung exceeds the eligible cases outside the held-out sites")

    fold_key = ",".join(sorted(str(site) for site in held_out_sites))
    if draw_seed is None:
        raise ValueError("numeric rungs require a draw seed")
    rng = _keyed_rng(draw_seed, cohort.name, fold_key, str(k))
    chosen = np.concatenate(
        (
            rng.choice(positive, size=k, replace=False),
            rng.choice(negative, size=n_negative, replace=False),
        )
    )
    return tuple(
        SampledCase(
            case_id=str(cohort.case_ids[index]),
            label=int(cohort.y[index]),
            site=str(cohort.sites[index]),
        )
        for index in chosen
    )


def preflight_rungs(
    target: Cohort,
    rungs: tuple[Rung, ...] = E1_RUNGS,
) -> None:
    """Fail before execution if any numeric rung cannot be drawn in every fold."""
    numeric_rungs = tuple(k for k in rungs if k != "all")
    folds = GroupKFold(n_splits=5).split(target.X, target.y, target.sites)
    for fold, (_, test_indices) in enumerate(folds):
        held_out_sites = tuple(
            sorted(str(site) for site in np.unique(target.sites[test_indices]))
        )
        for k in numeric_rungs:
            if k < 0:
                raise ValueError(f"{target.name} rung {k} must be non-negative")
            positive, negative, required_negative = _numeric_rung_pool(
                target, held_out_sites, k
            )
            if k > len(positive) or required_negative > len(negative):
                raise ValueError(
                    f"{target.name} rung {k} is infeasible in fold {fold}: "
                    f"needs {k} positives and {required_negative} negatives, "
                    f"has {len(positive)} positives and {len(negative)} negatives"
                )


def trace_paired_cell(
    source: Cohort,
    target: Cohort,
    k: Rung,
    draw_seed: int | None,
    arms: tuple[Arm, ...] = ("warm", "cold"),
) -> TraceResult:
    """Run one paired warm/cold E1 cell over site-grouped target folds."""
    target_index = {str(case_id): index for index, case_id in enumerate(target.case_ids)}
    draws: list[DrawRecord] = []
    predictions: list[PredictionRecord] = []

    folds = GroupKFold(n_splits=5).split(target.X, target.y, target.sites)
    for fold, (_, test_indices) in enumerate(folds):
        held_out_sites = tuple(sorted(str(site) for site in np.unique(target.sites[test_indices])))
        local_cases = sample_rung(target, held_out_sites, k, draw_seed)
        local_indices = np.asarray(
            [target_index[case.case_id] for case in local_cases], dtype=int
        )

        for arm in arms:
            if arm == "warm":
                training_X = np.concatenate((source.X, target.X[local_indices]))
                training_y = np.concatenate((source.y, target.y[local_indices]))
                training_rows = (
                    ("source", source, index) for index in range(source.n)
                )
            else:
                training_X = target.X[local_indices]
                training_y = target.y[local_indices]
                training_rows = iter(())

            for origin, cohort, index in training_rows:
                draws.append(
                    DrawRecord(
                        draw_seed=draw_seed,
                        k=k,
                        fold=fold,
                        held_out_sites=held_out_sites,
                        arm=arm,
                        source=source.name,
                        target=target.name,
                        origin=origin,
                        cohort=cohort.name,
                        case_id=str(cohort.case_ids[index]),
                        label=int(cohort.y[index]),
                        site=str(cohort.sites[index]),
                    )
                )
            for case in local_cases:
                draws.append(
                    DrawRecord(
                        draw_seed=draw_seed,
                        k=k,
                        fold=fold,
                        held_out_sites=held_out_sites,
                        arm=arm,
                        source=source.name,
                        target=target.name,
                        origin="target",
                        cohort=target.name,
                        case_id=case.case_id,
                        label=case.label,
                        site=case.site,
                    )
                )

            scores = (
                np.full(len(test_indices), 0.5)
                if arm == "cold" and k == 0
                else fit_predict(training_X, training_y, target.X[test_indices])
            )
            predictions.extend(
                PredictionRecord(
                    draw_seed=draw_seed,
                    k=k,
                    fold=fold,
                    held_out_sites=held_out_sites,
                    arm=arm,
                    source=source.name,
                    target=target.name,
                    case_id=str(target.case_ids[index]),
                    label=int(target.y[index]),
                    score=float(score),
                )
                for index, score in zip(test_indices, scores)
            )

    aucs = summarize_predictions(predictions)
    return TraceResult(tuple(draws), tuple(predictions), aucs)


def _pool_cohorts(cohorts: tuple[Cohort, ...]) -> Cohort:
    return Cohort(
        name="+".join(sorted(cohort.name for cohort in cohorts)),
        X=np.concatenate([cohort.X for cohort in cohorts]),
        y=np.concatenate([cohort.y for cohort in cohorts]),
        sites=np.concatenate([cohort.sites for cohort in cohorts]),
        case_ids=np.concatenate([cohort.case_ids for cohort in cohorts]),
    )


def _rename_source(result: TraceResult, source: str) -> TraceResult:
    return TraceResult(
        draws=tuple(replace(record, source=source) for record in result.draws),
        predictions=tuple(
            replace(record, source=source) for record in result.predictions
        ),
        aucs=tuple(replace(record, source=source) for record in result.aucs),
    )


def _restore_pooled_provenance(
    result: TraceResult,
    cohorts: tuple[Cohort, ...],
) -> TraceResult:
    origins = {
        str(case_id): cohort.name
        for cohort in cohorts
        for case_id in cohort.case_ids
    }
    if len(origins) != sum(cohort.n for cohort in cohorts):
        raise ValueError("pooled source cohorts contain duplicate case IDs")
    return TraceResult(
        draws=tuple(
            replace(record, cohort=origins[record.case_id])
            if record.origin == "source"
            else record
            for record in result.draws
        ),
        predictions=result.predictions,
        aucs=result.aucs,
    )


def run_e1_matrix(
    cohorts: dict[str, Cohort],
    rungs: tuple[Rung, ...] = E1_RUNGS,
    draw_ids: tuple[int, ...] = E1_DRAW_IDS,
) -> TraceResult:
    """Run every single and pooled foreign base with deduplicated cold arms."""
    if len(cohorts) < 2:
        raise ValueError("E1 requires at least two cohorts")
    for target in cohorts.values():
        preflight_rungs(target, rungs)

    draws: list[DrawRecord] = []
    predictions: list[PredictionRecord] = []
    aucs: list[AucRecord] = []

    def collect(result: TraceResult) -> None:
        draws.extend(result.draws)
        predictions.extend(result.predictions)
        aucs.extend(result.aucs)

    for target_name in sorted(cohorts):
        target = cohorts[target_name]
        foreign = tuple(
            cohorts[name] for name in sorted(cohorts) if name != target_name
        )
        bases = (
            *((cohort, (cohort,)) for cohort in foreign),
            (_pool_cohorts(foreign), foreign),
        )
        for k in rungs:
            seeds: tuple[int | None, ...] = (
                (None,) if k in (0, "all") else draw_ids
            )
            for draw_seed in seeds:
                cold = trace_paired_cell(
                    foreign[0], target, k, draw_seed, arms=("cold",)
                )
                collect(_rename_source(cold, "target-only"))
                for source, members in bases:
                    warm = trace_paired_cell(
                        source, target, k, draw_seed, arms=("warm",)
                    )
                    if len(members) > 1:
                        warm = _restore_pooled_provenance(warm, members)
                    collect(warm)
    return TraceResult(tuple(draws), tuple(predictions), tuple(aucs))


def _canonical_source(source: str) -> str:
    source = source.removesuffix(" (combined)")
    return "+".join(sorted(source.split("+")))


def validate_phase1_anchors(
    aucs: tuple[AucRecord, ...] | list[AucRecord],
    phase1_rows: Iterable[Mapping[str, object]],
    tolerance: Decimal = Decimal("0.000001"),
) -> None:
    """Require E1 deterministic endpoints to reproduce committed Phase-1 AUCs."""
    actual = {
        (record.arm, record.source, record.target): record.raw_auc
        for record in aucs
        if (record.arm == "warm" and record.k == 0)
        or (record.arm == "cold" and record.k == "all")
    }
    expected: dict[tuple[Arm, str, str], float] = {}
    for row in phase1_rows:
        component = str(row["component"])
        target = str(row["target"])
        if component == "zeroshot":
            key = ("warm", _canonical_source(str(row["source"])), target)
        elif component == "ceiling":
            key = ("cold", "target-only", target)
        else:
            continue
        expected[key] = float(row["auc"])

    for key in actual.keys() - expected.keys():
        raise ValueError(f"missing committed Phase-1 counterpart for endpoint: {key}")
    for key, expected_auc in expected.items():
        if key not in actual:
            raise ValueError(f"missing Phase-1 anchor endpoint: {key}")
        gap = abs(Decimal(str(actual[key])) - Decimal(str(expected_auc)))
        if gap > tolerance:
            raise ValueError(
                "Phase-1 anchor mismatch for "
                f"{key[0]} {key[1]} -> {key[2]}: "
                f"E1={actual[key]:.12g}, Phase-1={expected_auc:.12g}, gap={gap}"
            )
