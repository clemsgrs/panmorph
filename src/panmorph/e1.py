"""Site-clean paired warm/cold tracer for the phase-2 E1 experiment."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from decimal import Decimal
from typing import Iterable, Literal, Mapping

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from .data import Cohort
from .probe import fit_predict

Arm = Literal["warm", "cold"]
Origin = Literal["source", "target"]
Rung = int | Literal["all"]
E1_RUNGS: tuple[Rung, ...] = (0, 3, 5, 10, 25, 40, "all")
E1_DRAW_IDS: tuple[int, ...] = tuple(range(20))


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
