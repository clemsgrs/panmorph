"""Site-clean paired warm/cold tracer for the phase-2 E1 experiment."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from .data import Cohort
from .probe import fit_predict

Arm = Literal["warm", "cold"]
Origin = Literal["source", "target"]


@dataclass(frozen=True)
class SampledCase:
    """One target patient selected for a local-training rung."""

    case_id: str
    label: int
    site: str


@dataclass(frozen=True)
class DrawRecord:
    """One auditable row used to fit an arm for one held-out fold."""

    draw_seed: int
    k: int
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

    draw_seed: int
    k: int
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

    draw_seed: int
    k: int
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


def sample_rung(
    cohort: Cohort,
    held_out_sites: tuple[str, ...],
    k: int,
    draw_seed: int,
) -> tuple[SampledCase, ...]:
    """Select ``k`` positives and prevalence-matched negatives outside test sites."""
    if k < 0:
        raise ValueError("k must be non-negative")

    eligible = ~np.isin(cohort.sites, held_out_sites)
    positive = np.flatnonzero(eligible & (cohort.y == 1))
    negative = np.flatnonzero(eligible & (cohort.y == 0))
    n_negative = round(k * (1.0 - cohort.prevalence) / cohort.prevalence)
    if k > len(positive) or n_negative > len(negative):
        raise ValueError("rung exceeds the eligible cases outside the held-out sites")

    fold_key = ",".join(sorted(str(site) for site in held_out_sites))
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


def trace_paired_cell(
    source: Cohort,
    target: Cohort,
    k: int,
    draw_seed: int,
) -> TraceResult:
    """Run one paired warm/cold E1 cell over site-grouped target folds."""
    target_index = {str(case_id): index for index, case_id in enumerate(target.case_ids)}
    draws: list[DrawRecord] = []
    predictions: list[PredictionRecord] = []

    folds = GroupKFold(n_splits=5).split(target.X, target.y, target.sites)
    for fold, (_, test_indices) in enumerate(folds):
        held_out_sites = tuple(sorted(str(site) for site in np.unique(target.sites[test_indices])))
        local_cases = sample_rung(target, held_out_sites, k, draw_seed)
        local_indices = np.asarray([target_index[case.case_id] for case in local_cases])

        for arm in ("warm", "cold"):
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

            scores = fit_predict(training_X, training_y, target.X[test_indices])
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
