import numpy as np

from panmorph.data import Cohort
from panmorph.e1 import (
    PredictionRecord,
    rank_auc_diverged,
    sample_rung,
    summarize_predictions,
    trace_paired_cell,
)


def synthetic_cohorts() -> tuple[Cohort, Cohort]:
    target_y = np.tile(np.asarray([1, 0, 0, 0, 0]), 10)
    target_sites = np.repeat(np.asarray(["A", "B", "C", "D", "E"]), 10)
    target = Cohort(
        name="TARGET",
        X=np.column_stack(
            (2 * target_y - 1 + np.arange(50) / 100, np.arange(50) % 7)
        ).astype(np.float32),
        y=target_y,
        sites=target_sites,
        case_ids=np.asarray([f"T{i:02d}" for i in range(50)]),
    )
    source_y = np.tile(np.asarray([1, 0]), 10)
    source = Cohort(
        name="SOURCE",
        X=np.column_stack((2 * source_y - 1, np.arange(20) % 3)).astype(np.float32),
        y=source_y,
        sites=np.repeat(np.asarray(["X", "Y"]), 10),
        case_ids=np.asarray([f"S{i:02d}" for i in range(20)]),
    )
    return source, target


def test_sample_rung_selects_prevalence_matched_cases_outside_held_out_sites() -> None:
    cohort = Cohort(
        name="TARGET",
        X=np.arange(48, dtype=np.float32).reshape(24, 2),
        y=np.asarray([1] * 6 + [0] * 18),
        sites=np.asarray(["A", "B", "C", "D", "E", "F"] * 4),
        case_ids=np.asarray([f"T{i:02d}" for i in range(24)]),
    )

    selected = sample_rung(cohort, held_out_sites=("A",), k=2, draw_seed=17)

    assert sum(record.label == 1 for record in selected) == 2
    assert sum(record.label == 0 for record in selected) == 6
    assert len({record.case_id for record in selected}) == 8
    assert {record.site for record in selected}.isdisjoint({"A"})


def test_keyed_draws_do_not_depend_on_request_order() -> None:
    _, target = synthetic_cohorts()

    forward = {
        held_out: tuple(case.case_id for case in sample_rung(target, held_out, 1, 41))
        for held_out in (("A",), ("B",), ("C",))
    }
    reverse = {
        held_out: tuple(case.case_id for case in sample_rung(target, held_out, 1, 41))
        for held_out in (("C",), ("B",), ("A",))
    }

    assert reverse == forward


def test_trace_pairs_training_draws_and_covers_every_patient_in_five_site_folds() -> None:
    source, target = synthetic_cohorts()

    result = trace_paired_cell(source, target, k=1, draw_seed=29)

    assert len({record.fold for record in result.predictions}) == 5
    for arm in ("warm", "cold"):
        arm_predictions = [record for record in result.predictions if record.arm == arm]
        assert len(arm_predictions) == 50
        assert {record.case_id for record in arm_predictions} == set(target.case_ids)
        assert len({record.case_id for record in arm_predictions}) == 50

    for fold in range(5):
        fold_draws = [record for record in result.draws if record.fold == fold]
        warm_local = {
            record.case_id
            for record in fold_draws
            if record.arm == "warm" and record.origin == "target"
        }
        cold_local = {
            record.case_id
            for record in fold_draws
            if record.arm == "cold" and record.origin == "target"
        }
        warm_source = {
            record.case_id
            for record in fold_draws
            if record.arm == "warm" and record.origin == "source"
        }
        cold_source = [
            record for record in fold_draws if record.arm == "cold" and record.origin == "source"
        ]
        assert warm_local == cold_local
        assert len(warm_local) == 5
        assert warm_source == set(source.case_ids)
        assert cold_source == []
        assert {record.site for record in fold_draws if record.origin == "target"}.isdisjoint(
            fold_draws[0].held_out_sites
        )

    assert {(record.arm, record.k, record.draw_seed) for record in result.aucs} == {
        ("warm", 1, 29),
        ("cold", 1, 29),
    }


def test_auc_summary_pools_raw_scores_and_percentile_ranks_with_average_ties() -> None:
    labels_scores_folds = [
        (0, 0.1, 0),
        (1, 0.9, 0),
        (0, 0.8, 1),
        (0, 0.6, 1),
        (1, 0.6, 1),
        (1, 0.5, 1),
    ]
    predictions = tuple(
        PredictionRecord(
            draw_seed=3,
            k=1,
            fold=fold,
            held_out_sites=(str(fold),),
            arm="warm",
            source="SOURCE",
            target="TARGET",
            case_id=f"P{index}",
            label=label,
            score=score,
        )
        for index, (label, score, fold) in enumerate(labels_scores_folds)
    )

    (summary,) = summarize_predictions(predictions)

    assert summary.raw_auc == 11 / 18
    assert summary.rank_auc == 4 / 9
    assert summary.rank_gap == 0.16666666666666674
    assert summary.rank_diverged is True
    assert summary.raw_auc != (1.0 + 0.125) / 2


def test_rank_sensitivity_flags_only_gaps_greater_than_point_zero_one() -> None:
    assert rank_auc_diverged(0.70, 0.69) is False
    assert rank_auc_diverged(0.70, 0.689999) is True
