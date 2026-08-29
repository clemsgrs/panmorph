import csv
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from experiments.run_e1 import write_reportable_results
from panmorph.data import Cohort
from panmorph.e1 import (
    AucRecord,
    DrawRecord,
    E1_DRAW_IDS,
    E1_RUNGS,
    PredictionRecord,
    TraceResult,
    preflight_rungs,
    rank_auc_diverged,
    run_e1_matrix,
    sample_rung,
    summarize_predictions,
    trace_paired_cell,
    validate_phase1_anchors,
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

    assert [(record.case_id, record.label, record.site) for record in selected] == [
        ("T05", 1, "F"),
        ("T01", 1, "B"),
        ("T20", 0, "C"),
        ("T08", 0, "C"),
        ("T15", 0, "D"),
        ("T21", 0, "D"),
        ("T22", 0, "E"),
        ("T14", 0, "C"),
    ]
    assert sum(record.label == 1 for record in selected) == 2
    assert sum(record.label == 0 for record in selected) == 6
    assert len({record.case_id for record in selected}) == 8
    assert {record.site for record in selected}.isdisjoint({"A"})


def test_keyed_draws_do_not_depend_on_request_order() -> None:
    _, target = synthetic_cohorts()
    expected = {
        ("A",): ("T35", "T21", "T26", "T36", "T39"),
        ("B",): ("T35", "T09", "T39", "T08", "T48"),
        ("C",): ("T15", "T18", "T34", "T37", "T47"),
    }

    forward = {
        held_out: tuple(case.case_id for case in sample_rung(target, held_out, 1, 41))
        for held_out in (("A",), ("B",), ("C",))
    }
    reverse = {
        held_out: tuple(case.case_id for case in sample_rung(target, held_out, 1, 41))
        for held_out in (("C",), ("B",), ("A",))
    }

    assert forward == expected
    assert reverse == expected


def test_all_rung_returns_every_case_outside_the_held_out_sites() -> None:
    _, target = synthetic_cohorts()

    selected = sample_rung(target, held_out_sites=("A",), k="all", draw_seed=None)

    assert {case.case_id for case in selected} == {
        str(case_id)
        for case_id, site in zip(target.case_ids, target.sites)
        if site != "A"
    }
    assert len(selected) == 40


def test_preflight_rejects_an_infeasible_numeric_rung_without_clipping() -> None:
    _, target = synthetic_cohorts()

    with np.testing.assert_raises_regex(
        ValueError, "TARGET rung 10 is infeasible in fold 0"
    ):
        preflight_rungs(target, (3, 10))


def test_registered_sweep_uses_the_fixed_rungs_and_twenty_draw_ids() -> None:
    assert E1_RUNGS == (0, 3, 5, 10, 25, 40, "all")
    assert E1_DRAW_IDS == tuple(range(20))


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


def test_zero_endpoint_uses_foreign_only_warm_and_unfitted_cold_baseline() -> None:
    source, target = synthetic_cohorts()

    result = trace_paired_cell(source, target, k=0, draw_seed=None)

    assert {record.draw_seed for record in result.predictions} == {None}
    assert {record.score for record in result.predictions if record.arm == "cold"} == {
        0.5
    }
    assert {
        record.raw_auc for record in result.aucs if record.arm == "cold"
    } == {0.5}
    assert {
        record.case_id
        for record in result.draws
        if record.arm == "warm" and record.origin == "source"
    } == set(source.case_ids)
    assert not [record for record in result.draws if record.origin == "target"]


def test_all_endpoint_uses_every_eligible_target_training_case() -> None:
    source, target = synthetic_cohorts()

    result = trace_paired_cell(source, target, k="all", draw_seed=None)

    for fold in range(5):
        fold_predictions = [p for p in result.predictions if p.fold == fold]
        held_out = set(fold_predictions[0].held_out_sites)
        expected = {
            str(case_id)
            for case_id, site in zip(target.case_ids, target.sites)
            if site not in held_out
        }
        for arm in ("warm", "cold"):
            actual = {
                row.case_id
                for row in result.draws
                if row.fold == fold and row.arm == arm and row.origin == "target"
            }
            assert actual == expected


@pytest.fixture(scope="module")
def matrix_fixture() -> tuple[TraceResult, dict[str, Cohort]]:
    _, target = synthetic_cohorts()
    cohorts = {
        "A": replace(target, name="A"),
        "B": replace(
            target,
            name="B",
            case_ids=np.asarray([f"B{i:02d}" for i in range(50)]),
        ),
        "C": replace(
            target,
            name="C",
            case_ids=np.asarray([f"C{i:02d}" for i in range(50)]),
        ),
    }
    return run_e1_matrix(cohorts, rungs=(0, 1, "all"), draw_ids=(0, 1)), cohorts


def test_matrix_runs_every_single_and_pooled_foreign_base(
    matrix_fixture: tuple[TraceResult, dict[str, Cohort]],
) -> None:
    result, _ = matrix_fixture
    warm_sources = {
        (record.source, record.target)
        for record in result.aucs
        if record.arm == "warm"
    }
    assert warm_sources == {
        ("B", "A"),
        ("C", "A"),
        ("B+C", "A"),
        ("A", "B"),
        ("C", "B"),
        ("A+C", "B"),
        ("A", "C"),
        ("B", "C"),
        ("A+B", "C"),
    }


def test_pooled_base_audit_rows_preserve_each_source_cohort(
    matrix_fixture: tuple[TraceResult, dict[str, Cohort]],
) -> None:
    result, _ = matrix_fixture
    pooled_source_rows = [
        row
        for row in result.draws
        if row.source == "B+C" and row.target == "A" and row.origin == "source"
    ]

    assert {row.cohort for row in pooled_source_rows} == {"B", "C"}


def test_matrix_stores_each_source_independent_cold_result_once(
    matrix_fixture: tuple[TraceResult, dict[str, Cohort]],
) -> None:
    result, _ = matrix_fixture
    cold_keys = [
        (record.target, record.k, record.draw_seed)
        for record in result.aucs
        if record.arm == "cold"
    ]
    assert len(cold_keys) == len(set(cold_keys)) == 12
    assert {
        record.source for record in result.aucs if record.arm == "cold"
    } == {"target-only"}


def test_matrix_stores_endpoints_once_and_sampled_rungs_per_draw(
    matrix_fixture: tuple[TraceResult, dict[str, Cohort]],
) -> None:
    result, _ = matrix_fixture
    endpoint_records = [record for record in result.aucs if record.k in (0, "all")]
    sampled_records = [record for record in result.aucs if record.k == 1]
    assert {record.draw_seed for record in endpoint_records} == {None}
    assert {record.draw_seed for record in sampled_records} == {0, 1}
    assert len(result.aucs) == 48


def test_every_matrix_key_has_exactly_one_complete_pooled_oof_cohort(
    matrix_fixture: tuple[TraceResult, dict[str, Cohort]],
) -> None:
    result, cohorts = matrix_fixture
    prediction_keys = {
        (record.source, record.target, record.arm, record.k, record.draw_seed)
        for record in result.predictions
    }
    for key in prediction_keys:
        rows = [
            record
            for record in result.predictions
            if (record.source, record.target, record.arm, record.k, record.draw_seed)
            == key
        ]
        assert len(rows) == cohorts[key[1]].n
        assert len({record.case_id for record in rows}) == cohorts[key[1]].n


def _phase1_records() -> tuple[tuple[AucRecord, ...], tuple[dict[str, str], ...]]:
    return (
        (
        AucRecord(None, 0, "warm", "B+C", "A", 0.7, 0.7, 0.0, False),
        AucRecord(None, "all", "cold", "target-only", "A", 0.8, 0.8, 0.0, False),
        ),
        (
            {
                "component": "zeroshot",
                "source": "C+B (combined)",
                "target": "A",
                "auc": "0.700001",
            },
            {
                "component": "ceiling",
                "source": "A (within)",
                "target": "A",
                "auc": "0.799999",
            },
        ),
    )


def test_phase1_anchor_tolerance_is_inclusive_at_one_e_minus_six() -> None:
    endpoint_aucs, phase1_rows = _phase1_records()
    validate_phase1_anchors(endpoint_aucs, phase1_rows)


def test_phase1_anchor_rejects_a_gap_above_one_e_minus_six() -> None:
    endpoint_aucs, phase1_rows = _phase1_records()
    mismatched = (phase1_rows[0] | {"auc": "0.7000011"}, phase1_rows[1])
    with np.testing.assert_raises_regex(ValueError, "Phase-1 anchor mismatch"):
        validate_phase1_anchors(endpoint_aucs, mismatched)


def test_phase1_anchor_requires_a_committed_counterpart_for_every_endpoint() -> None:
    endpoint_aucs, phase1_rows = _phase1_records()
    with np.testing.assert_raises_regex(ValueError, "missing committed Phase-1 counterpart"):
        validate_phase1_anchors(endpoint_aucs, phase1_rows[:1])


def _write_phase1_anchor(path: Path, source: str, target: str, auc: float) -> None:
    path.write_text(
        "component,source,target,auc\n"
        f"zeroshot,{source},{target},{auc}\n"
    )


@pytest.fixture(scope="module")
def reportable_result() -> TraceResult:
    return TraceResult(
        draws=(
            DrawRecord(
                draw_seed=None,
                k=0,
                fold=0,
                held_out_sites=("E",),
                arm="warm",
                source="SOURCE",
                target="TARGET",
                origin="source",
                cohort="SOURCE",
                case_id="S00",
                label=1,
                site="X",
            ),
        ),
        predictions=(
            PredictionRecord(
                draw_seed=None,
                k=0,
                fold=0,
                held_out_sites=("E",),
                arm="warm",
                source="SOURCE",
                target="TARGET",
                case_id="T00",
                label=1,
                score=0.25,
            ),
        ),
        aucs=(
            AucRecord(
                draw_seed=None,
                k=0,
                arm="warm",
                source="SOURCE",
                target="TARGET",
                raw_auc=0.7,
                rank_auc=0.6,
                rank_gap=0.1,
                rank_diverged=True,
            ),
        ),
    )


@pytest.fixture
def report_output(
    tmp_path: Path,
    reportable_result: TraceResult,
) -> Path:
    phase1 = tmp_path / "phase1.csv"
    _write_phase1_anchor(phase1, "SOURCE", "TARGET", 0.7)
    out = tmp_path / "report"
    write_reportable_results(reportable_result, phase1, out)
    return out


def test_reportable_writer_emits_all_three_tables(report_output: Path) -> None:
    assert {path.name for path in report_output.iterdir()} == {
        "e1_draws.csv",
        "e1_predictions.csv",
        "e1_results.csv",
    }


def test_reportable_writer_serializes_the_tidy_result_schema(report_output: Path) -> None:
    with (report_output / "e1_results.csv").open(newline="") as handle:
        result_rows = tuple(csv.DictReader(handle))
    assert tuple(result_rows[0]) == (
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
    )
    assert result_rows[0] == {
        "experiment": "E1",
        "source": "SOURCE",
        "target": "TARGET",
        "base": "single",
        "arm": "warm",
        "k": "0",
        "draw_seed": "",
        "auc": "0.7",
        "rank_auc": "0.6",
        "rank_gap": "0.1",
        "rank_diverged": "True",
    }


def test_reportable_writer_serializes_auditable_training_rows(
    report_output: Path,
) -> None:
    with (report_output / "e1_draws.csv").open(newline="") as handle:
        draw_rows = tuple(csv.DictReader(handle))

    assert draw_rows[0] == {
        "draw_seed": "",
        "k": "0",
        "fold": "0",
        "held_out_sites": "('E',)",
        "arm": "warm",
        "source": "SOURCE",
        "target": "TARGET",
        "origin": "source",
        "cohort": "SOURCE",
        "case_id": "S00",
        "label": "1",
        "site": "X",
    }


def test_reportable_writer_serializes_auditable_prediction_rows(
    report_output: Path,
) -> None:
    with (report_output / "e1_predictions.csv").open(newline="") as handle:
        prediction_rows = tuple(csv.DictReader(handle))

    assert prediction_rows[0] == {
        "draw_seed": "",
        "k": "0",
        "fold": "0",
        "held_out_sites": "('E',)",
        "arm": "warm",
        "source": "SOURCE",
        "target": "TARGET",
        "case_id": "T00",
        "label": "1",
        "score": "0.25",
    }


def test_reportable_writer_creates_no_output_when_an_anchor_fails(
    tmp_path: Path,
    reportable_result: TraceResult,
) -> None:
    phase1 = tmp_path / "phase1.csv"
    _write_phase1_anchor(phase1, "SOURCE", "TARGET", 0.0)
    out = tmp_path / "report"

    with np.testing.assert_raises_regex(ValueError, "Phase-1 anchor mismatch"):
        write_reportable_results(reportable_result, phase1, out)

    assert not out.exists()


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
    assert rank_auc_diverged(0.70, 0.6899999999995) is True
    assert rank_auc_diverged(0.70, 0.689999) is True
