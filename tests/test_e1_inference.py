import csv
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import panmorph.e1 as e1_module
from joblib import parallel_config

from experiments.run_e1 import read_prediction_records, write_registered_inference
from panmorph.data import Cohort
from panmorph.e1 import (
    BOOTSTRAP_REPLICATES,
    CONFIRMATORY_CELL,
    ConfirmatoryResult,
    E1Inference,
    PERMUTATION_COUNT,
    PredictionRecord,
    AucRecord,
    E1_DRAW_IDS,
    TraceResult,
    empirical_superiority_p,
    estimate_e1_cell,
    estimate_e1_matrix,
    is_confirmatory_cell,
    local_positive_equivalence,
    run_confirmatory_test,
    source_label_permutations,
    stratified_bootstrap_indices,
    summarize_equivalence_bootstrap,
)


def test_registered_bootstrap_schedule_is_stratified_and_keyed() -> None:
    labels = np.asarray([1, 0, 1, 0, 0])

    schedule = stratified_bootstrap_indices(labels, seed=17, key=("STAD",))
    stratified_bootstrap_indices(labels, seed=17, key=("COAD",))
    after_unrelated_cell = stratified_bootstrap_indices(
        labels, seed=17, key=("STAD",)
    )

    assert BOOTSTRAP_REPLICATES == 2_000
    assert schedule.shape == (2_000, 5)
    assert np.all(np.sum(labels[schedule] == 1, axis=1) == 2)
    assert np.all(np.sum(labels[schedule] == 0, axis=1) == 3)
    assert np.array_equal(schedule, after_unrelated_cell)
    assert len(set(schedule[0])) < len(schedule[0])


def test_stored_prediction_reader_restores_the_issue_six_schema(tmp_path: Path) -> None:
    path = tmp_path / "e1_predictions.csv"
    path.write_text(
        "draw_seed,k,fold,held_out_sites,arm,source,target,case_id,label,score\n"
        '7,10,2,"(\'A\', \'B\')",warm,COAD,STAD,P01,1,0.75\n'
    )

    (record,) = read_prediction_records(path)

    assert record == PredictionRecord(
        draw_seed=7,
        k=10,
        fold=2,
        held_out_sites=("A", "B"),
        arm="warm",
        source="COAD",
        target="STAD",
        case_id="P01",
        label=1,
        score=0.75,
    )


def _cell_predictions() -> tuple[PredictionRecord, ...]:
    labels = (0, 0, 1, 1)
    scores = {
        (0, "cold"): (0.1, 0.8, 0.4, 0.9),
        (0, "warm"): (0.2, 0.7, 0.6, 0.8),
        (1, "cold"): (0.4, 0.3, 0.2, 0.1),
        (1, "warm"): (0.4, 0.3, 0.8, 0.7),
    }
    return tuple(
        PredictionRecord(
            draw_seed=draw,
            k=10,
            fold=index % 2,
            held_out_sites=(str(index % 2),),
            arm=arm,
            source="SOURCE" if arm == "warm" else "target-only",
            target="TARGET",
            case_id=f"P{index}",
            label=label,
            score=score,
        )
        for draw in E1_DRAW_IDS
        for arm in ("cold", "warm")
        for index, (label, score) in enumerate(zip(labels, scores[(draw % 2, arm)]))
    )


def test_cell_estimate_pairs_patients_and_fixed_draws_for_raw_auc_intervals() -> None:
    estimate = estimate_e1_cell(
        _cell_predictions(), "SOURCE", "TARGET", 10, seed=23
    )

    assert estimate.n_draws == 20
    assert estimate.warm.point == 0.875
    assert estimate.cold.point == 0.375
    assert estimate.lift.point == 0.5
    assert len(estimate.bootstrap_warm) == 2_000
    assert estimate.bootstrap_warm[:4].tolist() == [1.0, 0.875, 1.0, 0.75]
    assert estimate.bootstrap_cold[:4].tolist() == [0.5, 0.375, 0.5, 0.25]
    assert np.all(estimate.bootstrap_lift == 0.5)
    assert (estimate.warm.lower, estimate.warm.upper) == (0.5, 1.0)
    assert (estimate.cold.lower, estimate.cold.upper) == (0.0, 0.5)
    assert (estimate.lift.lower, estimate.lift.upper) == (0.5, 0.5)
    assert (estimate.rank_warm, estimate.rank_cold, estimate.rank_lift) == (
        1.0,
        0.5,
        0.5,
    )
    assert estimate.rank_diverged is True


def test_cell_estimate_rejects_an_incomplete_draw_schedule() -> None:
    incomplete = tuple(
        record for record in _cell_predictions() if record.draw_seed != E1_DRAW_IDS[-1]
    )

    with pytest.raises(ValueError, match="complete fixed draw schedule"):
        estimate_e1_cell(incomplete, "SOURCE", "TARGET", 10)


def test_equivalence_uses_equal_weight_isotonic_curve_and_first_linear_crossing() -> None:
    result = local_positive_equivalence(
        0.70,
        {0: 0.50, 3: 0.62, 5: 0.60, 10: 0.80, "all": 0.85},
        all_coordinate=12,
    )

    # Equal-weight isotonic regression pools k=3 and k=5 at 0.61, so the first
    # crossing of 0.70 lies 9/19 of the way from k=5 to k=10.
    assert result.value == pytest.approx(5 + 5 * 9 / 19)
    assert result.censored is False
    assert result.censor_at is None


def test_equivalence_zero_rule_and_non_attainment_are_explicit() -> None:
    cold_curve = {0: 0.50, 3: 0.58, "all": 0.65}

    zero = local_positive_equivalence(0.50, cold_curve, all_coordinate=8)
    censored = local_positive_equivalence(0.70, cold_curve, all_coordinate=8)

    assert (zero.value, zero.censored, zero.censor_at) == (0.0, False, None)
    assert (censored.value, censored.censored, censored.censor_at) == (
        None,
        True,
        8.0,
    )


def test_equivalence_is_recomputed_in_each_bootstrap_and_scaled_as_an_average() -> None:
    foreign_bootstrap = np.tile(np.asarray([0.60, 0.80]), 1_000)
    cold_bootstrap = {
        0: np.full(2_000, 0.50),
        10: np.full(2_000, 0.70),
        "all": np.full(2_000, 0.90),
    }

    summary = summarize_equivalence_bootstrap(
        foreign_only_auc=0.70,
        cold_curve={0: 0.50, 10: 0.70, "all": 0.90},
        foreign_bootstrap=foreign_bootstrap,
        cold_bootstrap=cold_bootstrap,
        all_coordinate=20,
        source_case_count=100,
    )

    assert summary.point.value == 10.0
    assert summary.bootstrap[0].value == 5.0
    assert summary.bootstrap[1].value == pytest.approx(15.0)
    assert summary.interval.lower == 5.0
    assert summary.interval.upper == pytest.approx(15.0)
    assert summary.average_source_case.point.value == 0.10
    assert summary.average_source_case.bootstrap[0].value == 0.05


def test_bootstrap_equivalence_interval_preserves_censored_upper_endpoint() -> None:
    summary = summarize_equivalence_bootstrap(
        foreign_only_auc=0.80,
        cold_curve={0: 0.50, "all": 0.70},
        foreign_bootstrap=np.full(2_000, 0.80),
        cold_bootstrap={
            0: np.full(2_000, 0.50),
            "all": np.full(2_000, 0.70),
        },
        all_coordinate=8,
        source_case_count=20,
    )

    assert summary.point.censored is True
    assert (summary.interval.upper, summary.interval.upper_censored) == (8.0, True)
    assert (
        summary.average_source_case.interval.upper,
        summary.average_source_case.interval.upper_censored,
    ) == (0.4, True)


def test_only_single_source_coad_to_stad_at_ten_is_confirmatory() -> None:
    assert CONFIRMATORY_CELL == ("COAD", "STAD", "single", 10)
    assert is_confirmatory_cell("COAD", "STAD", 10) is True
    assert is_confirmatory_cell("COAD+UCEC", "STAD", 10) is False
    assert is_confirmatory_cell("COAD", "STAD", 5) is False
    assert is_confirmatory_cell("STAD", "COAD", 10) is False


def test_source_label_permutations_preserve_labels_and_are_keyed() -> None:
    labels = np.asarray([1, 0, 0, 1, 0])

    schedule = source_label_permutations(labels, "COAD", seed=31)
    source_label_permutations(labels, "UCEC", seed=31)
    after_unrelated_source = source_label_permutations(labels, "COAD", seed=31)

    assert PERMUTATION_COUNT == 999
    assert schedule.shape == (999, 5)
    assert np.all(np.sum(schedule == 1, axis=1) == 2)
    assert np.all(np.sum(schedule == 0, axis=1) == 3)
    assert np.array_equal(schedule, after_unrelated_source)


def test_confirmatory_p_value_uses_one_sided_plus_one_convention() -> None:
    assert empirical_superiority_p(0.5, np.asarray([0.6, 0.4, 0.5])) == 0.75


def test_confirmatory_null_reuses_one_source_shuffle_across_all_draws(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels = (0, 0, 1, 1)
    observed = tuple(
        PredictionRecord(
            draw_seed=draw,
            k=10,
            fold=index,
            held_out_sites=(str(index),),
            arm=arm,
            source="COAD" if arm == "warm" else "target-only",
            target="STAD",
            case_id=f"P{index}",
            label=label,
            score=(0.1, 0.2, 0.8, 0.9)[index]
            if arm == "warm"
            else (0.1, 0.8, 0.2, 0.9)[index],
        )
        for draw in E1_DRAW_IDS
        for arm in ("warm", "cold")
        for index, label in enumerate(labels)
    )
    source = Cohort(
        name="COAD",
        X=np.arange(8, dtype=np.float32).reshape(4, 2),
        y=np.asarray(labels),
        sites=np.asarray(["A", "A", "B", "B"]),
        case_ids=np.asarray([f"S{i}" for i in range(4)]),
    )
    target = Cohort(
        name="STAD",
        X=np.arange(8, dtype=np.float32).reshape(4, 2),
        y=np.asarray(labels),
        sites=np.asarray(["A", "B", "C", "D"]),
        case_ids=np.asarray([f"P{i}" for i in range(4)]),
    )
    calls: list[tuple[int | None, tuple[int, ...]]] = []

    def fake_trace(
        shuffled_source: Cohort,
        _target: Cohort,
        k: int,
        draw_seed: int | None,
        arms: tuple[str, ...],
    ) -> TraceResult:
        calls.append((draw_seed, tuple(int(value) for value in shuffled_source.y)))
        auc = float(np.dot(shuffled_source.y, np.arange(1, 5))) / 10
        return TraceResult(
            (),
            (),
            (AucRecord(draw_seed, k, "warm", "COAD", "STAD", auc, auc, 0, False),),
        )

    monkeypatch.setattr(e1_module, "trace_paired_cell", fake_trace)

    serial = run_confirmatory_test(source, target, observed, seed=41, n_jobs=1)

    assert serial.n_permutations == 999
    assert len(serial.null_lifts) == 999
    for start in range(0, len(calls), 20):
        block = calls[start : start + 20]
        assert tuple(draw for draw, _ in block) == E1_DRAW_IDS
        assert len({shuffled for _, shuffled in block}) == 1

    calls.clear()
    with parallel_config(backend="threading"):
        parallel = run_confirmatory_test(source, target, observed, seed=41, n_jobs=2)
    assert np.array_equal(serial.null_lifts, parallel.null_lifts)
    assert serial.p_value == parallel.p_value


def test_inference_writer_assigns_a_p_value_only_to_the_confirmatory_cell(
    tmp_path: Path,
) -> None:
    base = estimate_e1_cell(_cell_predictions(), "SOURCE", "TARGET", 10, seed=23)
    confirmatory_cell = replace(
        base,
        source="COAD",
        target="STAD",
        confirmatory=True,
    )
    exploratory_cell = replace(
        base,
        source="UCEC",
        target="STAD",
        confirmatory=False,
    )
    confirmatory = ConfirmatoryResult(0.5, 0.01, True, 999, np.zeros(999))

    write_registered_inference(
        E1Inference((confirmatory_cell, exploratory_cell), ()), confirmatory, tmp_path
    )

    with (tmp_path / "e1_estimates.csv").open(newline="") as handle:
        rows = tuple(csv.DictReader(handle))
    assert [row["permutation_p"] for row in rows] == ["0.01", ""]
    with (tmp_path / "e1_permutation_null.csv").open(newline="") as handle:
        assert len(tuple(csv.DictReader(handle))) == 999


def test_matrix_inference_derives_equivalence_from_stored_predictions() -> None:
    labels = np.asarray([0] * 5 + [1] * 15)
    predictions = []
    for k, draws in ((0, (None,)), (10, E1_DRAW_IDS), ("all", (None,))):
        for draw in draws:
            for arm, source in (("warm", "COAD"), ("cold", "target-only")):
                for index, label in enumerate(labels):
                    if arm == "cold" and k == 0:
                        score = 0.5
                    else:
                        score = float(label + index / 100)
                    predictions.append(
                        PredictionRecord(
                            draw_seed=draw,
                            k=k,
                            fold=index % 5,
                            held_out_sites=(str(index % 5),),
                            arm=arm,
                            source=source,
                            target="STAD",
                            case_id=f"P{index:02d}",
                            label=int(label),
                            score=score,
                        )
                    )

    inference = estimate_e1_matrix(tuple(predictions), {"COAD": 100}, seed=13)

    assert len(inference.cells) == 3
    assert sum(cell.confirmatory for cell in inference.cells) == 1
    (equivalence,) = inference.equivalences
    assert equivalence.local_positive.point.value == 10.0
    assert equivalence.local_positive.average_source_case.point.value == 0.10
