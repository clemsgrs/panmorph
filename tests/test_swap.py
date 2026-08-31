import numpy as np
import pytest

from panmorph.data import Cohort
from panmorph.swap import (
    SWAP_BUDGETS,
    SWAP_DRAW_IDS,
    SWAP_TARGET_SHARES,
    ConditionalEquivalence,
    build_target_reference,
    conditional_equivalence_interval,
    conditional_source_case_equivalence,
    sample_prevalence_prefix,
    trace_swap_cell,
)


def _cohort(name: str = "COAD") -> Cohort:
    labels = np.asarray([1] * 4 + [0] * 16)
    return Cohort(
        name=name,
        X=np.arange(40, dtype=np.float32).reshape(20, 2),
        y=labels,
        sites=np.asarray(["A", "B"] * 10),
        case_ids=np.asarray([f"{name}-{index:02d}" for index in range(20)]),
    )


def test_registered_swap_grid_has_three_budgets_eleven_mixtures_and_twenty_draws() -> None:
    assert SWAP_BUDGETS == (50, 100, 200)
    assert SWAP_TARGET_SHARES == (0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
    assert SWAP_DRAW_IDS == (
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
        10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    )


def test_prevalence_prefix_has_exact_full_cohort_prevalence_counts() -> None:
    cohort = _cohort()

    selected = sample_prevalence_prefix(cohort, (), 10, draw_seed=7, fold=0)

    assert (sum(case.label for case in selected), len(selected)) == (2, 10)
    assert len({case.case_id for case in selected}) == 10


def test_prevalence_samples_are_nested_prefixes() -> None:
    cohort = _cohort()

    small = sample_prevalence_prefix(cohort, (), 5, draw_seed=7, fold=0)
    large = sample_prevalence_prefix(cohort, (), 10, draw_seed=7, fold=0)

    assert {case.case_id for case in small} < {case.case_id for case in large}


def test_target_prefix_excludes_held_out_sites() -> None:
    cohort = _cohort("STAD")

    selected = sample_prevalence_prefix(
        cohort, ("A",), 5, draw_seed=3, fold=1
    )

    assert {case.site for case in selected} == {"B"}
    assert (sum(case.label for case in selected), len(selected)) == (1, 5)


@pytest.fixture(scope="module")
def swap_trace():
    source_fixture = _cohort()
    source = Cohort(
        name=source_fixture.name,
        X=np.column_stack((2 * source_fixture.y - 1, np.arange(20) % 7)).astype(np.float32),
        y=source_fixture.y,
        sites=source_fixture.sites,
        case_ids=source_fixture.case_ids,
    )
    labels = np.tile(np.asarray([1, 0, 0, 0, 0]), 10)
    target = Cohort(
        name="STAD",
        X=np.column_stack((2 * labels - 1, np.arange(50) % 7)).astype(np.float32),
        y=labels,
        sites=np.repeat(np.asarray(["A", "B", "C", "D", "E"]), 10),
        case_ids=np.asarray([f"STAD-{index:02d}" for index in range(50)]),
    )

    return trace_swap_cell(
        source, target, budget=10, target_share=50, draw_seed=4
    ), target


def test_swap_cell_evaluates_the_complete_site_grouped_target_oof_cohort(swap_trace) -> None:
    result, target = swap_trace

    assert len(result.predictions) == 50
    assert {row.case_id for row in result.predictions} == set(target.case_ids)
    assert len({row.fold for row in result.predictions}) == 5


def test_swap_cell_records_its_source_and_target_direction(swap_trace) -> None:
    result, _ = swap_trace

    assert {(row.source, row.target) for row in result.draws} == {("COAD", "STAD")}
    assert {(row.source, row.target) for row in result.predictions} == {("COAD", "STAD")}
    assert (result.auc.source, result.auc.target) == ("COAD", "STAD")


def test_swap_cell_reports_raw_pooled_oof_auc(swap_trace) -> None:
    result, _ = swap_trace

    assert result.auc.raw_auc == pytest.approx(1.0)


def test_swap_cell_holds_the_mixture_counts_fixed_in_every_fold(swap_trace) -> None:
    result, _ = swap_trace

    for fold in range(5):
        rows = [row for row in result.draws if row.fold == fold]
        assert sum(row.origin == "source" for row in rows) == 5
        assert sum(row.origin == "target" for row in rows) == 5


def test_swap_cell_excludes_held_out_target_sites_from_training(swap_trace) -> None:
    result, _ = swap_trace

    for fold in range(5):
        rows = [row for row in result.draws if row.fold == fold]
        held_out = next(row.held_out_sites for row in result.predictions if row.fold == fold)
        assert {
            row.site for row in rows if row.origin == "target"
        }.isdisjoint(held_out)


def test_conditional_average_source_case_equivalence_uses_target_case_curve() -> None:
    curve = {0.0: 0.50, 50.0: 0.60, 100.0: 0.70, 200.0: 0.80, 300.0: 0.85}

    positive = conditional_source_case_equivalence(0.65, 50, 50, curve)
    negative = conditional_source_case_equivalence(0.55, 100, 50, curve)

    assert positive.value == pytest.approx(0.5)
    assert positive.defined and not positive.censored
    assert negative.value == pytest.approx(-1.5)
    assert negative.defined and not negative.censored


def test_target_only_equivalence_is_undefined() -> None:
    curve = {0.0: 0.50, 100.0: 0.70, 300.0: 0.80}

    target_only = conditional_source_case_equivalence(0.70, 100, 0, curve)

    assert target_only.value is None
    assert not target_only.defined and not target_only.censored


def test_nonattained_equivalence_is_right_censored() -> None:
    curve = {0.0: 0.50, 100.0: 0.70, 300.0: 0.80}

    censored = conditional_source_case_equivalence(0.90, 50, 50, curve)

    assert censored.value is None
    assert censored.defined and censored.censored
    assert censored.censor_at == pytest.approx(5.0)


def test_conditional_equivalence_interval_preserves_a_censored_upper_bound() -> None:
    interval = conditional_equivalence_interval((
        ConditionalEquivalence(-1.0, True, False, None),
        ConditionalEquivalence(0.5, True, False, None),
        ConditionalEquivalence(2.0, True, False, None),
        ConditionalEquivalence(None, True, True, 5.0),
    ))

    assert interval.lower == -1.0
    assert interval.upper == 5.0
    assert not interval.lower_censored
    assert interval.upper_censored


def test_target_reference_combines_few_label_and_swap_target_only_coordinates() -> None:
    points = build_target_reference(
        few_label_cold={0: 0.50, 3: 0.58, 10: 0.68, "all": 0.82},
        target_prevalence=0.20,
        all_case_coordinate=120.0,
        swap_target_only={60: 0.72, 100: 0.79},
    )

    assert [(point.cases, point.origin, point.auc) for point in points] == [
        (0.0, "few-label", 0.50),
        (15.0, "few-label", 0.58),
        (50.0, "few-label", 0.68),
        (60.0, "swap", 0.72),
        (100.0, "swap", 0.79),
        (120.0, "few-label", 0.82),
    ]
