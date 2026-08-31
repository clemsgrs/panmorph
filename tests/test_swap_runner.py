import csv
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from panmorph.data import Cohort
from panmorph.e1_runner import run_e1_bundle
from panmorph.swap import SwapAucRecord, SwapPredictionRecord
from panmorph.swap_runner import (
    estimate_swap_cells,
    rebuild_swap_figures,
    run_swap_bundle,
    validate_swap_bundle,
)


def _cohorts() -> dict[str, Cohort]:
    cohorts = {}
    for offset, name in enumerate(("COAD", "STAD")):
        labels = np.tile(np.asarray([1, 0, 0, 0]), 25)
        cohorts[name] = Cohort(
            name=name,
            X=np.column_stack((2 * labels - 1, (np.arange(100) + offset) % 7)).astype(
                np.float32
            ),
            y=labels,
            sites=np.repeat(np.asarray(["A", "B", "C", "D", "E"]), 20),
            case_ids=np.asarray([f"{name}-{index:03d}" for index in range(100)]),
        )
    return cohorts


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope="module")
def swap_bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("swap") / "bundle"
    cohorts = _cohorts()
    run_e1_bundle(out, profile="quick", cohorts=cohorts, workers=1)
    run_swap_bundle(
        out,
        cohorts=cohorts,
        budgets=(10, 20),
        target_shares=(0, 50, 100),
        draw_ids=(0, 1),
        n_bootstraps=20,
        workers=1,
    )
    validate_swap_bundle(out)
    return out


def test_swap_runner_records_exploratory_configuration(swap_bundle: Path) -> None:
    manifest = json.loads((swap_bundle / "manifest.json").read_text())

    assert manifest["downstream"]["swap"] == {
        "schema_version": "panmorph.swap.bundle/v2",
        "status": "complete",
        "directions": [
            {"source": "COAD", "target": "STAD"},
            {"source": "STAD", "target": "COAD"},
        ],
        "budgets": [10, 20],
        "target_shares": [0, 50, 100],
        "draw_ids": [0, 1],
        "bootstrap_replicates": 20,
        "confirmatory": False,
        "permutations": 0,
        "exploratory": True,
    }


def test_swap_runner_emits_the_complete_configured_grid(swap_bundle: Path) -> None:
    assert len(_rows(swap_bundle / "swap_draws.csv")) == 1_800
    assert len(_rows(swap_bundle / "swap_aucs.csv")) == 24
    assert len(_rows(swap_bundle / "swap_summaries.csv")) == 12
    assert len(_rows(swap_bundle / "swap_predictions.csv")) == 2_400

    for name in (
        "swap_draws.csv", "swap_predictions.csv", "swap_aucs.csv",
        "swap_summaries.csv", "swap_reference.csv", "swap_equivalence.csv",
    ):
        with (swap_bundle / name).open(newline="") as handle:
            assert tuple(csv.DictReader(handle).fieldnames or ())[:2] == (
                "source", "target",
            )


def test_swap_runner_preserves_undefined_target_only_equivalence(swap_bundle: Path) -> None:
    equivalence = _rows(swap_bundle / "swap_equivalence.csv")

    assert len(equivalence) == 12
    assert all(row["defined"] == "False" for row in equivalence if row["target_share"] == "100")


def test_swap_runner_records_the_monotone_target_reference(swap_bundle: Path) -> None:
    assert _rows(swap_bundle / "swap_reference.csv") == [
        {"source": "COAD", "target": "STAD", "cases": "0.0", "origin": "e1", "raw_auc": "0.5", "monotone_auc": "0.5"},
        {"source": "COAD", "target": "STAD", "cases": "10.0", "origin": "swap", "raw_auc": "1.0", "monotone_auc": "1.0"},
        {"source": "COAD", "target": "STAD", "cases": "20.0", "origin": "swap", "raw_auc": "1.0", "monotone_auc": "1.0"},
        {"source": "COAD", "target": "STAD", "cases": "40.0", "origin": "e1", "raw_auc": "1.0", "monotone_auc": "1.0"},
        {"source": "COAD", "target": "STAD", "cases": "80.0", "origin": "e1", "raw_auc": "1.0", "monotone_auc": "1.0"},
        {"source": "STAD", "target": "COAD", "cases": "0.0", "origin": "e1", "raw_auc": "0.5", "monotone_auc": "0.5"},
        {"source": "STAD", "target": "COAD", "cases": "10.0", "origin": "swap", "raw_auc": "1.0", "monotone_auc": "1.0"},
        {"source": "STAD", "target": "COAD", "cases": "20.0", "origin": "swap", "raw_auc": "1.0", "monotone_auc": "1.0"},
        {"source": "STAD", "target": "COAD", "cases": "40.0", "origin": "e1", "raw_auc": "1.0", "monotone_auc": "1.0"},
        {"source": "STAD", "target": "COAD", "cases": "80.0", "origin": "e1", "raw_auc": "1.0", "monotone_auc": "1.0"},
    ]


def test_swap_runner_bootstraps_conditional_equivalence(swap_bundle: Path) -> None:
    row = next(
        row for row in _rows(swap_bundle / "swap_equivalence.csv")
        if (row["source"], row["target"], row["budget"], row["target_share"])
        == ("COAD", "STAD", "10", "50")
    )

    assert row == {
        "source": "COAD", "target": "STAD", "budget": "10",
        "target_share": "50", "source_cases": "5",
        "target_cases": "5", "defined": "True",
        "average_source_case_equivalence": "1.0", "censored": "False",
        "censor_at": "", "ci_lower": "1.0", "ci_upper": "1.0",
        "ci_lower_censored": "False", "ci_upper_censored": "False",
    }


def test_swap_runner_emits_all_integrated_tables_and_figures(swap_bundle: Path) -> None:
    for name in (
        "swap_draws.csv",
        "swap_predictions.csv",
        "swap_aucs.csv",
        "swap_summaries.csv",
        "swap_reference.csv",
        "swap_equivalence.csv",
        "swap_auc.png",
        "swap_auc.pdf",
        "swap_equivalence.png",
        "swap_equivalence.pdf",
    ):
        assert (swap_bundle / name).stat().st_size > 0


def test_swap_figures_can_be_rebuilt_from_directional_tables(
    swap_bundle: Path, tmp_path: Path
) -> None:
    out = tmp_path / "rebuild"
    shutil.copytree(swap_bundle, out)
    for name in ("swap_auc.png", "swap_auc.pdf", "swap_equivalence.png", "swap_equivalence.pdf"):
        (out / name).unlink()

    rebuild_swap_figures(out)

    assert (out / "swap_auc.png").read_bytes().startswith(b"\x89PNG")
    assert (out / "swap_auc.pdf").read_bytes().startswith(b"%PDF")
    assert (out / "swap_equivalence.png").read_bytes().startswith(b"\x89PNG")
    assert (out / "swap_equivalence.pdf").read_bytes().startswith(b"%PDF")


def test_swap_validation_rejects_incomplete_oof_coverage(
    swap_bundle: Path, tmp_path: Path
) -> None:
    out = tmp_path / "corrupt"
    shutil.copytree(swap_bundle, out)
    predictions = out / "swap_predictions.csv"
    lines = predictions.read_text().splitlines()
    predictions.write_text("\n".join(lines[:-1]) + "\n")
    with pytest.raises(ValueError, match="complete target OOF coverage"):
        validate_swap_bundle(out)


def test_swap_validation_rejects_corrupt_derived_values(
    swap_bundle: Path, tmp_path: Path
) -> None:
    out = tmp_path / "corrupt"
    shutil.copytree(swap_bundle, out)
    summaries = out / "swap_summaries.csv"
    rows = _rows(summaries)
    rows[0]["raw_auc"] = "0.123"
    with summaries.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="not reproducible"):
        validate_swap_bundle(out)


def test_swap_estimate_bootstraps_raw_and_rank_auc() -> None:
    predictions = tuple(
        SwapPredictionRecord(
            "COAD", "STAD", 10, 50, draw, fold, (f"S{fold}",),
            case_id, label, score
        )
        for draw in (0, 1)
        for fold, cases in enumerate((
            (("p1", 1, 0.9), ("n1", 0, 0.1)),
            (("p2", 1, 0.8), ("n2", 0, 0.2)),
        ))
        for case_id, label, score in cases
    )
    aucs = tuple(
        SwapAucRecord("COAD", "STAD", 10, 50, draw, 1.0, 1.0, 0.0, False)
        for draw in (0, 1)
    )

    (estimate,) = estimate_swap_cells(
        predictions, aucs, draw_ids=(0, 1), n_bootstraps=20
    )

    assert (estimate.raw.point, estimate.raw.lower, estimate.raw.upper) == (1.0, 1.0, 1.0)
    assert (estimate.rank.point, estimate.rank.lower, estimate.rank.upper) == (1.0, 1.0, 1.0)
    assert estimate.rank_gap == 0.0
    assert not estimate.rank_diverged


def test_swap_estimates_keep_opposite_directions_separate() -> None:
    predictions = tuple(
        SwapPredictionRecord(
            source, target, 10, 50, 0, fold, (f"S{fold}",),
            f"{target}-{case_id}", label, score,
        )
        for source, target, scores in (
            ("COAD", "STAD", (0.9, 0.1, 0.8, 0.2)),
            ("STAD", "COAD", (0.1, 0.9, 0.2, 0.8)),
        )
        for fold, (case_id, label, score) in enumerate(zip(
            ("p1", "n1", "p2", "n2"), (1, 0, 1, 0), scores,
        ))
    )
    aucs = (
        SwapAucRecord("COAD", "STAD", 10, 50, 0, 1.0, 1.0, 0.0, False),
        SwapAucRecord("STAD", "COAD", 10, 50, 0, 0.0, 0.0, 0.0, False),
    )

    estimates = estimate_swap_cells(
        predictions, aucs, draw_ids=(0,), n_bootstraps=20
    )

    assert [
        (estimate.source, estimate.target, estimate.raw.point)
        for estimate in estimates
    ] == [
        ("COAD", "STAD", 1.0),
        ("STAD", "COAD", 0.0),
    ]
