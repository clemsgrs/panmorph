import csv
import json
from pathlib import Path

import numpy as np
import pytest
import panmorph.e1_runner as runner_module

from panmorph.data import Cohort
from panmorph.e1_runner import (
    rebuild_e1_reports,
    run_e1_bundle,
    validate_e1_bundle,
)


def _quick_cohorts() -> dict[str, Cohort]:
    cohorts = {}
    for cohort_index, name in enumerate(("COAD", "STAD")):
        labels = np.tile(np.asarray([1, 0, 0, 0]), 25)
        cohorts[name] = Cohort(
            name=name,
            X=np.column_stack(
                (
                    2 * labels - 1 + np.arange(100) / 1_000,
                    (np.arange(100) + cohort_index) % 7,
                )
            ).astype(np.float32),
            y=labels,
            sites=np.repeat(np.asarray(["A", "B", "C", "D", "E"]), 20),
            case_ids=np.asarray([f"{name}-{index:03d}" for index in range(100)]),
        )
    return cohorts


def test_quick_runner_emits_complete_non_reportable_bundle(tmp_path: Path) -> None:
    out = tmp_path / "bundle"

    run_e1_bundle(out, profile="quick", cohorts=_quick_cohorts(), workers=1)

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["schema_version"] == "panmorph.e1.bundle/v1"
    assert manifest["profile"] == "quick"
    assert manifest["reportable"] is False
    assert manifest["status"] == "complete"
    assert manifest["configuration"]["rungs"] == [0, 10, "all"]
    assert manifest["configuration"]["draw_ids"] == [0]
    assert manifest["configuration"]["bootstrap_replicates"] == 50
    assert manifest["configuration"]["permutations"] == 9
    assert manifest["resources"]["workers"] == 1
    assert manifest["features"]["hashes"] == {
        "COAD": "lxbzb8rd",
        "STAD": "oowdp902",
    }

    expected = {
        "e1_draws.csv",
        "e1_predictions.csv",
        "e1_aucs.csv",
        "e1_summaries.csv",
        "e1_equivalence.csv",
        "e1_confirmatory_null.csv",
        "e1_value.png",
        "e1_value.pdf",
        "e1_lift.png",
        "e1_lift.pdf",
    }
    assert expected <= {path.name for path in out.iterdir()}
    with (out / "e1_confirmatory_null.csv").open(newline="") as handle:
        assert len(tuple(csv.DictReader(handle))) == 9


def test_bundle_validation_rejects_a_schema_failure(tmp_path: Path) -> None:
    out = tmp_path / "bundle"
    run_e1_bundle(out, profile="quick", cohorts=_quick_cohorts(), workers=1)
    (out / "e1_aucs.csv").write_text("wrong,column\n1,2\n")

    with pytest.raises(ValueError, match="invalid schema for e1_aucs.csv"):
        validate_e1_bundle(out)


def test_reports_can_be_rebuilt_and_numeric_corruption_is_rejected(tmp_path: Path) -> None:
    out = tmp_path / "bundle"
    run_e1_bundle(out, profile="quick", cohorts=_quick_cohorts(), workers=1)
    summaries = out / "e1_summaries.csv"
    with summaries.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = tuple(rows[0])
    rows[0]["lift"] = "999"
    with summaries.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="not reproducible from bundle inputs"):
        validate_e1_bundle(out)

    rebuild_e1_reports(out)
    validate_e1_bundle(out)


def test_compatible_resume_uses_complete_cell_and_permutation_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "bundle"
    cohorts = _quick_cohorts()
    run_e1_bundle(out, profile="quick", cohorts=cohorts, workers=1)
    expected = (out / "e1_predictions.csv").read_bytes()

    def unexpected_execution(*_args, **_kwargs):
        raise AssertionError("a complete checkpoint was recomputed")

    monkeypatch.setattr(runner_module, "trace_paired_cell", unexpected_execution)
    run_e1_bundle(out, profile="quick", cohorts=cohorts, workers=1)

    assert (out / "e1_predictions.csv").read_bytes() == expected


def test_resume_recomputes_incomplete_cell_and_permutation_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "bundle"
    cohorts = _quick_cohorts()
    run_e1_bundle(out, profile="quick", cohorts=cohorts, workers=1)
    checkpoint = next(
        path
        for path in (out / "checkpoints" / "cells").glob("*/draws.csv")
        if len(path.read_text().splitlines()) > 1
    )
    checkpoint.write_text(",".join(runner_module.DRAW_FIELDS) + "\n")
    original = runner_module.trace_paired_cell
    calls = 0

    def recording_execution(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(runner_module, "trace_paired_cell", recording_execution)
    run_e1_bundle(out, profile="quick", cohorts=cohorts, workers=1)

    assert calls == 1
    validate_e1_bundle(out)

    permutation = next((out / "checkpoints" / "permutations").glob("*.csv"))
    with permutation.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = tuple(rows[0])
    rows[0]["null_mean_lift"] = "nan"
    with permutation.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    original_null = runner_module.evaluate_confirmatory_null
    null_calls = 0

    def recording_null(*args, **kwargs):
        nonlocal null_calls
        null_calls += 1
        return original_null(*args, **kwargs)

    monkeypatch.setattr(runner_module, "evaluate_confirmatory_null", recording_null)
    run_e1_bundle(out, profile="quick", cohorts=cohorts, workers=1)

    assert null_calls == 1
    validate_e1_bundle(out)


def test_resume_refuses_changed_data_identity(tmp_path: Path) -> None:
    out = tmp_path / "bundle"
    cohorts = _quick_cohorts()
    run_e1_bundle(out, profile="quick", cohorts=cohorts, workers=1)
    changed = dict(cohorts)
    changed_stad = changed["STAD"]
    changed_X = changed_stad.X.copy()
    changed_X[0, 0] += 0.25
    changed["STAD"] = Cohort(
        changed_stad.name, changed_X, changed_stad.y, changed_stad.sites,
        changed_stad.case_ids,
    )

    with pytest.raises(ValueError, match="incompatible E1 partial results"):
        run_e1_bundle(out, profile="quick", cohorts=changed, workers=1)


def test_parallel_worker_count_does_not_change_result_tables(tmp_path: Path) -> None:
    cohorts = _quick_cohorts()
    serial = tmp_path / "serial"
    parallel = tmp_path / "parallel"

    run_e1_bundle(serial, profile="quick", cohorts=cohorts, workers=1)
    run_e1_bundle(parallel, profile="quick", cohorts=cohorts, workers=2)

    for name in (
        "e1_draws.csv", "e1_predictions.csv", "e1_aucs.csv",
        "e1_summaries.csv", "e1_equivalence.csv", "e1_confirmatory_null.csv",
    ):
        assert (serial / name).read_bytes() == (parallel / name).read_bytes()
