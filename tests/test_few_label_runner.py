import csv
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import panmorph.few_label_runner as runner_module

from panmorph.data import Cohort
from panmorph.few_label_runner import (
    FewLabelProfile,
    _run_cells,
    rebuild_few_label_reports,
    run_few_label_bundle,
    validate_few_label_bundle,
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

    run_few_label_bundle(out, profile="quick", cohorts=_quick_cohorts(), workers=1)

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["schema_version"] == "panmorph.few-label.bundle/v1"
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
        "few_label_draws.csv",
        "few_label_folds.csv",
        "few_label_cohort_cases.csv",
        "few_label_source_bases.csv",
        "few_label_predictions.csv",
        "few_label_aucs.csv",
        "few_label_summaries.csv",
        "few_label_equivalence.csv",
        "few_label_confirmatory_null.csv",
        "few_label_value.png",
        "few_label_value.pdf",
        "few_label_lift.png",
        "few_label_lift.pdf",
    }
    assert expected <= {path.name for path in out.iterdir()}
    with (out / "few_label_draws.csv").open(newline="") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames or ()) == (
            "target", "k", "draw_seed", "fold", "case_id",
        )
    with (out / "few_label_folds.csv").open(newline="") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames or ()) == (
            "target", "fold", "held_out_sites",
        )
    with (out / "few_label_cohort_cases.csv").open(newline="") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames or ()) == (
            "cohort", "case_id", "label", "site",
        )
        assert next(reader) == {
            "cohort": "COAD", "case_id": "COAD-000", "label": "1", "site": "A",
        }
    with (out / "few_label_source_bases.csv").open(newline="") as handle:
        assert list(csv.DictReader(handle)) == [
            {"source": "COAD", "target": "STAD", "cohort": "COAD"},
            {"source": "STAD", "target": "COAD", "cohort": "STAD"},
        ]
    with (out / "few_label_confirmatory_null.csv").open(newline="") as handle:
        assert len(tuple(csv.DictReader(handle))) == 9


def test_runner_derives_inference_only_for_write_and_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0
    original = runner_module.estimate_few_label_matrix

    def recording_estimate(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(runner_module, "estimate_few_label_matrix", recording_estimate)

    run_few_label_bundle(
        tmp_path / "bundle", profile="quick", cohorts=_quick_cohorts(), workers=1
    )

    assert calls == 2


def test_bundle_validation_rejects_a_schema_failure(tmp_path: Path) -> None:
    out = tmp_path / "bundle"
    run_few_label_bundle(out, profile="quick", cohorts=_quick_cohorts(), workers=1)
    (out / "few_label_aucs.csv").write_text("wrong,column\n1,2\n")

    with pytest.raises(ValueError, match="invalid schema for few_label_aucs.csv"):
        validate_few_label_bundle(out)


def test_reports_can_be_rebuilt_and_numeric_corruption_is_rejected(tmp_path: Path) -> None:
    out = tmp_path / "bundle"
    run_few_label_bundle(out, profile="quick", cohorts=_quick_cohorts(), workers=1)
    summaries = out / "few_label_summaries.csv"
    with summaries.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = tuple(rows[0])
    rows[0]["lift"] = "999"
    with summaries.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="not reproducible from bundle inputs"):
        validate_few_label_bundle(out)

    rebuild_few_label_reports(out)
    validate_few_label_bundle(out)


def test_compatible_resume_uses_complete_cell_and_permutation_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "bundle"
    cohorts = _quick_cohorts()
    run_few_label_bundle(out, profile="quick", cohorts=cohorts, workers=1)
    expected = (out / "few_label_predictions.csv").read_bytes()

    def unexpected_execution(*_args, **_kwargs):
        raise AssertionError("a complete checkpoint was recomputed")

    monkeypatch.setattr(runner_module, "trace_paired_cell", unexpected_execution)
    run_few_label_bundle(out, profile="quick", cohorts=cohorts, workers=1)

    assert (out / "few_label_predictions.csv").read_bytes() == expected


def test_resume_recomputes_incomplete_cell_and_permutation_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "bundle"
    cohorts = _quick_cohorts()
    run_few_label_bundle(out, profile="quick", cohorts=cohorts, workers=1)
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
    run_few_label_bundle(out, profile="quick", cohorts=cohorts, workers=1)

    assert calls == 1
    validate_few_label_bundle(out)

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
    run_few_label_bundle(out, profile="quick", cohorts=cohorts, workers=1)

    assert null_calls == 1
    validate_few_label_bundle(out)


def test_resume_refuses_changed_data_identity(tmp_path: Path) -> None:
    out = tmp_path / "bundle"
    cohorts = _quick_cohorts()
    run_few_label_bundle(out, profile="quick", cohorts=cohorts, workers=1)
    changed = dict(cohorts)
    changed_stad = changed["STAD"]
    changed_X = changed_stad.X.copy()
    changed_X[0, 0] += 0.25
    changed["STAD"] = Cohort(
        changed_stad.name, changed_X, changed_stad.y, changed_stad.sites,
        changed_stad.case_ids,
    )

    with pytest.raises(ValueError, match="incompatible few-label partial results"):
        run_few_label_bundle(out, profile="quick", cohorts=changed, workers=1)


def test_parallel_worker_count_does_not_change_result_tables(tmp_path: Path) -> None:
    cohorts = _quick_cohorts()
    serial = tmp_path / "serial"
    parallel = tmp_path / "parallel"

    run_few_label_bundle(serial, profile="quick", cohorts=cohorts, workers=1)
    run_few_label_bundle(parallel, profile="quick", cohorts=cohorts, workers=2)

    for name in (
        "few_label_draws.csv", "few_label_predictions.csv", "few_label_aucs.csv",
        "few_label_summaries.csv", "few_label_equivalence.csv", "few_label_confirmatory_null.csv",
    ):
        assert (serial / name).read_bytes() == (parallel / name).read_bytes()


def test_parallel_cells_run_warm_zero_in_the_zero_shot_numerical_environment(
    tmp_path: Path,
) -> None:
    cohorts = {}
    dimensions = np.arange(1, 1_281)[None, :]
    for cohort_index, name in enumerate(("COAD", "STAD"), start=1):
        labels = np.tile(np.asarray([1, 0]), 5)
        rows = np.arange(1, 11)[:, None]
        cohorts[name] = Cohort(
            name=name,
            X=(
                np.sin(rows * dimensions / (37 + cohort_index))
                + labels[:, None] * 0.02
            ).astype(np.float32),
            y=labels,
            sites=np.repeat(np.asarray(["A", "B", "C", "D", "E"]), 2),
            case_ids=np.asarray([f"{name}-{index:02}" for index in range(10)]),
        )
    profile = FewLabelProfile(
        "quick", False, ("COAD", "STAD"), (("COAD", "STAD"),),
        (0,), (), 1, 1, 1,
    )

    result = _run_cells(tmp_path, profile, cohorts, workers=2)
    actual_by_case = {
        record.case_id: record.score
        for record in result.predictions if record.arm == "warm"
    }
    actual = tuple(
        round(actual_by_case[str(case_id)], 6)
        for case_id in cohorts["STAD"].case_ids
    )

    assert actual == (
        0.494334,
        0.000312,
        0.002762,
        0.172390,
        0.168031,
        0.989638,
        0.986182,
        0.754600,
        0.981660,
        0.008484,
    )


@pytest.fixture(scope="module")
def completed_quick_bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("validation") / "bundle"
    run_few_label_bundle(out, profile="quick", cohorts=_quick_cohorts(), workers=1)
    return out


def _copy_bundle(source: Path, destination: Path) -> Path:
    return Path(shutil.copytree(source, destination))


def _read_csv_rows(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return tuple(rows[0]), rows


def _write_csv_rows(
    path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, str]]
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_bundle_validation_rejects_duplicate_draw_grain(
    tmp_path: Path, completed_quick_bundle: Path
) -> None:
    out = _copy_bundle(completed_quick_bundle, tmp_path / "bundle")
    path = out / "few_label_draws.csv"
    fieldnames, rows = _read_csv_rows(path)
    rows.append(dict(rows[0]))
    _write_csv_rows(path, fieldnames, rows)

    with pytest.raises(ValueError, match="draw grain contains a duplicate row"):
        validate_few_label_bundle(out)


def test_bundle_validation_rejects_incomplete_draw_coverage(
    tmp_path: Path, completed_quick_bundle: Path
) -> None:
    out = _copy_bundle(completed_quick_bundle, tmp_path / "bundle")
    path = out / "few_label_draws.csv"
    fieldnames, rows = _read_csv_rows(path)
    rows.pop()
    _write_csv_rows(path, fieldnames, rows)

    with pytest.raises(ValueError, match="draw audit row coverage is incomplete"):
        validate_few_label_bundle(out)


def test_bundle_validation_rejects_changed_source_base_composition(
    tmp_path: Path, completed_quick_bundle: Path
) -> None:
    out = _copy_bundle(completed_quick_bundle, tmp_path / "bundle")
    path = out / "few_label_source_bases.csv"
    fieldnames, rows = _read_csv_rows(path)
    rows[0]["cohort"] = "STAD"
    _write_csv_rows(path, fieldnames, rows)

    with pytest.raises(ValueError, match="source-base composition is invalid"):
        validate_few_label_bundle(out)


def test_bundle_validation_rejects_changed_fold_membership(
    tmp_path: Path, completed_quick_bundle: Path
) -> None:
    out = _copy_bundle(completed_quick_bundle, tmp_path / "bundle")
    path = out / "few_label_folds.csv"
    fieldnames, rows = _read_csv_rows(path)
    rows[0]["held_out_sites"] = "('not-a-site',)"
    _write_csv_rows(path, fieldnames, rows)

    with pytest.raises(ValueError, match="fold site partition is invalid"):
        validate_few_label_bundle(out)


def test_bundle_validation_rejects_changed_loaded_case_membership(
    tmp_path: Path, completed_quick_bundle: Path
) -> None:
    out = _copy_bundle(completed_quick_bundle, tmp_path / "bundle")
    path = out / "few_label_cohort_cases.csv"
    fieldnames, rows = _read_csv_rows(path)
    rows[0]["site"] = "not-a-site"
    _write_csv_rows(path, fieldnames, rows)

    with pytest.raises(ValueError, match="cohort-case coverage is incomplete"):
        validate_few_label_bundle(out)


def test_bundle_validation_rejects_incomplete_prediction_coverage(
    tmp_path: Path, completed_quick_bundle: Path
) -> None:
    out = _copy_bundle(completed_quick_bundle, tmp_path / "bundle")
    path = out / "few_label_predictions.csv"
    fieldnames, rows = _read_csv_rows(path)
    rows.pop()
    _write_csv_rows(path, fieldnames, rows)

    with pytest.raises(ValueError, match="prediction row coverage is incomplete"):
        validate_few_label_bundle(out)


def test_bundle_validation_can_require_reportable_status(
    completed_quick_bundle: Path,
) -> None:
    with pytest.raises(ValueError, match="bundle is not reportable"):
        validate_few_label_bundle(completed_quick_bundle, require_reportable=True)


def test_bundle_validation_can_require_completion(
    tmp_path: Path, completed_quick_bundle: Path
) -> None:
    out = _copy_bundle(completed_quick_bundle, tmp_path / "bundle")
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["status"] = "running"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="bundle is not complete"):
        validate_few_label_bundle(out, require_complete=True)
