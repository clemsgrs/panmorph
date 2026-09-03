import importlib.util
import sys
from pathlib import Path

import numpy as np

from panmorph.data import Cohort
from panmorph.gate import cell_passes, run_gate


def _synthetic_cohorts() -> dict[str, Cohort]:
    """Three organs: COAD and STAD share one MSI-predictive feature, UCEC has none."""
    rng = np.random.default_rng(7)
    n = 120
    labels = np.tile(np.asarray([1, 0, 0, 0]), n // 4)
    sites = np.tile(np.asarray(["A", "B", "C", "D", "E", "F"]), n // 6)
    cohorts = {}
    for name in ("COAD", "UCEC", "STAD"):
        X = rng.normal(size=(n, 4))
        if name != "UCEC":
            X[:, 0] += 3 * (2 * labels - 1)
        cohorts[name] = Cohort(
            name=name,
            X=X.astype(np.float32),
            y=labels.copy(),
            sites=sites.copy(),
            case_ids=np.asarray([f"{name}-{index:03d}" for index in range(n)]),
        )
    return cohorts


def test_planted_signal_passes_only_between_the_two_planted_organs() -> None:
    result = run_gate(_synthetic_cohorts(), n_perm=40, n_boot=50, seed=0, n_jobs=1)

    assert result.verdicts == {
        ("COAD", "STAD"): True,
        ("STAD", "COAD"): True,
        ("UCEC", "COAD"): False,
        ("COAD", "UCEC"): False,
        ("STAD", "UCEC"): False,
        ("UCEC", "STAD"): False,
    }
    assert result.verdict.startswith("STRONG PASS")


def test_pass_rule_needs_lower_bound_above_060_and_p_below_005() -> None:
    assert cell_passes(ci_lo=0.61, perm_p=0.049)
    assert not cell_passes(ci_lo=0.60, perm_p=0.049)
    assert not cell_passes(ci_lo=0.61, perm_p=0.05)


def test_script_writes_the_module_table(tmp_path: Path, monkeypatch, capsys) -> None:
    script = _load_script()
    cohorts = _synthetic_cohorts()
    monkeypatch.setattr(script, "load_all", lambda **_kwargs: cohorts)
    monkeypatch.setattr(
        sys, "argv",
        ["run_gate.py", "--n-perm", "20", "--n-boot", "20", "--n-jobs", "1",
         "--out", str(tmp_path)],
    )

    script.main()

    written = (tmp_path / "gate_results.csv").read_text()
    expected = run_gate(cohorts, n_perm=20, n_boot=20, seed=0, n_jobs=1).table.to_csv(index=False)
    assert written == expected
    out = capsys.readouterr().out
    assert "WITHIN-ORGAN CEILING" in out
    assert "ZERO-SHOT TRANSFER" in out
    assert "VERDICT" in out
    assert ">>> STRONG PASS" in out


def test_results_directory_is_per_feature_set_except_for_the_default() -> None:
    from panmorph.gate import results_dir

    assert results_dir("prism", Path("/repo")) == Path("/repo/results")
    assert results_dir("prism2-diagnostic", Path("/repo")) == Path("/repo/results/prism2-diagnostic")


def test_script_writes_the_default_feature_set_to_the_results_root() -> None:
    args = _load_script().parse_args([], root=Path("/repo"))

    assert args.features == "prism"
    assert args.out == Path("/repo/results")


def test_script_writes_a_named_feature_set_to_its_own_results_directory() -> None:
    args = _load_script().parse_args(["--features", "prism2-base"], root=Path("/repo"))

    assert args.features == "prism2-base"
    assert args.out == Path("/repo/results/prism2-base")


def test_script_keeps_an_explicit_out_for_a_named_feature_set() -> None:
    args = _load_script().parse_args(
        ["--features", "prism2-diagnostic", "--out", "/elsewhere"], root=Path("/repo")
    )

    assert args.out == Path("/elsewhere")


def test_script_loads_the_requested_feature_set(tmp_path: Path, monkeypatch) -> None:
    script = _load_script()
    requested = []

    def fake_load_all(features="prism"):
        requested.append(features)
        return _synthetic_cohorts()

    monkeypatch.setattr(script, "load_all", fake_load_all)
    monkeypatch.setattr(
        sys, "argv",
        ["run_gate.py", "--n-perm", "20", "--n-boot", "20", "--n-jobs", "1",
         "--features", "prism2-base", "--out", str(tmp_path)],
    )

    script.main()

    assert requested == ["prism2-base"]
    assert (tmp_path / "gate_results.csv").is_file()


def _load_script():
    path = Path(__file__).resolve().parent.parent / "experiments" / "run_gate.py"
    spec = importlib.util.spec_from_file_location("run_gate_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
