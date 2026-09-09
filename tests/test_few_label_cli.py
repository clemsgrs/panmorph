import sys
from pathlib import Path

import pytest

import experiments.run_few_label as script
from experiments.run_few_label import build_parser, parse_args


def test_few_label_command_uses_clear_default_result_directory() -> None:
    args = parse_args([], root=Path("/repo"))

    assert args.out == Path("/repo/results/few-label")


def test_few_label_command_default_result_directory_follows_the_feature_set() -> None:
    args = parse_args(["--features", "prism2-base"], root=Path("/repo"))

    assert args.out == Path("/repo/results/prism2-base/few-label")


def test_few_label_command_keeps_an_explicit_result_directory() -> None:
    args = parse_args(["--features", "prism2-base", "--out", "x"], root=Path("/repo"))

    assert args.out == Path("x")


def test_few_label_command_defaults_to_the_prism_feature_set() -> None:
    args = build_parser(Path("/repo")).parse_args([])

    assert args.features == "prism"


def test_few_label_command_accepts_a_registered_feature_set() -> None:
    args = build_parser(Path("/repo")).parse_args(["--features", "prism2-diagnostic"])

    assert args.features == "prism2-diagnostic"


def test_few_label_command_rejects_an_unregistered_feature_set() -> None:
    with pytest.raises(SystemExit):
        build_parser(Path("/repo")).parse_args(["--features", "nope"])


def test_few_label_command_forwards_the_feature_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def fake_run(out, **kwargs):
        calls.append((out, kwargs))
        return out

    monkeypatch.setattr(script, "run_few_label_bundle", fake_run)
    monkeypatch.setattr(
        sys, "argv",
        ["run_few_label.py", "--profile", "quick", "--workers", "1",
         "--features", "prism2-base", "--out", str(tmp_path)],
    )

    script.main()

    assert calls == [(tmp_path, {"profile": "quick", "workers": 1, "features": "prism2-base"})]
