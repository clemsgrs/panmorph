import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from panmorph.verdict_matrix import build_verdict_matrix, render_markdown


def _gate_table(cells: dict[tuple[str, str], tuple[float, float, float, float]]) -> pd.DataFrame:
    """A synthetic gate table with one COAD ceiling and the given zero-shot cells.

    ``cells`` maps (source, target) to (auc, ci_lo, ci_hi, perm_p); combined-source
    cells carry no verdict, single-source cells get the gate pass rule.
    """
    rows = [
        dict(component="ceiling", source="COAD (within)", target="COAD", auc=0.77,
             ci_lo=0.70, ci_hi=0.83, perm_p=np.nan, auc_random_cv=0.81, n_target=391,
             pos_target=74, role="ceiling", passed=np.nan),
    ]
    for (src, tgt), (auc, lo, hi, p) in cells.items():
        combined = src.endswith("(combined)")
        rows.append(
            dict(component="zeroshot", source=src, target=tgt, auc=auc, ci_lo=lo,
                 ci_hi=hi, perm_p=p, auc_random_cv=np.nan, n_target=391, pos_target=74,
                 role="exploratory",
                 passed=np.nan if combined else (lo > 0.60 and p < 0.05))
        )
    return pd.DataFrame(rows)


def test_three_feature_sets_give_one_matrix_with_flags_relative_to_prism() -> None:
    tables = {
        "prism": _gate_table({
            ("STAD", "COAD"): (0.74, 0.68, 0.80, 0.003),
            ("UCEC", "COAD"): (0.57, 0.50, 0.64, 0.221),
            ("UCEC+STAD (combined)", "COAD"): (0.66, 0.59, 0.73, 0.016),
        }),
        "uni": _gate_table({
            ("STAD", "COAD"): (0.55, 0.48, 0.62, 0.300),
            ("UCEC", "COAD"): (0.71, 0.65, 0.77, 0.001),
            ("UCEC+STAD (combined)", "COAD"): (0.70, 0.64, 0.76, 0.001),
        }),
        "prism2": _gate_table({
            ("STAD", "COAD"): (0.78, 0.72, 0.84, 0.001),
            ("UCEC", "COAD"): (0.60, 0.53, 0.67, 0.100),
            ("UCEC+STAD (combined)", "COAD"): (0.69, 0.62, 0.76, 0.010),
        }),
    }

    matrix = build_verdict_matrix(tables, reference="prism")

    assert list(matrix.columns) == [
        "component", "source", "target", "feature_set", "auc", "ci_lo", "ci_hi",
        "perm_p", "passed", "newly_passing", "no_longer_passing",
    ]
    zeroshot = matrix[matrix.component == "zeroshot"]
    assert [
        (r.source, r.target, r.feature_set, r.auc, r.ci_lo, r.ci_hi, r.perm_p,
         r.passed, r.newly_passing, r.no_longer_passing)
        for r in zeroshot.itertuples()
    ] == [
        ("STAD", "COAD", "prism", 0.74, 0.68, 0.80, 0.003, True, False, False),
        ("STAD", "COAD", "uni", 0.55, 0.48, 0.62, 0.300, False, False, True),
        ("STAD", "COAD", "prism2", 0.78, 0.72, 0.84, 0.001, True, False, False),
        ("UCEC", "COAD", "prism", 0.57, 0.50, 0.64, 0.221, False, False, False),
        ("UCEC", "COAD", "uni", 0.71, 0.65, 0.77, 0.001, True, True, False),
        ("UCEC", "COAD", "prism2", 0.60, 0.53, 0.67, 0.100, False, False, False),
        ("UCEC+STAD (combined)", "COAD", "prism", 0.66, 0.59, 0.73, 0.016, None, False, False),
        ("UCEC+STAD (combined)", "COAD", "uni", 0.70, 0.64, 0.76, 0.001, None, False, False),
        ("UCEC+STAD (combined)", "COAD", "prism2", 0.69, 0.62, 0.76, 0.010, None, False, False),
    ]


def test_within_organ_ceilings_are_never_flagged() -> None:
    prism = _gate_table({("STAD", "COAD"): (0.74, 0.68, 0.80, 0.003)})
    uni = _gate_table({("STAD", "COAD"): (0.55, 0.48, 0.62, 0.300)})
    prism.loc[prism.component == "ceiling", "passed"] = True
    uni.loc[uni.component == "ceiling", "passed"] = False

    matrix = build_verdict_matrix({"prism": prism, "uni": uni})
    ceiling = matrix[matrix.component == "ceiling"]

    assert list(ceiling.source) == ["COAD (within)", "COAD (within)"]
    assert list(ceiling.passed) == [None, None]
    assert not ceiling.newly_passing.any()
    assert not ceiling.no_longer_passing.any()


def test_markdown_has_feature_sets_as_columns_and_cross_organ_cells_as_rows() -> None:
    tables = {
        "prism": _gate_table({
            ("STAD", "COAD"): (0.74, 0.68, 0.80, 0.003),
            ("UCEC", "COAD"): (0.57, 0.50, 0.64, 0.221),
            ("UCEC+STAD (combined)", "COAD"): (0.66, 0.59, 0.73, 0.016),
        }),
        "uni": _gate_table({
            ("STAD", "COAD"): (0.55, 0.48, 0.62, 0.300),
            ("UCEC", "COAD"): (0.71, 0.65, 0.77, 0.001),
            ("UCEC+STAD (combined)", "COAD"): (0.70, 0.64, 0.76, 0.001),
        }),
    }

    markdown = render_markdown(build_verdict_matrix(tables))

    assert markdown == "\n".join([
        "| Source -> target | prism | uni |",
        "|---|:---:|:---:|",
        "| STAD -> COAD | 0.74 [0.68, 0.80], p=0.003, pass | 0.55 [0.48, 0.62], p=0.300, fail (was pass) |",
        "| UCEC -> COAD | 0.57 [0.50, 0.64], p=0.221, fail | 0.71 [0.65, 0.77], p=0.001, pass (new) |",
        "| UCEC+STAD (combined) -> COAD | 0.66 [0.59, 0.73], p=0.016 | 0.70 [0.64, 0.76], p=0.001 |",
        "",
        "Each cell shows the target-organ AUC, its 95% bootstrap interval, and the permutation p.",
        "`pass` is the gate verdict of that feature set. `(new)` marks a cell that passes with",
        "this feature set but not with prism; `(was pass)` marks the opposite. Combined-source",
        "cells have no verdict. No test between feature sets is computed.",
        "",
        "Within-organ ceiling (site-held-out AUC):",
        "",
        "| Organ | prism | uni |",
        "|---|:---:|:---:|",
        "| COAD | 0.77 [0.70, 0.83] | 0.77 [0.70, 0.83] |",
    ])


def test_script_reads_named_gate_tables_and_prints_the_markdown(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    tables = {
        "prism": _gate_table({("STAD", "COAD"): (0.74, 0.68, 0.80, 0.003)}),
        "uni": _gate_table({("STAD", "COAD"): (0.55, 0.48, 0.62, 0.300)}),
    }
    for name, table in tables.items():
        (tmp_path / name).mkdir()
        table.to_csv(tmp_path / name / "gate_results.csv", index=False)
    script = _load_script()
    monkeypatch.setattr(
        sys, "argv",
        ["render_verdict_matrix.py",
         f"prism={tmp_path / 'prism' / 'gate_results.csv'}",
         f"uni={tmp_path / 'uni' / 'gate_results.csv'}"],
    )

    script.main()

    out = capsys.readouterr().out
    assert out == render_markdown(build_verdict_matrix(tables)) + "\n"
    assert "| STAD -> COAD | 0.74 [0.68, 0.80], p=0.003, pass | 0.55 [0.48, 0.62], p=0.300, fail (was pass) |" in out


def _load_script():
    path = Path(__file__).resolve().parent.parent / "experiments" / "render_verdict_matrix.py"
    spec = importlib.util.spec_from_file_location("render_verdict_matrix_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
