"""Verdict matrix: the gate tables of several feature sets side by side.

The matrix keeps each feature set's own gate values (AUC, bootstrap interval,
permutation p, pass verdict) and flags where a cell's verdict differs from the
reference feature set. It computes no statistic between feature sets.
"""
from __future__ import annotations

from typing import Mapping

import pandas as pd

REFERENCE = "prism"

MATRIX_COLUMNS = [
    "component", "source", "target", "feature_set", "auc", "ci_lo", "ci_hi",
    "perm_p", "passed", "newly_passing", "no_longer_passing",
]


def _verdict(row: pd.Series) -> bool | None:
    """The pass verdict of one gate row: True, False, or None when the gate gave none.

    Within-organ ceilings never carry a verdict; the gate applies its rule only to
    single-source zero-shot cells.
    """
    value = row["passed"]
    if row["component"] == "ceiling" or pd.isna(value):
        return None
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def build_verdict_matrix(
    tables: Mapping[str, pd.DataFrame], reference: str = REFERENCE
) -> pd.DataFrame:
    """One long table with a row per gate cell and feature set.

    ``tables`` maps a feature-set name to its gate table (``gate_results.csv``).
    Cells follow the order of the reference table; feature sets follow the order of
    ``tables``. ``newly_passing`` marks a single-source zero-shot cell that passes
    while the reference cell does not; ``no_longer_passing`` marks the opposite.
    """
    if reference not in tables:
        raise ValueError(f"reference feature set {reference!r} is not among {list(tables)}")
    keyed = {
        name: table.set_index(["component", "source", "target"], drop=False)
        for name, table in tables.items()
    }
    cells = list(keyed[reference].index)
    for name, table in keyed.items():
        cells += [cell for cell in table.index if cell not in cells]
    rows = []
    for cell in cells:
        ref = keyed[reference].loc[cell] if cell in keyed[reference].index else None
        ref_passed = None if ref is None else _verdict(ref)
        for name, table in keyed.items():
            if cell not in table.index:
                continue
            row = table.loc[cell]
            passed = _verdict(row)
            flagged = passed is not None and ref_passed is not None
            rows.append(
                dict(
                    component=cell[0], source=cell[1], target=cell[2], feature_set=name,
                    auc=row["auc"], ci_lo=row["ci_lo"], ci_hi=row["ci_hi"],
                    perm_p=row["perm_p"], passed=passed,
                    newly_passing=bool(flagged and passed and not ref_passed),
                    no_longer_passing=bool(flagged and ref_passed and not passed),
                )
            )
    return pd.DataFrame(rows, columns=MATRIX_COLUMNS)


def _cell_text(row) -> str:
    """One Markdown cell: AUC, interval, permutation p, and the verdict with its flag."""
    if pd.isna(row.auc):
        return "-"
    text = f"{row.auc:.2f} [{row.ci_lo:.2f}, {row.ci_hi:.2f}]"
    if row.component == "ceiling":
        return text
    text += f", p={row.perm_p:.3f}"
    if row.passed is None:
        return text
    text += ", pass" if row.passed else ", fail"
    if row.newly_passing:
        text += " (new)"
    if row.no_longer_passing:
        text += " (was pass)"
    return text


def _table(header: str, feature_sets: list[str], rows: list[tuple[str, dict]]) -> list[str]:
    lines = [
        "| " + " | ".join([header, *feature_sets]) + " |",
        "|---|" + ":---:|" * len(feature_sets),
    ]
    for label, cells in rows:
        lines.append("| " + " | ".join([label, *(cells.get(f, "-") for f in feature_sets)]) + " |")
    return lines


def render_markdown(matrix: pd.DataFrame, reference: str = REFERENCE) -> str:
    """The verdict matrix as Markdown: feature sets as columns, gate cells as rows.

    Zero-shot cells come first, then the within-organ ceilings in their own table.
    """
    feature_sets = list(dict.fromkeys(matrix.feature_set))
    grouped: dict[str, list[tuple[str, dict]]] = {"zeroshot": [], "ceiling": []}
    for (component, source, target), group in matrix.groupby(
        ["component", "source", "target"], sort=False
    ):
        label = target if component == "ceiling" else f"{source} -> {target}"
        cells = {row.feature_set: _cell_text(row) for row in group.itertuples()}
        grouped[component].append((label, cells))
    lines = _table("Source -> target", feature_sets, grouped["zeroshot"])
    lines += [
        "",
        "Each cell shows the target-organ AUC, its 95% bootstrap interval, and the permutation p.",
        "`pass` is the gate verdict of that feature set. `(new)` marks a cell that passes with",
        f"this feature set but not with {reference}; `(was pass)` marks the opposite. Combined-source",
        "cells have no verdict. No test between feature sets is computed.",
        "",
        "Within-organ ceiling (site-held-out AUC):",
        "",
    ]
    lines += _table("Organ", feature_sets, grouped["ceiling"])
    return "\n".join(lines)
