"""Compare the few-label AUC gain of several feature sets on one figure and one table.

Each feature set contributes the ``few_label_summaries.csv`` of its complete bundle.
The comparison is descriptive: the series are drawn side by side and no test between
feature sets is computed.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from panmorph.few_label_plot import _FULL_MATRIX_LAYOUT, _rung_order  # noqa: E402

Rung = int | str

GI_DIRECTIONS: tuple[tuple[str, str], ...] = (("STAD", "COAD"), ("COAD", "STAD"))

_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
_RUNG_POSITIONS: dict[Rung, int] = {3: 0, 5: 1, 10: 2, 25: 3, 40: 4, "all": 5}


@dataclass(frozen=True)
class LiftPoint:
    k: Rung
    lift: float
    lower: float
    upper: float


@dataclass(frozen=True)
class LiftPanel:
    source: str
    target: str
    base: str
    series: Mapping[str, tuple[LiftPoint, ...]]

    @property
    def gi_direction(self) -> bool:
        return (self.source, self.target) in GI_DIRECTIONS


@dataclass(frozen=True)
class LiftComparison:
    feature_sets: tuple[str, ...]
    panels: tuple[LiftPanel, ...]

    def panel(self, source: str, target: str, base: str = "single") -> LiftPanel:
        for panel in self.panels:
            if (panel.source, panel.target, panel.base) == (source, target, base):
                return panel
        raise KeyError((source, target, base))


def _rung(value: object) -> Rung:
    text = str(value)
    return text if text == "all" else int(float(text))


def build_lift_comparison(summaries: Mapping[str, pd.DataFrame]) -> LiftComparison:
    """Arrange the per-feature-set gains into the facets of the few-label figure.

    Zero-shot rows (k = 0) are dropped: no local-only model exists there, so the
    gain is not a paired comparison. Every feature set must cover the same panels.
    """
    if not summaries:
        raise ValueError("at least one feature set is required")
    names = tuple(summaries)
    grouped: dict[tuple[str, str, str], dict[str, list[LiftPoint]]] = {}
    for name, table in summaries.items():
        for row in table.itertuples(index=False):
            k = _rung(row.k)
            if k == 0:
                continue
            key = (str(row.source), str(row.target), str(row.base))
            grouped.setdefault(key, {}).setdefault(name, []).append(
                LiftPoint(k, float(row.lift), float(row.lift_ci_lower), float(row.lift_ci_upper))
            )
    for key, series in grouped.items():
        missing = set(names) - set(series)
        if missing:
            raise ValueError(f"{key} is missing from feature set(s) {sorted(missing)}")
    ordered = [key for key in _FULL_MATRIX_LAYOUT if key in grouped]
    ordered += sorted(set(grouped) - set(ordered))
    panels = tuple(
        LiftPanel(
            source, target, base,
            {
                name: tuple(sorted(grouped[(source, target, base)][name], key=lambda p: _rung_order(p.k)))
                for name in names
            },
        )
        for source, target, base in ordered
    )
    return LiftComparison(names, panels)


def _fmt(point: LiftPoint) -> str:
    return f"{point.lift:+.3f} [{point.lower:+.3f}, {point.upper:+.3f}]".replace("-", "−")


def render_markdown(
    comparison: LiftComparison,
    directions: tuple[tuple[str, str], ...] = GI_DIRECTIONS,
) -> str:
    """One Markdown table per direction: rows are local-positive counts, columns feature sets."""
    blocks = []
    for source, target in directions:
        panel = comparison.panel(source, target)
        header = f"| {source}→{target}: local MSI-positive cases | " + " | ".join(comparison.feature_sets) + " |"
        align = "|---:|" + "|".join(":---:" for _ in comparison.feature_sets) + "|"
        lines = [header, align]
        rungs = [p.k for p in panel.series[comparison.feature_sets[0]]]
        for k in rungs:
            cells = []
            for name in comparison.feature_sets:
                point = next(p for p in panel.series[name] if p.k == k)
                cells.append(_fmt(point))
            label = "All available" if k == "all" else str(k)
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + "\n"


def render_figure(comparison: LiftComparison, out_stem: Path) -> tuple[Path, Path]:
    """Write ``<out_stem>.png`` and ``<out_stem>.pdf`` with one gain series per feature set."""
    n_panels = len(comparison.panels)
    n_cols = 3
    n_rows = max(1, -(-n_panels // n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(15.6, 3.7 * n_rows + 1.2), sharex=True, sharey=True, squeeze=False
    )
    flat = axes.ravel()
    n_sets = len(comparison.feature_sets)
    offsets = np.linspace(-0.18, 0.18, n_sets) if n_sets > 1 else np.zeros(1)
    for ax, panel in zip(flat, comparison.panels):
        ax.axhline(0, color="black", linewidth=0.9)
        for index, name in enumerate(comparison.feature_sets):
            points = panel.series[name]
            x = np.asarray([_RUNG_POSITIONS[p.k] for p in points], dtype=float) + offsets[index]
            est = np.asarray([p.lift for p in points])
            lo = np.asarray([p.lower for p in points])
            hi = np.asarray([p.upper for p in points])
            ax.errorbar(
                x, est, yerr=np.vstack((est - lo, hi - est)), fmt="o-",
                color=_COLORS[index % len(_COLORS)], capsize=2.5, linewidth=1.4,
                markersize=3.8, label=name,
            )
        title = f"{panel.source} → {panel.target}"
        if panel.base == "pooled":
            title += "  ·  pooled source"
        ax.set_title(title, fontsize=10, fontweight="bold" if panel.gi_direction else None)
        if panel.gi_direction:
            for spine in ax.spines.values():
                spine.set_color("#0072B2")
                spine.set_linewidth(1.5)
        ax.set_xticks(list(_RUNG_POSITIONS.values()), [str(k) for k in _RUNG_POSITIONS])
        ax.grid(axis="y", color="0.9", linewidth=0.6)
    for ax in flat[n_panels:]:
        ax.set_visible(False)
    for ax in axes[-1, :]:
        if ax.get_visible():
            ax.set_xlabel("Local MSI-positive cases")
    for ax in axes[:, 0]:
        if ax.get_visible():
            ax.set_ylabel("AUC gain from other-organ data")
    fig.suptitle("AUC gain from adding other-organ data, by feature set", fontsize=15)
    handles, labels = flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.945), ncol=n_sets)
    fig.text(
        0.5, 0.005,
        "Values above zero favor other-organ + local training. Bars are 95% intervals. "
        "Series are offset for legibility; no test between feature sets is computed.",
        ha="center", fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.025, 1, 0.91))
    png, pdf = out_stem.with_suffix(".png"), out_stem.with_suffix(".pdf")
    fig.savefig(png, dpi=180)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf
