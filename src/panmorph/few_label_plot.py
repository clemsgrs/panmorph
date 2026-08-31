"""Semantic specification for the few-label result figures."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

from .few_label import Base, Rung


@dataclass(frozen=True)
class FewLabelPlotCell:
    """One reported few-label rung, independent of any plotting library."""

    source: str
    target: str
    base: Base
    k: Rung
    warm: tuple[float, float, float]
    cold: tuple[float, float, float]
    lift: tuple[float, float, float]
    rank_diverged: bool
    confirmatory: bool
    permutation_p: float | None


@dataclass(frozen=True)
class FewLabelEquivalenceMark:
    """Foreign-only value expressed as local target-positive cases."""

    source: str
    target: str
    base: Base
    point: float | None
    lower: float | None
    upper: float | None
    point_censored: bool
    lower_censored: bool
    upper_censored: bool


@dataclass(frozen=True)
class FewLabelPlotPanel:
    """One source-to-target facet in the publication figure."""

    row: int
    column: int
    source: str
    target: str
    base: Base
    gi_direction: bool
    cells: tuple[FewLabelPlotCell, ...]
    equivalence: FewLabelEquivalenceMark | None

    @property
    def local_comparison_cells(self) -> tuple[FewLabelPlotCell, ...]:
        """Cells where both arms use the same local training cases."""
        return tuple(cell for cell in self.cells if cell.k != 0)

    @property
    def local_ceiling(self) -> tuple[Literal["all"], float] | None:
        cell = next((cell for cell in self.cells if cell.k == "all"), None)
        return None if cell is None else ("all", cell.cold[0])


@dataclass(frozen=True)
class FewLabelPlotSpec:
    """Complete stable semantics consumed by the Matplotlib renderer."""

    reportable: bool
    panels: tuple[FewLabelPlotPanel, ...]


_FULL_MATRIX_LAYOUT: tuple[tuple[str, str, Base], ...] = (
    ("STAD", "COAD", "single"),
    ("COAD", "STAD", "single"),
    ("COAD", "UCEC", "single"),
    ("UCEC", "COAD", "single"),
    ("UCEC", "STAD", "single"),
    ("STAD", "UCEC", "single"),
    ("STAD+UCEC", "COAD", "pooled"),
    ("COAD+UCEC", "STAD", "pooled"),
    ("COAD+STAD", "UCEC", "pooled"),
)


def _rung_order(k: Rung) -> int:
    return (0, 3, 5, 10, 25, 40, "all").index(k)


def build_few_label_plot_spec(
    cells: Iterable[FewLabelPlotCell],
    equivalences: Iterable[FewLabelEquivalenceMark],
    *,
    reportable: bool,
) -> FewLabelPlotSpec:
    """Arrange few-label estimates into stable, unrelated-series-safe facets."""
    grouped: dict[tuple[str, str, Base], list[FewLabelPlotCell]] = {}
    for cell in cells:
        grouped.setdefault((cell.source, cell.target, cell.base), []).append(cell)
    marks = {
        (mark.source, mark.target, mark.base): mark for mark in equivalences
    }
    registered = [key for key in _FULL_MATRIX_LAYOUT if key in grouped]
    extras = sorted(set(grouped) - set(registered))
    ordered = registered + extras
    panels = tuple(
        FewLabelPlotPanel(
            row=index // 3,
            column=index % 3,
            source=source,
            target=target,
            base=base,
            gi_direction=(source, target) in {("COAD", "STAD"), ("STAD", "COAD")},
            cells=tuple(sorted(grouped[(source, target, base)], key=lambda cell: _rung_order(cell.k))),
            equivalence=marks.get((source, target, base)),
        )
        for index, (source, target, base) in enumerate(ordered)
    )
    return FewLabelPlotSpec(reportable=reportable, panels=panels)
