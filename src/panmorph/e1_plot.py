"""Semantic specification for the E1 result figures."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

from .e1 import Base, Rung


@dataclass(frozen=True)
class E1PlotCell:
    """One reported E1 rung, independent of any plotting library."""

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
class E1EquivalenceMark:
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
class E1PlotPanel:
    """One source-to-target facet in the publication figure."""

    row: int
    column: int
    source: str
    target: str
    base: Base
    gi_direction: bool
    cells: tuple[E1PlotCell, ...]
    equivalence: E1EquivalenceMark | None

    @property
    def phase1_anchor(self) -> tuple[int, float] | None:
        cell = next((cell for cell in self.cells if cell.k == 0), None)
        return None if cell is None else (0, cell.warm[0])

    @property
    def confirmatory_mark(
        self,
    ) -> tuple[int, float, float, float, float] | None:
        cell = next((cell for cell in self.cells if cell.confirmatory), None)
        if cell is None or not isinstance(cell.k, int) or cell.permutation_p is None:
            return None
        return (cell.k, *cell.lift, cell.permutation_p)

    @property
    def local_ceiling(self) -> tuple[Literal["all"], float] | None:
        cell = next((cell for cell in self.cells if cell.k == "all"), None)
        return None if cell is None else ("all", cell.cold[0])

    @property
    def rank_sensitive(self) -> bool:
        return any(cell.rank_diverged for cell in self.cells)


@dataclass(frozen=True)
class E1PlotSpec:
    """Complete stable semantics consumed by the Matplotlib renderer."""

    reportable: bool
    panels: tuple[E1PlotPanel, ...]


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


def build_e1_plot_spec(
    cells: Iterable[E1PlotCell],
    equivalences: Iterable[E1EquivalenceMark],
    *,
    reportable: bool,
) -> E1PlotSpec:
    """Arrange E1 estimates into stable, unrelated-series-safe facets."""
    grouped: dict[tuple[str, str, Base], list[E1PlotCell]] = {}
    for cell in cells:
        grouped.setdefault((cell.source, cell.target, cell.base), []).append(cell)
    marks = {
        (mark.source, mark.target, mark.base): mark for mark in equivalences
    }
    registered = [key for key in _FULL_MATRIX_LAYOUT if key in grouped]
    extras = sorted(set(grouped) - set(registered))
    ordered = registered + extras
    panels = tuple(
        E1PlotPanel(
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
    return E1PlotSpec(reportable=reportable, panels=panels)
