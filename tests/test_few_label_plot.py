from panmorph.few_label_plot import FewLabelEquivalenceMark, FewLabelPlotCell, build_few_label_plot_spec


def _cell(source: str, target: str, base: str, k=0) -> FewLabelPlotCell:
    return FewLabelPlotCell(
        source=source,
        target=target,
        base=base,
        k=k,
        warm=(0.75, 0.70, 0.80),
        cold=(0.50, 0.50, 0.50),
        lift=(0.25, 0.20, 0.30),
        fold_diverged=False,
        confirmatory=False,
        passed=None,
    )


def test_full_matrix_facets_keep_both_gi_directions_prominent() -> None:
    cells = tuple(
        _cell(source, target, base)
        for source, target, base in reversed((
            ("STAD", "COAD", "single"),
            ("COAD", "STAD", "single"),
            ("COAD", "UCEC", "single"),
            ("UCEC", "COAD", "single"),
            ("UCEC", "STAD", "single"),
            ("STAD", "UCEC", "single"),
            ("STAD+UCEC", "COAD", "pooled"),
            ("COAD+UCEC", "STAD", "pooled"),
            ("COAD+STAD", "UCEC", "pooled"),
        ))
    )

    spec = build_few_label_plot_spec(cells, (), reportable=True)

    assert tuple(
        (panel.row, panel.column, panel.source, panel.target, panel.base, panel.gi_direction)
        for panel in spec.panels
    ) == (
        (0, 0, "STAD", "COAD", "single", True),
        (0, 1, "COAD", "STAD", "single", True),
        (0, 2, "COAD", "UCEC", "single", False),
        (1, 0, "UCEC", "COAD", "single", False),
        (1, 1, "UCEC", "STAD", "single", False),
        (1, 2, "STAD", "UCEC", "single", False),
        (2, 0, "STAD+UCEC", "COAD", "pooled", False),
        (2, 1, "COAD+UCEC", "STAD", "pooled", False),
        (2, 2, "COAD+STAD", "UCEC", "pooled", False),
    )


def test_local_comparisons_exclude_the_zero_shot_point() -> None:
    cells = tuple(
        _cell("STAD", "COAD", "single", k)
        for k in (0, 3, 5, 10, 25, 40, "all")
    )

    panel = build_few_label_plot_spec(cells, (), reportable=True).panels[0]

    assert tuple(cell.k for cell in panel.local_comparison_cells) == (
        3, 5, 10, 25, 40, "all",
    )

def test_panel_exposes_local_ceiling_and_equivalence_mark() -> None:
    cells = (
        FewLabelPlotCell(
            "COAD", "STAD", "single", 0,
            (0.76, 0.70, 0.82), (0.50, 0.50, 0.50),
            (0.26, 0.20, 0.32), False, False, None,
        ),
        FewLabelPlotCell(
            "COAD", "STAD", "single", 10,
            (0.802, 0.750, 0.851), (0.788, 0.753, 0.821),
            (0.014, -0.031, 0.061), False, True, False,
        ),
        FewLabelPlotCell(
            "COAD", "STAD", "single", "all",
            (0.834, 0.773, 0.886), (0.858, 0.811, 0.902),
            (-0.024, -0.068, 0.020), True, False, None,
        ),
    )
    equivalence = FewLabelEquivalenceMark(
        "COAD", "STAD", "single", 8.0, 4.0, 20.0,
        False, False, False,
    )

    panel = build_few_label_plot_spec(cells, (equivalence,), reportable=True).panels[0]

    assert panel.local_ceiling == ("all", 0.858)
    assert panel.equivalence == equivalence
