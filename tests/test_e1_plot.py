from panmorph.e1_plot import E1EquivalenceMark, E1PlotCell, build_e1_plot_spec


def _cell(source: str, target: str, base: str) -> E1PlotCell:
    return E1PlotCell(
        source=source,
        target=target,
        base=base,
        k=0,
        warm=(0.75, 0.70, 0.80),
        cold=(0.50, 0.50, 0.50),
        lift=(0.25, 0.20, 0.30),
        rank_diverged=False,
        confirmatory=False,
        permutation_p=None,
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

    spec = build_e1_plot_spec(cells, (), reportable=True)

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


def test_panel_exposes_zero_shot_confirmatory_ceiling_and_equivalence_marks() -> None:
    cells = (
        E1PlotCell(
            "COAD", "STAD", "single", 0,
            (0.76, 0.70, 0.82), (0.50, 0.50, 0.50),
            (0.26, 0.20, 0.32), False, False, None,
        ),
        E1PlotCell(
            "COAD", "STAD", "single", 10,
            (0.802, 0.750, 0.851), (0.788, 0.753, 0.821),
            (0.014, -0.031, 0.061), False, True, 0.001,
        ),
        E1PlotCell(
            "COAD", "STAD", "single", "all",
            (0.834, 0.773, 0.886), (0.858, 0.811, 0.902),
            (-0.024, -0.068, 0.020), True, False, None,
        ),
    )
    equivalence = E1EquivalenceMark(
        "COAD", "STAD", "single", 8.0, 4.0, 20.0,
        False, False, False,
    )

    panel = build_e1_plot_spec(cells, (equivalence,), reportable=True).panels[0]

    assert panel.phase1_anchor == (0, 0.76)
    assert panel.confirmatory_mark == (10, 0.014, -0.031, 0.061, 0.001)
    assert panel.local_ceiling == ("all", 0.858)
    assert panel.equivalence == equivalence
    assert panel.rank_sensitive
