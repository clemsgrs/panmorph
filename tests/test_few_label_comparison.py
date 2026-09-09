from pathlib import Path

import pandas as pd
import pytest

from panmorph.few_label_comparison import build_lift_comparison, render_figure, render_markdown


def _summaries(offset: float) -> pd.DataFrame:
    rows = []
    for source, target, base in (("STAD", "COAD", "single"), ("COAD", "STAD", "single"),
                                 ("COAD+UCEC", "STAD", "pooled")):
        for k in (0, 3, 10, "all"):
            lift = 0.10 + offset if k != 0 else 0.25
            rows.append(dict(source=source, target=target, base=base, k=k, lift=lift,
                             lift_ci_lower=lift - 0.05, lift_ci_upper=lift + 0.05))
    return pd.DataFrame(rows)


def test_panels_follow_the_figure_layout_and_drop_zero_shot() -> None:
    comparison = build_lift_comparison({"prism": _summaries(0.0), "uni": _summaries(0.02)})

    assert comparison.feature_sets == ("prism", "uni")
    assert [(p.source, p.target, p.base) for p in comparison.panels] == [
        ("STAD", "COAD", "single"), ("COAD", "STAD", "single"), ("COAD+UCEC", "STAD", "pooled"),
    ]
    assert [p.k for p in comparison.panel("STAD", "COAD").series["uni"]] == [3, 10, "all"]
    assert comparison.panel("STAD", "COAD").series["uni"][0].lift == pytest.approx(0.12)


def test_a_feature_set_missing_a_panel_is_refused() -> None:
    partial = _summaries(0.0)
    partial = partial[partial.source != "COAD+UCEC"]

    with pytest.raises(ValueError, match="missing from feature set"):
        build_lift_comparison({"prism": _summaries(0.0), "uni": partial})


def test_markdown_has_one_table_per_gi_direction_with_a_column_per_feature_set() -> None:
    comparison = build_lift_comparison({"prism": _summaries(0.0), "uni": _summaries(-0.2)})

    text = render_markdown(comparison)

    assert text.startswith("| STAD→COAD: local MSI-positive cases | prism | uni |\n|---:|:---:|:---:|\n")
    assert "| 3 | +0.100 [+0.050, +0.150] | −0.100 [−0.150, −0.050] |" in text
    assert "| All available |" in text
    assert "| COAD→STAD: local MSI-positive cases | prism | uni |" in text


def test_figure_is_written_as_png_and_pdf(tmp_path: Path) -> None:
    comparison = build_lift_comparison({"prism": _summaries(0.0), "uni": _summaries(0.02)})

    png, pdf = render_figure(comparison, tmp_path / "lift")

    assert png.exists() and png.stat().st_size > 0
    assert pdf.exists() and pdf.stat().st_size > 0
