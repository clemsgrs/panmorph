import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from experiments.extract_prism2 import (
    VARIANTS,
    AccessError,
    IntegrityError,
    build_dataset_table,
    build_parser,
    check_prism2_access,
    extract_prism2,
    require_soma_version,
    verify_features,
)

_PROVENANCE = {"soma_commit": "abc123", "soma_version": "1.13.0", "slide2vec_version": "5.8.2"}


def _write_label_table(path: Path, rows: list[tuple[str, int]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "case_id": [case for case, _ in rows],
            "msi": [0.5 for _ in rows],
            "msi_high": [label for _, label in rows],
            "wsi_path": [f"/slides/{case}.tif" for case, _ in rows],
            "mask_path": [f"/masks/{case}_tissue.tif" for case, _ in rows],
        }
    )
    frame.to_csv(path, index=False)
    return path


def test_dataset_table_has_soma_columns_and_one_row_per_labelled_case(tmp_path: Path) -> None:
    label_csv = _write_label_table(
        tmp_path / "labels.csv", [("TCGA-AA-0001", 1), ("TCGA-AA-0002", 0)]
    )

    table = build_dataset_table(label_csv, "msi_high")

    assert list(table.columns) == ["sample_id", "image_path", "label", "mask_path"]
    assert table["sample_id"].tolist() == ["TCGA-AA-0001", "TCGA-AA-0002"]
    assert table["image_path"].tolist() == ["/slides/TCGA-AA-0001.tif", "/slides/TCGA-AA-0002.tif"]
    assert table["mask_path"].tolist() == [
        "/masks/TCGA-AA-0001_tissue.tif", "/masks/TCGA-AA-0002_tissue.tif",
    ]
    assert table["label"].tolist() == [1, 0]


def test_dataset_table_drops_cases_without_a_label(tmp_path: Path) -> None:
    label_csv = _write_label_table(tmp_path / "labels.csv", [("TCGA-AA-0001", 1)])
    frame = pd.read_csv(label_csv)
    frame.loc[len(frame)] = ["TCGA-AA-0009", 0.1, None, "/slides/x.tif", "/masks/x.tif"]
    frame.to_csv(label_csv, index=False)

    table = build_dataset_table(label_csv, "msi_high")

    assert table["sample_id"].tolist() == ["TCGA-AA-0001"]


def _write_features(directory: Path, cases: dict[str, int]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for case, width in cases.items():
        torch.save(torch.zeros(width), directory / f"{case}.pt")


def test_integrity_check_passes_for_one_file_per_case_with_the_expected_width(
    tmp_path: Path,
) -> None:
    _write_features(tmp_path, {"TCGA-AA-0001": 2560, "TCGA-AA-0002": 2560})

    verify_features(tmp_path, ["TCGA-AA-0001", "TCGA-AA-0002"], width=2560)


def test_integrity_check_names_the_missing_case(tmp_path: Path) -> None:
    _write_features(tmp_path, {"TCGA-AA-0001": 2560})

    with pytest.raises(IntegrityError, match="missing.*TCGA-AA-0002"):
        verify_features(tmp_path, ["TCGA-AA-0001", "TCGA-AA-0002"], width=2560)


def test_integrity_check_names_the_duplicate_case_identifier(tmp_path: Path) -> None:
    _write_features(tmp_path, {"TCGA-AA-0001": 2560})

    with pytest.raises(IntegrityError, match="duplicate.*TCGA-AA-0001"):
        verify_features(tmp_path, ["TCGA-AA-0001", "TCGA-AA-0001"], width=2560)


def test_integrity_check_names_the_case_with_the_wrong_width(tmp_path: Path) -> None:
    _write_features(tmp_path, {"TCGA-AA-0001": 2560, "TCGA-AA-0002": 1280})

    with pytest.raises(IntegrityError, match="TCGA-AA-0002.*1280.*2560"):
        verify_features(tmp_path, ["TCGA-AA-0001", "TCGA-AA-0002"], width=2560)


class FakeExtractor:
    """Write one synthetic embedding per dataset row, as soma would, and log each call."""

    def __init__(self, widths: dict[str, int] | None = None) -> None:
        self.calls: list[tuple[Path, str, Path]] = []
        self.widths = widths or dict(VARIANTS)

    def __call__(self, dataset_csv: Path, variant: str, output_root: Path) -> Path:
        self.calls.append((dataset_csv, variant, output_root))
        feature_dir = output_root / "features"
        table = pd.read_csv(dataset_csv)
        _write_features(feature_dir, {case: self.widths[variant] for case in table["sample_id"]})
        return feature_dir


def _cohorts(tmp_path: Path) -> dict[str, tuple[Path, str, Path]]:
    cohorts = {}
    for name, size in (("COAD", 3), ("STAD", 2)):
        rows = [(f"TCGA-{name[:2]}-{index:04d}", index % 2) for index in range(size)]
        label_csv = _write_label_table(tmp_path / "labels" / f"{name}.csv", rows)
        cohorts[name] = (label_csv, "msi_high", tmp_path / "prism" / name)
    return cohorts


def test_fake_extraction_lands_both_variants_in_the_per_feature_set_per_cohort_layout(
    tmp_path: Path,
) -> None:
    features_root = tmp_path / "features"
    extractor = FakeExtractor()

    extract_prism2(
        _cohorts(tmp_path),
        features_root=features_root,
        work_root=tmp_path / "work",
        extractor=extractor,
        access_check=lambda: None,
        provenance=_PROVENANCE,
    )

    expected = set()
    for variant in ("base", "diagnostic"):
        for cohort, size in (("COAD", 3), ("STAD", 2)):
            for index in range(size):
                expected.add(f"prism2-{variant}/{cohort}/TCGA-{cohort[:2]}-{index:04d}.pt")
    found = {
        str(path.relative_to(features_root)) for path in features_root.rglob("*.pt")
    }
    assert found == expected
    assert sorted((variant, root.name) for _, variant, root in extractor.calls) == [
        ("base", "COAD"), ("base", "STAD"), ("diagnostic", "COAD"), ("diagnostic", "STAD"),
    ]


def test_manifest_records_provenance_tiling_and_variant(tmp_path: Path) -> None:
    features_root = tmp_path / "features"

    extract_prism2(
        _cohorts(tmp_path),
        features_root=features_root,
        work_root=tmp_path / "work",
        extractor=FakeExtractor(),
        access_check=lambda: None,
        provenance=_PROVENANCE,
    )

    manifest = json.loads((features_root / "prism2-diagnostic" / "STAD" / "manifest.json").read_text())
    assert manifest["soma_commit"] == "abc123"
    assert manifest["soma_version"] == "1.13.0"
    assert manifest["slide2vec_version"] == "5.8.2"
    assert manifest["prism2"] == {
        "repository": "paige-ai/Prism2",
        "revision": "450352d0ddc6b42b21ce20794ce0fbefe6b5a47a",
        "tile_encoder": "virchow2",
    }
    assert manifest["tiling"] == {"spacing_um": 0.5, "tile_size_px": 224}
    assert manifest["variant"] == "diagnostic"
    assert manifest["width"] == 3072
    assert manifest["cohort"] == "STAD"
    assert manifest["n_cases"] == 2


def test_missing_prism2_access_stops_the_script_before_any_extraction(tmp_path: Path) -> None:
    extractor = FakeExtractor()

    def no_access() -> None:
        raise AccessError("no access to paige-ai/Prism2")

    with pytest.raises(AccessError, match="paige-ai/Prism2"):
        extract_prism2(
            _cohorts(tmp_path),
            features_root=tmp_path / "features",
            work_root=tmp_path / "work",
            extractor=extractor,
            access_check=no_access,
            provenance=_PROVENANCE,
        )

    assert extractor.calls == []
    assert not (tmp_path / "features").exists()
    assert not (tmp_path / "work").exists()


def test_extraction_resumes_and_skips_cohorts_that_are_already_verified(tmp_path: Path) -> None:
    features_root = tmp_path / "features"
    cohorts = _cohorts(tmp_path)
    first = FakeExtractor()
    extract_prism2(
        cohorts,
        features_root=features_root,
        work_root=tmp_path / "work",
        extractor=first,
        access_check=lambda: None,
        provenance=_PROVENANCE,
    )
    # A cohort whose manifest never landed is redone; the rest is left alone.
    (features_root / "prism2-base" / "STAD" / "manifest.json").unlink()
    second = FakeExtractor()

    extract_prism2(
        cohorts,
        features_root=features_root,
        work_root=tmp_path / "work",
        extractor=second,
        access_check=lambda: None,
        provenance=_PROVENANCE,
    )

    assert [(variant, root.name) for _, variant, root in second.calls] == [("base", "STAD")]
    assert (features_root / "prism2-base" / "STAD" / "manifest.json").exists()


def test_extraction_fails_when_the_extractor_writes_the_wrong_width(tmp_path: Path) -> None:
    with pytest.raises(IntegrityError, match="width 1280, expected 2560"):
        extract_prism2(
            _cohorts(tmp_path),
            features_root=tmp_path / "features",
            work_root=tmp_path / "work",
            extractor=FakeExtractor(widths={"base": 1280, "diagnostic": 3072}),
            access_check=lambda: None,
            provenance=_PROVENANCE,
        )

    assert not (tmp_path / "features" / "prism2-base" / "COAD" / "manifest.json").exists()


def test_soma_version_below_the_slide_level_fix_is_refused() -> None:
    with pytest.raises(RuntimeError, match="1.12.1.*1.13.0"):
        require_soma_version("1.12.1")


def test_soma_version_at_or_above_the_slide_level_fix_is_accepted() -> None:
    require_soma_version("1.13.0")
    require_soma_version("1.14.2")


def test_command_defaults_to_the_shared_feature_root_and_both_variants() -> None:
    args = build_parser(Path("/repo")).parse_args([])

    assert args.features_root == Path("/data/pathology/projects/clement/mutation-prediction/features")
    assert args.work_root == Path("/repo/results/prism2-extraction")
    assert args.variants == ["base", "diagnostic"]
    assert args.cohorts == ["COAD", "STAD", "UCEC"]


class FakeHubApi:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.checked: list[tuple[str, str | None]] = []

    def auth_check(self, repo_id: str, *, revision: str | None = None) -> None:
        self.checked.append((repo_id, revision))
        if self.error is not None:
            raise self.error


def test_access_check_asks_the_hub_about_the_pinned_prism2_revision() -> None:
    api = FakeHubApi()

    check_prism2_access(api=api)

    assert api.checked == [("paige-ai/Prism2", "450352d0ddc6b42b21ce20794ce0fbefe6b5a47a")]


def test_access_check_turns_a_hub_refusal_into_a_clear_error() -> None:
    api = FakeHubApi(error=PermissionError("gated"))

    with pytest.raises(AccessError, match="paige-ai/Prism2"):
        check_prism2_access(api=api)
