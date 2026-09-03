from panmorph.data import tss


def test_tss_extracts_the_tissue_source_site() -> None:
    assert tss("TCGA-AB-1234") == "AB"


def test_gate_bootstrap_interval_uses_the_shared_stratified_schedule() -> None:
    import numpy as np

    from panmorph.metrics import (
        bootstrap_auc,
        stratified_bootstrap_auc_ci,
        stratified_bootstrap_indices,
    )

    labels = np.asarray([1, 0, 1, 0, 0, 0, 1, 0])
    scores = np.asarray([0.9, 0.2, 0.7, 0.4, 0.1, 0.8, 0.6, 0.3])

    lower, upper, stats = stratified_bootstrap_auc_ci(
        labels, scores, key=("STAD",), n_boot=200, seed=0
    )
    schedule = stratified_bootstrap_indices(labels, seed=0, key=("STAD",), n_replicates=200)

    assert np.array_equal(stats, bootstrap_auc(labels, scores, schedule))
    assert np.all(np.sum(labels[schedule] == 1, axis=1) == 3)
    assert 0.0 <= lower <= upper <= 1.0


def test_probe_fit_is_independent_of_the_outer_blas_thread_count() -> None:
    import numpy as np
    from threadpoolctl import threadpool_limits

    from panmorph.probe import fit_predict

    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 64)).astype(np.float32)
    y = (X[:, 0] + 0.5 * rng.normal(size=120) > 0).astype(int)
    Xte = rng.normal(size=(30, 64)).astype(np.float32)

    with threadpool_limits(limits=1):
        single = fit_predict(X, y, Xte)
    with threadpool_limits(limits=8):
        many = fit_predict(X, y, Xte)

    assert np.array_equal(single, many)


def _write_synthetic_features(root, cohort: str, width: int, case_ids) -> None:
    import torch

    directory = root / cohort
    directory.mkdir(parents=True)
    for index, case_id in enumerate(case_ids):
        torch.save(torch.full((width,), float(index)), directory / f"{case_id}.pt")


def _synthetic_registry(tmp_path):
    import pandas as pd

    from panmorph.data import FeatureSet

    case_ids = ["TCGA-AA-0001", "TCGA-AB-0002", "TCGA-AA-0003"]
    labels = tmp_path / "labels.csv"
    pd.DataFrame({"case_id": case_ids, "msi_high": [1, 0, 0]}).to_csv(labels, index=False)
    features_root = tmp_path / "features"
    feature_sets = {
        "synthetic": FeatureSet(
            name="synthetic",
            extractor="SYNTHETIC",
            width=8,
            dirs={"COAD": features_root / "synthetic" / "COAD"},
            identities={"COAD": "synthetic/COAD"},
        ),
    }
    registry = {"COAD": (labels, "msi_high", tmp_path / "unused")}
    return registry, feature_sets, features_root, case_ids


def test_named_feature_set_loads_synthetic_files_of_the_registered_width(tmp_path) -> None:
    import numpy as np

    from panmorph.data import load_cohort

    registry, feature_sets, features_root, case_ids = _synthetic_registry(tmp_path)
    _write_synthetic_features(features_root / "synthetic", "COAD", 8, case_ids)

    cohort = load_cohort("COAD", registry, features="synthetic", feature_sets=feature_sets)

    assert cohort.X.shape == (3, 8)
    assert cohort.X.dtype == np.float32
    assert list(cohort.case_ids) == case_ids
    assert list(cohort.y) == [1, 0, 0]
    assert list(cohort.sites) == ["AA", "AB", "AA"]
    assert cohort.X[2, 0] == 2.0


def test_unknown_feature_set_name_is_rejected(tmp_path) -> None:
    import pytest

    from panmorph.data import load_cohort

    registry, feature_sets, _, _ = _synthetic_registry(tmp_path)

    with pytest.raises(ValueError, match="unknown feature set 'nope'"):
        load_cohort("COAD", registry, features="nope", feature_sets=feature_sets)


def test_width_mismatch_is_rejected_with_the_case_identifier(tmp_path) -> None:
    import pytest

    from panmorph.data import load_cohort

    registry, feature_sets, features_root, case_ids = _synthetic_registry(tmp_path)
    _write_synthetic_features(features_root / "synthetic", "COAD", 5, case_ids)

    with pytest.raises(ValueError, match="TCGA-AA-0001.*width 5.*expected 8"):
        load_cohort("COAD", registry, features="synthetic", feature_sets=feature_sets)


def test_registry_holds_prism_default_and_both_prism2_sets() -> None:
    from panmorph.data import DEFAULT_FEATURE_SET, FEATURE_SETS, MSI_COHORTS, ROOT

    assert DEFAULT_FEATURE_SET == "prism"
    prism = FEATURE_SETS["prism"]
    assert (prism.extractor, prism.width) == ("PRISM", 1280)
    assert prism.dirs == {name: MSI_COHORTS[name][2] for name in MSI_COHORTS}
    assert prism.identities == {"COAD": "lxbzb8rd", "UCEC": "kooqa1ym", "STAD": "oowdp902"}
    base, diagnostic = FEATURE_SETS["prism2-base"], FEATURE_SETS["prism2-diagnostic"]
    assert (base.extractor, base.width) == ("PRISM2", 2560)
    assert (diagnostic.extractor, diagnostic.width) == ("PRISM2", 3072)
    assert base.dirs["STAD"] == ROOT / "features" / "prism2-base" / "STAD"
    assert diagnostic.dirs["UCEC"] == ROOT / "features" / "prism2-diagnostic" / "UCEC"
