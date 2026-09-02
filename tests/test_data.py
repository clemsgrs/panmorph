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
