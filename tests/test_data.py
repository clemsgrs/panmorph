from panmorph.data import tss


def test_tss_extracts_the_tissue_source_site() -> None:
    assert tss("TCGA-AB-1234") == "AB"
