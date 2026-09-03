"""Extract PRISM2 features for the MSI cohorts with an injected extractor.

The script turns each committed label table into a soma dataset table. It runs one
extraction per PRISM2 output variant on the same tile cache. It mirrors the per-case files
into ``<features root>/prism2-<variant>/<cohort>/`` next to the PRISM features. It checks
the result and writes a provenance manifest. The extractor is a callable, so tests use a
fake that writes synthetic files. The real path needs soma 1.13.0 or newer.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Callable, Mapping, NamedTuple, Protocol, Sequence

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from panmorph.data import (  # noqa: E402
    FEATURES_ROOT as DEFAULT_FEATURES_ROOT,
    MSI_COHORTS,
    PRISM2_WIDTHS as VARIANTS,
    prism2_feature_set,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
MIN_SOMA_VERSION = (1, 13, 0)

DATASET_COLUMNS = ["sample_id", "image_path", "label", "mask_path"]
PRISM2_REPO = "paige-ai/Prism2"
PRISM2_REVISION = "450352d0ddc6b42b21ce20794ce0fbefe6b5a47a"
TILE_ENCODER = "virchow2"
TILING = {"spacing_um": 0.5, "tile_size_px": 224}
MANIFEST_SCHEMA = "panmorph.prism2-features/v1"

# (dataset csv, output variant, output root) -> directory of one <case_id>.pt per case.
Extractor = Callable[[Path, str, Path], Path]
CohortRegistry = Mapping[str, tuple[Path, str, Path]]


class AccessError(RuntimeError):
    """The local Hugging Face token cannot read the gated PRISM2 repository."""


class IntegrityError(RuntimeError):
    """The mirrored feature directory does not match the labelled case list."""


class HubApi(Protocol):
    """The one call this script makes on ``huggingface_hub.HfApi``."""

    def auth_check(self, repo_id: str) -> None: ...


def hub_refusal_type() -> type[BaseException]:
    """Return the error ``huggingface_hub`` raises when a token cannot read a repository."""
    from huggingface_hub.errors import HfHubHTTPError

    return HfHubHTTPError


def check_prism2_access(
    api: HubApi | None = None, *, refusal: type[BaseException] | None = None
) -> None:
    """Stop early when the cached Hugging Face token cannot read PRISM2.

    The Hub grants access to the repository as a whole, not to one revision. ``refusal``
    is the error type the Hub raises for a refusal. Tests inject it with a fake ``api``
    so that ``huggingface_hub`` is imported only on the real path.
    """
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()
    try:
        api.auth_check(PRISM2_REPO)
    except Exception as error:
        if not isinstance(error, refusal or hub_refusal_type()):
            raise
        raise AccessError(
            f"No access to the gated Hugging Face repository {PRISM2_REPO}. "
            "Request access on the model page and log in with `huggingface-cli login`. "
            f"Hub said: {error}"
        ) from error


def require_unique_case_ids(case_ids: Sequence[str], *, where: object) -> None:
    """Refuse a case list that names one case twice."""
    duplicates = sorted(case for case, count in Counter(case_ids).items() if count > 1)
    if duplicates:
        raise IntegrityError(f"{where}: duplicate case identifiers: {duplicates}")


def verify_features(directory: Path, case_ids: Sequence[str], *, width: int) -> None:
    """Check one feature file per case with the expected embedding width."""
    require_unique_case_ids(case_ids, where=directory)
    missing = sorted(case for case in case_ids if not (directory / f"{case}.pt").exists())
    if missing:
        raise IntegrityError(f"{directory}: missing feature files for cases: {missing}")
    for case in case_ids:
        tensor = torch.load(directory / f"{case}.pt", map_location="cpu", weights_only=True)
        found = int(tensor.reshape(-1).numel())
        if found != width:
            raise IntegrityError(
                f"{directory}: case {case} has embedding width {found}, expected {width}"
            )


def feature_set_dir(features_root: Path, variant: str, cohort: str) -> Path:
    """Return the directory the registered feature set reads for one cohort."""
    return prism2_feature_set(variant, features_root).dirs[cohort]


def mirror_features(feature_dir: Path, destination: Path, case_ids: Sequence[str]) -> None:
    """Copy one ``<case_id>.pt`` per case from the extractor output into ``destination``."""
    destination.mkdir(parents=True, exist_ok=True)
    for case in case_ids:
        shutil.copy2(feature_dir / f"{case}.pt", destination / f"{case}.pt")


def write_manifest(
    destination: Path,
    *,
    cohort: str,
    variant: str,
    case_ids: Sequence[str],
    provenance: Mapping[str, str],
) -> Path:
    """Record what produced the files in ``destination``."""
    payload = {
        "schema_version": MANIFEST_SCHEMA,
        "cohort": cohort,
        "variant": variant,
        "width": VARIANTS[variant],
        "n_cases": len(case_ids),
        "prism2": {
            "repository": PRISM2_REPO,
            "revision": PRISM2_REVISION,
            "tile_encoder": TILE_ENCODER,
        },
        "tiling": dict(TILING),
        "soma_commit": provenance["soma_commit"],
        "soma_version": provenance["soma_version"],
        "slide2vec_version": provenance["slide2vec_version"],
    }
    path = destination / "manifest.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def is_complete(destination: Path, case_ids: Sequence[str], *, width: int) -> bool:
    """Return True when a manifest exists and the mirrored files pass the check."""
    if not (destination / "manifest.json").exists():
        return False
    try:
        verify_features(destination, case_ids, width=width)
    except IntegrityError:
        return False
    return True


def build_dataset_table(label_csv: Path, label_column: str) -> pd.DataFrame:
    """Turn one label table into a soma dataset table with one row per labelled case."""
    labels = pd.read_csv(label_csv).dropna(subset=[label_column])
    require_unique_case_ids(labels["case_id"].astype(str).tolist(), where=label_csv)
    return pd.DataFrame(
        {
            "sample_id": labels["case_id"].astype(str).tolist(),
            "image_path": labels["wsi_path"].astype(str).tolist(),
            "label": labels[label_column].astype(int).tolist(),
            "mask_path": labels["mask_path"].astype(str).tolist(),
        },
        columns=DATASET_COLUMNS,
    )


def extract_prism2(
    cohorts: CohortRegistry,
    *,
    features_root: Path,
    work_root: Path,
    extractor: Extractor,
    access_check: Callable[[], None],
    provenance: Mapping[str, str],
    variants: Sequence[str] = tuple(VARIANTS),
) -> None:
    """Extract every variant for every cohort, mirror, verify, and record provenance."""
    access_check()
    for variant in variants:
        for cohort, (label_csv, label_column, _) in cohorts.items():
            table = build_dataset_table(label_csv, label_column)
            case_ids = table["sample_id"].tolist()
            destination = feature_set_dir(features_root, variant, cohort)
            if is_complete(destination, case_ids, width=VARIANTS[variant]):
                print(f"[{variant}/{cohort}] already verified, skipping")
                continue
            output_root = work_root / variant / cohort
            output_root.mkdir(parents=True, exist_ok=True)
            dataset_csv = output_root / "dataset.csv"
            table.to_csv(dataset_csv, index=False)
            feature_dir = extractor(dataset_csv, variant, output_root)
            mirror_features(feature_dir, destination, case_ids)
            verify_features(destination, case_ids, width=VARIANTS[variant])
            write_manifest(
                destination,
                cohort=cohort,
                variant=variant,
                case_ids=case_ids,
                provenance=provenance,
            )




def require_soma_version(version: str) -> None:
    """Refuse a soma older than 1.13.0, the release that fixed slide-level variants."""
    match = re.match(r"(\d+)\.(\d+)(?:\.(\d+))?", version)
    parts = tuple(int(part or 0) for part in match.groups()) if match else (0, 0, 0)
    if parts < MIN_SOMA_VERSION:
        wanted = ".".join(str(part) for part in MIN_SOMA_VERSION)
        raise RuntimeError(
            f"soma {version} is too old; this script needs soma {wanted} or newer "
            "(the release with the slide-level variant fix, clemsgrs/soma#454)."
        )


def require_pinned_prism2_revision(found: str) -> None:
    """Refuse a slide2vec that loads a PRISM2 revision other than the one in the manifest."""
    if found != PRISM2_REVISION:
        raise RuntimeError(
            f"slide2vec loads {PRISM2_REPO} at revision {found}, but this script records "
            f"{PRISM2_REVISION}. Update PRISM2_REVISION or pin slide2vec."
        )


class SomaBackend(NamedTuple):
    """The real extractor and the provenance of the environment that runs it."""

    extractor: Extractor
    provenance: dict[str, str]


def make_soma_backend(cache_root: Path) -> SomaBackend:
    """Build the real extractor. All variants share ``cache_root``, so tiling runs once.

    This is the only function that imports soma. Tests never call it.
    """
    from importlib.metadata import version

    import soma
    from soma.config import CacheConfig, EncoderConfig, ExecutionConfig, PreprocessingConfig
    from soma.dataset import Dataset
    from soma.extraction import FeatureExtractor
    from soma.provenance import soma_git_state
    from slide2vec.encoders.models import prism2 as slide2vec_prism2

    require_soma_version(soma.__version__)
    require_pinned_prism2_revision(slide2vec_prism2.PRISM2_REVISION)
    state = soma_git_state()
    commit = state.sha or "unknown"
    if state.dirty:
        commit += "-dirty"
    provenance = {
        "soma_commit": commit,
        "soma_version": soma.__version__,
        "slide2vec_version": version("slide2vec"),
    }

    def extract(dataset_csv: Path, variant: str, output_root: Path) -> Path:
        result = FeatureExtractor(
            Dataset(dataset_csv),
            EncoderConfig(name="prism2", output_variant=variant),
            PreprocessingConfig(
                requested_spacing_um=TILING["spacing_um"],
                requested_tile_size_px=TILING["tile_size_px"],
            ),
            execution=ExecutionConfig(),
            cache=CacheConfig(root_dir=cache_root),
            output_root=output_root,
        ).extract()
        return Path(result.artifacts.feature_dir)

    return SomaBackend(extract, provenance)


def build_parser(root: Path = REPO_ROOT) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-root", type=Path, default=DEFAULT_FEATURES_ROOT)
    parser.add_argument("--work-root", type=Path, default=root / "results" / "prism2-extraction")
    parser.add_argument(
        "--variants", nargs="+", choices=sorted(VARIANTS), default=list(VARIANTS)
    )
    parser.add_argument(
        "--cohorts", nargs="+", choices=sorted(MSI_COHORTS), default=sorted(MSI_COHORTS)
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    backend = make_soma_backend(args.work_root / "cache")
    cohorts = {name: MSI_COHORTS[name] for name in args.cohorts}
    extract_prism2(
        cohorts,
        features_root=args.features_root,
        work_root=args.work_root,
        extractor=backend.extractor,
        access_check=check_prism2_access,
        provenance=backend.provenance,
        variants=args.variants,
    )
    print(f"Saved verified PRISM2 features under {args.features_root}")


if __name__ == "__main__":
    main()
