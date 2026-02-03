from __future__ import annotations

import csv
from pathlib import Path

from src import PROJECT_ROOT
from src.data_utils.lppdbb.artifacts import (
    DEFAULT_INDEX_PATH,
    DEFAULT_MANIFEST_PATH,
    load_index,
    load_manifest,
    validate_index_rows,
    validate_manifest,
)
from src.data_utils.lppdbb.rdkit.distances import (
    DEFAULT_FINGERPRINTS_PATH,
    load_fingerprints,
)

DEFAULT_SPLIT_PATH = (
    PROJECT_ROOT / "data" / "interim" / "lppdbb" / "lppdbb_train_test_split.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "plots" / "ood_figure_pack"


def require_rdkit_data_structs():
    try:
        from rdkit import DataStructs  # noqa: WPS433  # pyright: ignore[reportMissingImports]
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "RDKit is required for similarity computations. "
            "Install it with `uv sync --extra rdkit`."
        ) from exc
    return DataStructs


def require_rdkit_chem():
    try:
        from rdkit import Chem  # noqa: WPS433, TID252  # pyright: ignore[reportMissingImports]
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "RDKit is required for scaffold and descriptor computations. "
            "Install it with `uv sync --extra rdkit`."
        ) from exc
    return Chem


def require_rdkit_scaffolds():
    try:
        from rdkit.Chem.Scaffolds import (  # noqa: WPS433  # pyright: ignore[reportMissingImports]
            MurckoScaffold,
        )
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "RDKit is required for Murcko scaffold computations. "
            "Install it with `uv sync --extra rdkit`."
        ) from exc
    return MurckoScaffold


def require_rdkit_descriptors():
    try:
        from rdkit.Chem import (  # noqa: WPS433  # pyright: ignore[reportMissingImports]
            Crippen,
            Descriptors,
            Lipinski,
            rdMolDescriptors,
        )
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "RDKit is required for property descriptor computations. "
            "Install it with `uv sync --extra rdkit`."
        ) from exc
    return Descriptors, Crippen, Lipinski, rdMolDescriptors


def require_umap():
    try:
        import umap  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "UMAP is required for the UMAP plots. Install it with `uv add umap-learn`."
        ) from exc
    return umap


def require_scipy_stats():
    try:
        from scipy import stats  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "SciPy is required for KDE/KS statistics. Install it with `uv add scipy`."
        ) from exc
    return stats


def load_split_assignments(path: Path) -> dict[int, dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Split CSV has no header: {path}")
        required = {"row_idx", "complex_id", "split"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Split CSV missing columns: {sorted(missing)}")
        rows: dict[int, dict[str, str]] = {}
        for row in reader:
            row_idx = int(row["row_idx"])
            rows[row_idx] = {
                "complex_id": row["complex_id"],
                "split": row["split"],
            }
    return rows


def load_index_rows(path: Path) -> list[dict[str, str]]:
    index_rows = load_index(path)
    validate_index_rows(index_rows)
    return index_rows


def validate_split_matches_index(
    index_rows: list[dict[str, str]],
    split_rows: dict[int, dict[str, str]],
) -> None:
    if len(split_rows) != len(index_rows):
        raise ValueError(
            "Split CSV count mismatch with index. "
            f"Index rows: {len(index_rows)}, split rows: {len(split_rows)}."
        )
    for idx, index_row in enumerate(index_rows):
        split_row = split_rows.get(idx)
        if split_row is None:
            raise ValueError(f"Split CSV missing row_idx {idx}.")
        if split_row["complex_id"] != index_row["complex_id"]:
            raise ValueError(
                "Split complex_id mismatch at row "
                f"{idx}: {split_row['complex_id']} != {index_row['complex_id']}"
            )


def maybe_validate_manifest(
    manifest_path: Path,
    *,
    index_path: Path,
    fingerprints_path: Path | None = None,
) -> None:
    if not manifest_path.exists():
        return
    manifest = load_manifest(manifest_path)
    validate_manifest(
        manifest,
        num_items=len(load_index(index_path)),
        index_path=index_path,
        fingerprints_path=fingerprints_path,
    )


def load_fingerprints_checked(
    fingerprints_path: Path,
    *,
    index_rows: list[dict[str, str]],
) -> list[object]:
    fingerprints = load_fingerprints(fingerprints_path)
    if len(fingerprints) != len(index_rows):
        raise ValueError(
            "Fingerprint count mismatch with index. "
            f"Index rows: {len(index_rows)}, fingerprints: {len(fingerprints)}."
        )
    return fingerprints


def split_fingerprints(
    fingerprints: list[object],
    split_rows: dict[int, dict[str, str]],
) -> tuple[list[object], list[object]]:
    train_fps: list[object] = []
    test_fps: list[object] = []
    for idx, fp in enumerate(fingerprints):
        split = split_rows[idx]["split"]
        if split == "train":
            train_fps.append(fp)
        elif split == "test":
            test_fps.append(fp)
        else:
            raise ValueError(f"Unknown split label: {split}")
    return train_fps, test_fps


def split_indices(split_rows: dict[int, dict[str, str]]) -> tuple[list[int], list[int]]:
    train_indices: list[int] = []
    test_indices: list[int] = []
    for idx, row in split_rows.items():
        split = row["split"]
        if split == "train":
            train_indices.append(idx)
        elif split == "test":
            test_indices.append(idx)
        else:
            raise ValueError(f"Unknown split label: {split}")
    return train_indices, test_indices


def split_smiles(
    index_rows: list[dict[str, str]],
    split_rows: dict[int, dict[str, str]],
) -> tuple[list[str], list[str]]:
    train_smiles: list[str] = []
    test_smiles: list[str] = []
    for idx, row in enumerate(index_rows):
        split = split_rows[idx]["split"]
        if split == "train":
            train_smiles.append(row["smiles"])
        elif split == "test":
            test_smiles.append(row["smiles"])
        else:
            raise ValueError(f"Unknown split label: {split}")
    return train_smiles, test_smiles


__all__ = [
    "DEFAULT_FINGERPRINTS_PATH",
    "DEFAULT_INDEX_PATH",
    "DEFAULT_MANIFEST_PATH",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_SPLIT_PATH",
    "load_fingerprints_checked",
    "load_index_rows",
    "load_split_assignments",
    "maybe_validate_manifest",
    "require_rdkit_data_structs",
    "require_rdkit_chem",
    "require_rdkit_descriptors",
    "require_rdkit_scaffolds",
    "require_scipy_stats",
    "require_umap",
    "split_fingerprints",
    "split_indices",
    "split_smiles",
    "validate_split_matches_index",
]
