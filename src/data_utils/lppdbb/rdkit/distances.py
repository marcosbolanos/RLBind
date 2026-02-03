from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from src import PROJECT_ROOT
from src.data_utils.lppdbb.artifacts import (
    DEFAULT_INDEX_PATH,
    DEFAULT_MANIFEST_PATH,
    build_manifest,
    compute_file_sha256,
    index_rows_from_records,
    load_index,
    load_manifest,
    validate_index_matches_records,
    validate_index_rows,
    validate_manifest,
    write_index,
    write_manifest,
)
from src.data_utils.lppdbb.records import DEFAULT_CSV_PATH
from src.data_utils.lppdbb.rdkit.preprocessing import preprocess_ligands_with_report

DEFAULT_DISTANCES_PATH = (
    PROJECT_ROOT / "data" / "interim" / "lppdbb" / "lppdbb_distances.npy"
)
DEFAULT_FINGERPRINTS_PATH = (
    PROJECT_ROOT / "data" / "interim" / "lppdbb" / "lppdbb_fingerprints.npy"
)
DEFAULT_LOG_PATH = (
    PROJECT_ROOT / "data" / "interim" / "lppdbb" / "lppdbb_preprocess_log.csv"
)


def _require_rdkit():
    try:
        from rdkit import DataStructs  # noqa: WPS433  # pyright: ignore[reportMissingImports]
        from rdkit.Chem import (  # noqa: WPS433  # pyright: ignore[reportMissingImports]
            rdFingerprintGenerator,
        )
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "RDKit is required for LP-PDBBind distance computation. "
            "Install it with `uv sync --extra rdkit`."
        ) from exc
    return DataStructs, rdFingerprintGenerator


def expected_num_pairs(num_items: int) -> int:
    return num_items * (num_items - 1) // 2


def compute_morgan_fingerprints(
    mols: Iterable[Any],
    *,
    radius: int = 2,
    n_bits: int = 2048,
) -> list[Any]:
    DataStructs, rdFingerprintGenerator = _require_rdkit()
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=n_bits,
    )
    return [generator.GetFingerprint(mol) for mol in mols]


def save_fingerprints(
    fingerprints: list[Any],
    path: str | Path,
) -> np.ndarray:
    DataStructs, _ = _require_rdkit()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not fingerprints:
        return np.empty((0, 0), dtype=np.uint8)

    first = DataStructs.BitVectToBinaryText(fingerprints[0])
    row_len = len(first)
    array = np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=np.uint8,
        shape=(len(fingerprints), row_len),
    )
    array[0] = np.frombuffer(first, dtype=np.uint8)
    for idx, fp in enumerate(fingerprints[1:], start=1):
        array[idx] = np.frombuffer(
            DataStructs.BitVectToBinaryText(fp),
            dtype=np.uint8,
        )
    array.flush()
    return array


def load_fingerprints(
    path: str | Path,
    *,
    n_bits: int | None = None,
) -> list[Any]:
    DataStructs, _ = _require_rdkit()
    data = np.load(path, mmap_mode="r")
    if data.ndim != 2:
        raise ValueError(f"Invalid fingerprint array shape: {data.shape}")
    fingerprints = [DataStructs.CreateFromBinaryText(row.tobytes()) for row in data]
    if n_bits is not None and fingerprints:
        fp_bits = fingerprints[0].GetNumBits()
        if fp_bits != n_bits:
            raise ValueError(
                "Fingerprint bit size mismatch. "
                f"Expected {n_bits}, found {fp_bits} in {path}."
            )
    return fingerprints


def compute_pairwise_distances(
    mols: Iterable[Any],
    *,
    radius: int = 2,
    n_bits: int = 2048,
    output_path: str | Path | None = None,
    fingerprints_path: str | Path | None = None,
) -> np.ndarray:
    DataStructs, _ = _require_rdkit()
    mols = list(mols)
    if not mols:
        return np.empty(0, dtype=np.float32)

    fps: list[Any]
    if fingerprints_path is not None:
        fingerprints_path = Path(fingerprints_path)
        if fingerprints_path.exists():
            fps = load_fingerprints(fingerprints_path, n_bits=n_bits)
        else:
            fps = compute_morgan_fingerprints(mols, radius=radius, n_bits=n_bits)
            save_fingerprints(fps, fingerprints_path)
    else:
        fps = compute_morgan_fingerprints(mols, radius=radius, n_bits=n_bits)
    num_pairs = expected_num_pairs(len(fps))
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        distances: np.ndarray = np.lib.format.open_memmap(
            output_path,
            mode="w+",
            dtype=np.float32,
            shape=(num_pairs,),
        )
    else:
        distances = np.empty(num_pairs, dtype=np.float32)

    cursor = 0
    for idx in range(1, len(fps)):
        similarities = DataStructs.BulkTanimotoSimilarity(fps[idx], fps[:idx])
        end = cursor + idx
        distances[cursor:end] = 1.0 - np.asarray(similarities, dtype=np.float32)
        cursor = end

    if isinstance(distances, np.memmap):
        distances.flush()
    return distances


def load_pairwise_distances(
    path: str | Path,
    *,
    num_items: int | None = None,
    mmap_mode: Literal["r", "r+", "w+", "c"] | None = "r",
) -> np.ndarray:
    path = Path(path)
    distances = np.load(
        path,
        mmap_mode=cast(Literal["r", "r+", "w+", "c"] | None, mmap_mode),
    )
    if num_items is not None:
        expected = expected_num_pairs(num_items)
        if distances.shape[0] != expected:
            raise ValueError(
                "Distance file size mismatch. "
                f"Expected {expected} pairs for {num_items} items, "
                f"found {distances.shape[0]} in {path}."
            )
    return distances


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute LP-PDBBind pairwise Tanimoto distances."
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help="Path to LP_PDBBind.csv for SMILES-based preprocessing.",
    )
    parser.add_argument(
        "--log-output",
        type=Path,
        default=DEFAULT_LOG_PATH,
        help="Output CSV path for preprocessing failures.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_DISTANCES_PATH,
        help="Output .npy path for condensed distances.",
    )
    parser.add_argument(
        "--index-output",
        type=Path,
        default=DEFAULT_INDEX_PATH,
        help="Output CSV path for ligand index ordering.",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Output JSON path for artifact manifest.",
    )
    parser.add_argument(
        "--fingerprints-output",
        type=Path,
        default=DEFAULT_FINGERPRINTS_PATH,
        help="Output .npy path for fingerprints.",
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=2,
        help="Morgan fingerprint radius.",
    )
    parser.add_argument(
        "--n-bits",
        type=int,
        default=2048,
        help="Morgan fingerprint bit size.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    distances_exist = args.output.exists()
    fingerprints_exist = args.fingerprints_output.exists()
    index_exists = args.index_output.exists()
    manifest_exists = args.manifest_output.exists()
    if (
        distances_exist
        and fingerprints_exist
        and index_exists
        and manifest_exists
        and not args.overwrite
    ):
        print(
            "Distances and fingerprints already exist. Use --overwrite to replace them."
        )
        return

    records, skipped, report = preprocess_ligands_with_report(
        args.csv_path,
        report_path=args.log_output,
    )
    if not records:
        raise RuntimeError("No ligands were loaded. Check the dataset path.")

    num_mols = len(records)
    num_pairs = expected_num_pairs(num_mols)

    index_rows = index_rows_from_records(records)
    if index_exists and not args.overwrite:
        existing_rows = load_index(args.index_output)
        validate_index_rows(existing_rows)
        validate_index_matches_records(existing_rows, records)
    else:
        write_index(index_rows, args.index_output)

    if manifest_exists and not args.overwrite:
        existing_manifest = load_manifest(args.manifest_output)
        validate_manifest(
            existing_manifest,
            csv_path=Path(args.csv_path),
            num_items=num_mols,
            radius=args.radius,
            n_bits=args.n_bits,
            index_path=args.index_output,
            distances_path=args.output,
            fingerprints_path=args.fingerprints_output,
        )

    if not fingerprints_exist or args.overwrite:
        fingerprints = compute_morgan_fingerprints(
            [record.mol for record in records],
            radius=args.radius,
            n_bits=args.n_bits,
        )
        save_fingerprints(fingerprints, args.fingerprints_output)
    else:
        fingerprints = load_fingerprints(
            args.fingerprints_output,
            n_bits=args.n_bits,
        )

    distances = None
    if not distances_exist or args.overwrite:
        distances = compute_pairwise_distances(
            [record.mol for record in records],
            radius=args.radius,
            n_bits=args.n_bits,
            output_path=args.output,
            fingerprints_path=args.fingerprints_output,
        )

    source_hash = compute_file_sha256(Path(args.csv_path))
    manifest = build_manifest(
        source_csv_path=Path(args.csv_path),
        source_csv_sha256=source_hash,
        total_rows=report.total_rows,
        retained=report.retained,
        skipped_counts=report.skipped_counts,
        radius=args.radius,
        n_bits=args.n_bits,
        distance_dtype="float32",
        num_pairs=num_pairs,
        index_path=args.index_output,
        fingerprints_path=args.fingerprints_output,
        distances_path=args.output,
    )
    if not manifest_exists or args.overwrite:
        write_manifest(args.manifest_output, manifest)

    print(
        "Computed fingerprints and distances. "
        f"ligands: {num_mols}, "
        f"pairs: {num_pairs}, "
        f"skipped: {len(skipped)}, "
        f"fingerprints: {args.fingerprints_output}, "
        f"distances: {args.output}, "
        f"index: {args.index_output}, "
        f"manifest: {args.manifest_output}, "
        f"log: {report.log_path}"
    )
    if distances is not None and distances.shape[0] != num_pairs:
        raise RuntimeError("Distance vector size mismatch after computation.")


if __name__ == "__main__":
    main()
