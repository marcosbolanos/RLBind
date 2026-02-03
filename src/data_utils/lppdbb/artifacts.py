from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from src import PROJECT_ROOT
from src.data_utils.lppdbb.records import LigandRecord

DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "data" / "interim" / "lppdbb"
DEFAULT_INDEX_PATH = DEFAULT_ARTIFACT_DIR / "lppdbb_index.csv"
DEFAULT_MANIFEST_PATH = DEFAULT_ARTIFACT_DIR / "lppdbb_manifest.json"
MANIFEST_VERSION = 1


def compute_file_sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def index_rows_from_records(records: Sequence[LigandRecord]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for idx, record in enumerate(records):
        rows.append(
            {
                "row_idx": str(idx),
                "complex_id": record.complex_id,
                "smiles": record.smiles,
                "ligand_path": str(record.ligand_path),
            }
        )
    return rows


def write_index(rows: Sequence[Mapping[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["row_idx", "complex_id", "smiles", "ligand_path"],
        )
        writer.writeheader()
        writer.writerows(rows)


def load_index(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Index CSV has no header: {path}")
        rows = [dict(row) for row in reader]
    for row in rows:
        if "row_idx" in row:
            row["row_idx"] = str(int(row["row_idx"]))
    return rows


def validate_index_rows(rows: Sequence[Mapping[str, str]]) -> None:
    for expected, row in enumerate(rows):
        if str(row.get("row_idx")) != str(expected):
            raise ValueError(
                "Index row order mismatch. "
                f"Expected row_idx {expected}, got {row.get('row_idx')}."
            )


def validate_index_matches_records(
    rows: Sequence[Mapping[str, str]],
    records: Sequence[LigandRecord],
) -> None:
    if len(rows) != len(records):
        raise ValueError(
            f"Index row count mismatch. Expected {len(records)}, got {len(rows)}."
        )
    for idx, (row, record) in enumerate(zip(rows, records, strict=True)):
        if row.get("complex_id") != record.complex_id:
            raise ValueError(
                "Index complex_id mismatch at row "
                f"{idx}: {row.get('complex_id')} != {record.complex_id}"
            )


def build_manifest(
    *,
    source_csv_path: Path,
    source_csv_sha256: str,
    total_rows: int,
    retained: int,
    skipped_counts: Mapping[str, int],
    radius: int,
    n_bits: int,
    distance_dtype: str,
    num_pairs: int,
    index_path: Path,
    fingerprints_path: Path,
    distances_path: Path,
    clusters_path: Path | None = None,
    threshold: float | None = None,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "version": MANIFEST_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_csv": {
            "path": str(source_csv_path),
            "sha256": source_csv_sha256,
            "total_rows": total_rows,
            "retained": retained,
            "skipped_counts": dict(skipped_counts),
        },
        "index": {
            "path": str(index_path),
            "count": retained,
        },
        "fingerprints": {
            "path": str(fingerprints_path),
            "radius": radius,
            "n_bits": n_bits,
            "dtype": "uint8",
            "count": retained,
        },
        "distances": {
            "path": str(distances_path),
            "dtype": distance_dtype,
            "num_pairs": num_pairs,
        },
    }
    if clusters_path is not None:
        manifest["clusters"] = {
            "path": str(clusters_path),
            "threshold": threshold,
        }
    return manifest


def write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_manifest(
    manifest: Mapping[str, Any],
    *,
    csv_path: Path | None = None,
    num_items: int | None = None,
    radius: int | None = None,
    n_bits: int | None = None,
    index_path: Path | None = None,
    distances_path: Path | None = None,
    fingerprints_path: Path | None = None,
) -> None:
    if csv_path is not None:
        csv_path = csv_path.resolve()
        manifest_csv = Path(manifest["source_csv"]["path"]).resolve()
        if manifest_csv != csv_path:
            raise ValueError(
                f"Manifest CSV path mismatch. Expected {csv_path}, got {manifest_csv}."
            )
        manifest_sha = manifest["source_csv"]["sha256"]
        actual_sha = compute_file_sha256(csv_path)
        if manifest_sha != actual_sha:
            raise ValueError("Manifest CSV hash mismatch.")

    if num_items is not None:
        expected_pairs = num_items * (num_items - 1) // 2
        manifest_pairs = manifest["distances"]["num_pairs"]
        if manifest_pairs != expected_pairs:
            raise ValueError(
                "Manifest pair count mismatch. "
                f"Expected {expected_pairs}, got {manifest_pairs}."
            )

    if radius is not None:
        manifest_radius = manifest["fingerprints"]["radius"]
        if manifest_radius != radius:
            raise ValueError(
                f"Manifest radius mismatch. Expected {radius}, got {manifest_radius}."
            )

    if n_bits is not None:
        manifest_bits = manifest["fingerprints"]["n_bits"]
        if manifest_bits != n_bits:
            raise ValueError(
                f"Manifest n_bits mismatch. Expected {n_bits}, got {manifest_bits}."
            )

    if index_path is not None:
        manifest_index = Path(manifest["index"]["path"]).resolve()
        if manifest_index != index_path.resolve():
            raise ValueError("Manifest index path mismatch.")

    if distances_path is not None:
        manifest_distances = Path(manifest["distances"]["path"]).resolve()
        if manifest_distances != distances_path.resolve():
            raise ValueError("Manifest distances path mismatch.")

    if fingerprints_path is not None:
        manifest_fps = Path(manifest["fingerprints"]["path"]).resolve()
        if manifest_fps != fingerprints_path.resolve():
            raise ValueError("Manifest fingerprints path mismatch.")
