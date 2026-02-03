from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Any, Mapping

from src import PROJECT_ROOT
from src.data_utils.lppdbb.artifacts import (
    DEFAULT_INDEX_PATH,
    DEFAULT_MANIFEST_PATH,
    load_index,
    load_manifest,
    validate_index_rows,
    validate_manifest,
    write_manifest,
)
from src.data_utils.lppdbb.rdkit.distances import (
    DEFAULT_FINGERPRINTS_PATH,
    load_fingerprints,
)

DEFAULT_CLUSTERS_PATH = (
    PROJECT_ROOT / "data" / "interim" / "lppdbb" / "lppdbb_ligand_clusters.csv"
)
DEFAULT_SPLIT_PATH = (
    PROJECT_ROOT / "data" / "interim" / "lppdbb" / "lppdbb_train_test_split.csv"
)
NN_SIM_THRESHOLD = 0.65


def _require_rdkit():
    try:
        from rdkit import DataStructs  # noqa: WPS433  # pyright: ignore[reportMissingImports]
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "RDKit is required for NN-based split filtering. "
            "Install it with `uv sync --extra rdkit`."
        ) from exc
    return DataStructs


def load_cluster_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Cluster CSV has no header: {path}")
        required = {
            "complex_id",
            "ligand_path",
            "smiles",
            "cluster_id",
            "cluster_size",
        }
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Cluster CSV missing columns: {sorted(missing)}")
        rows: list[dict[str, Any]] = []
        for row in reader:
            row["cluster_id"] = int(row["cluster_id"])
            row["cluster_size"] = int(row["cluster_size"])
            rows.append(row)
    return rows


def assign_cluster_splits(
    cluster_sizes: Mapping[int, int],
    *,
    test_fraction: float,
    seed: int,
) -> dict[int, str]:
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1.")

    rng = random.Random(seed)
    cluster_ids = list(cluster_sizes.keys())
    rng.shuffle(cluster_ids)

    total = sum(cluster_sizes.values())
    target = round(total * test_fraction)
    remaining = total
    test_count = 0
    assignments: dict[int, str] = {}
    for cluster_id in cluster_ids:
        size = cluster_sizes[cluster_id]
        remaining -= size
        need = target - test_count
        if need <= 0:
            assignments[cluster_id] = "train"
            continue
        if remaining < need:
            assignments[cluster_id] = "test"
            test_count += size
            continue
        if test_count + size <= target:
            assignments[cluster_id] = "test"
            test_count += size
        else:
            assignments[cluster_id] = "train"
    return assignments


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate train/test splits based on ligand clusters."
    )
    parser.add_argument(
        "--clusters-path",
        type=Path,
        default=DEFAULT_CLUSTERS_PATH,
        help="Path to ligand cluster CSV.",
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default=DEFAULT_INDEX_PATH,
        help="Path to ligand index CSV.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Path to artifact manifest JSON.",
    )
    parser.add_argument(
        "--fingerprints-path",
        type=Path,
        default=DEFAULT_FINGERPRINTS_PATH,
        help="Path to cached fingerprints (.npy).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_SPLIT_PATH,
        help="Output CSV path for train/test split.",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="Fraction of items assigned to test split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed for cluster assignment.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.output.exists() and not args.overwrite:
        print(
            f"Split file already exists. Use --overwrite to replace it: {args.output}"
        )
        return
    if not args.clusters_path.exists():
        raise FileNotFoundError(f"Cluster CSV not found: {args.clusters_path}")
    if not args.index_path.exists():
        raise FileNotFoundError(
            "Index CSV not found. Run the distances module to generate it."
        )
    if not args.manifest_path.exists():
        raise FileNotFoundError(
            "Manifest JSON not found. Run the distances module to generate it."
        )
    if not args.fingerprints_path.exists():
        raise FileNotFoundError(
            "Fingerprints .npy not found. Run the distances module to generate it."
        )

    index_rows = load_index(args.index_path)
    validate_index_rows(index_rows)

    manifest = load_manifest(args.manifest_path)
    validate_manifest(
        manifest,
        num_items=len(index_rows),
        index_path=args.index_path,
        fingerprints_path=args.fingerprints_path,
    )

    cluster_rows = load_cluster_rows(args.clusters_path)
    by_complex_id: dict[str, dict[str, Any]] = {}
    cluster_sizes: dict[int, int] = {}
    for row in cluster_rows:
        complex_id = str(row["complex_id"])
        if complex_id in by_complex_id:
            raise ValueError(f"Duplicate complex_id in clusters: {complex_id}")
        by_complex_id[complex_id] = row
        cluster_id = int(row["cluster_id"])
        cluster_sizes[cluster_id] = int(row["cluster_size"])

    if len(by_complex_id) != len(index_rows):
        raise ValueError(
            "Cluster CSV count mismatch with index. "
            f"Index rows: {len(index_rows)}, cluster rows: {len(by_complex_id)}."
        )

    assignments = assign_cluster_splits(
        cluster_sizes,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )

    split_by_idx: list[str] = []
    for index_row in index_rows:
        complex_id = index_row["complex_id"]
        cluster_row = by_complex_id.get(complex_id)
        if cluster_row is None:
            raise ValueError(f"Cluster CSV missing complex_id from index: {complex_id}")
        cluster_id = int(cluster_row["cluster_id"])
        split_by_idx.append(assignments[cluster_id])

    fingerprints = load_fingerprints(args.fingerprints_path)
    if len(fingerprints) != len(index_rows):
        raise ValueError(
            "Fingerprint count mismatch with index. "
            f"Index rows: {len(index_rows)}, fingerprints: {len(fingerprints)}."
        )
    train_indices = [idx for idx, split in enumerate(split_by_idx) if split == "train"]
    test_indices = [idx for idx, split in enumerate(split_by_idx) if split == "test"]
    if not train_indices:
        raise RuntimeError("NN filtering requires at least one train molecule.")

    DataStructs = _require_rdkit()
    moved_total = 0
    passes = 0
    if test_indices:
        while True:
            train_fps = [fingerprints[idx] for idx in train_indices]
            moved_this_pass = 0
            remaining_test: list[int] = []
            for idx in test_indices:
                sims = DataStructs.BulkTanimotoSimilarity(
                    fingerprints[idx],
                    train_fps,
                )
                max_sim = float(max(sims)) if sims else 0.0
                if max_sim > NN_SIM_THRESHOLD:
                    split_by_idx[idx] = "train"
                    train_indices.append(idx)
                    moved_this_pass += 1
                else:
                    remaining_test.append(idx)
            test_indices = remaining_test
            moved_total += moved_this_pass
            passes += 1
            if moved_this_pass == 0:
                break

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "row_idx",
                "complex_id",
                "split",
                "cluster_id",
                "cluster_size",
                "ligand_path",
                "smiles",
            ],
        )
        writer.writeheader()
        train_count = 0
        test_count = 0
        for idx, index_row in enumerate(index_rows):
            complex_id = index_row["complex_id"]
            cluster_row = by_complex_id.get(complex_id)
            if cluster_row is None:
                raise ValueError(
                    f"Cluster CSV missing complex_id from index: {complex_id}"
                )
            cluster_id = int(cluster_row["cluster_id"])
            split = split_by_idx[idx]
            if split == "test":
                test_count += 1
            else:
                train_count += 1
            writer.writerow(
                {
                    "row_idx": index_row["row_idx"],
                    "complex_id": complex_id,
                    "split": split,
                    "cluster_id": cluster_id,
                    "cluster_size": int(cluster_row["cluster_size"]),
                    "ligand_path": cluster_row["ligand_path"],
                    "smiles": cluster_row["smiles"],
                }
            )

    split_entry = {
        "path": str(args.output),
        "test_fraction": args.test_fraction,
        "seed": args.seed,
        "strategy": "cluster",
        "clusters_path": str(args.clusters_path),
        "nn_similarity_threshold": NN_SIM_THRESHOLD,
        "nn_moved_to_train": moved_total,
        "nn_passes": passes,
    }
    splits = manifest.get("splits")
    if splits is None:
        manifest["splits"] = [split_entry]
    elif isinstance(splits, list):
        splits.append(split_entry)
        manifest["splits"] = splits
    else:
        manifest["splits"] = [splits, split_entry]
    write_manifest(args.manifest_path, manifest)

    total = train_count + test_count
    print(
        "Generated train/test split. "
        f"total: {total}, "
        f"train: {train_count}, "
        f"test: {test_count}, "
        f"nn_moved: {moved_total}, "
        f"nn_threshold: {NN_SIM_THRESHOLD}, "
        f"output: {args.output}"
    )


if __name__ == "__main__":
    main()
