from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src import PROJECT_ROOT
from src.data_utils.lppdbb.artifacts import (
    DEFAULT_INDEX_PATH,
    load_index,
    validate_index_rows,
)
from src.data_utils.lppdbb.records import DEFAULT_CSV_PATH
from src.generator.embedding_baseline1.generate_boltz_yaml import (
    write_boltz_yaml_for_row,
)

DEFAULT_EMBEDDING_ROOT = PROJECT_ROOT / "data" / "interim" / "lppdbb" / "embeddings"
DEFAULT_SPLIT_PATH = (
    PROJECT_ROOT / "data" / "interim" / "lppdbb" / "lppdbb_train_test_split.csv"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "processed" / "generator" / "embedding_baseline1"
)
DEFAULT_OUTPUT_NAME = "nearest_train_binders.csv"


@dataclass(frozen=True)
class EmbeddingRecord:
    row_idx: int
    item_id: str
    embedding_path: Path


@dataclass(frozen=True)
class LigandRecord:
    row_idx: int
    complex_id: str
    smiles: str
    seq: str
    embedding_path: Path


def _find_latest_embedding_dir(root: Path) -> Path:
    if not root.exists():
        raise FileNotFoundError(f"Embedding root not found: {root}")
    manifests = list(root.rglob("manifest.json"))
    if not manifests:
        raise FileNotFoundError(
            "No embedding manifests found. Provide --embedding-dir explicitly."
        )
    latest = max(manifests, key=lambda path: path.stat().st_mtime)
    return latest.parent


def _load_embedding_manifest(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_embedding_index(path: Path, output_dir: Path) -> list[EmbeddingRecord]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Embedding index CSV has no header: {path}")
        required = {"row_idx", "item_id", "embedding_path"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"Embedding index missing columns: {sorted(missing)} in {path}"
            )
        records: list[EmbeddingRecord] = []
        for row in reader:
            record_path = Path(row["embedding_path"])
            if not record_path.is_absolute():
                record_path = output_dir / record_path
            records.append(
                EmbeddingRecord(
                    row_idx=int(row["row_idx"]),
                    item_id=row["item_id"],
                    embedding_path=record_path,
                )
            )
    return records


def _build_embedding_lookup(
    embedding_records: list[EmbeddingRecord],
    match_by: str,
) -> tuple[dict[str | int, Path], int]:
    mapping: dict[str | int, Path] = {}
    duplicates = 0
    for record in embedding_records:
        key: str | int
        if match_by == "complex_id":
            key = record.item_id
        else:
            key = record.row_idx
        if key in mapping:
            duplicates += 1
            continue
        mapping[key] = record.embedding_path
    return mapping, duplicates


def _resolve_complex_id_column(fieldnames: Sequence[str]) -> str:
    for candidate in ("complex_id", "pdb_id", "id", "name"):
        if candidate in fieldnames:
            return candidate
    if "" in fieldnames:
        return ""
    return fieldnames[0]


def _load_sequences(csv_path: Path) -> dict[str, str]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {csv_path}")
        if "seq" not in reader.fieldnames:
            raise ValueError("CSV missing 'seq' column.")
        id_column = _resolve_complex_id_column(reader.fieldnames)
        sequences: dict[str, str] = {}
        for row in reader:
            complex_id = (row.get(id_column) or "").strip()
            if not complex_id:
                continue
            seq = (row.get("seq") or "").strip()
            if not seq:
                continue
            if complex_id in sequences:
                continue
            sequences[complex_id] = seq
    return sequences


def _load_split_assignments(path: Path) -> dict[int, dict[str, str]]:
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


def _build_split_labels(
    index_rows: list[dict[str, str]],
    split_rows: dict[int, dict[str, str]],
) -> list[str]:
    if len(split_rows) != len(index_rows):
        raise ValueError(
            "Split CSV count mismatch with index. "
            f"Index rows: {len(index_rows)}, split rows: {len(split_rows)}."
        )
    labels: list[str] = []
    for idx, index_row in enumerate(index_rows):
        split_row = split_rows.get(idx)
        if split_row is None:
            raise ValueError(f"Split CSV missing row_idx {idx}.")
        if split_row["complex_id"] != index_row["complex_id"]:
            raise ValueError(
                "Split complex_id mismatch at row "
                f"{idx}: {split_row['complex_id']} != {index_row['complex_id']}"
            )
        labels.append(split_row["split"])
    return labels


def _load_embedding_matrix(
    records: list[LigandRecord],
    *,
    desc: str,
) -> tuple[np.ndarray, list[LigandRecord], dict[str, int]]:
    vectors: list[np.ndarray] = []
    kept: list[LigandRecord] = []
    skipped_missing = 0
    skipped_empty = 0
    for record in tqdm(records, desc=desc):
        if not record.embedding_path.exists():
            skipped_missing += 1
            continue
        vector = np.load(record.embedding_path)
        vector = np.ravel(vector).astype(np.float32)
        if vector.size == 0:
            skipped_empty += 1
            continue
        vectors.append(vector)
        kept.append(record)

    if not vectors:
        raise RuntimeError("No embeddings loaded for analysis.")

    dim = vectors[0].shape[0]
    for vector in vectors:
        if vector.shape[0] != dim:
            raise ValueError("Embedding dimensionality mismatch across items.")

    matrix = np.vstack(vectors).astype(np.float32, copy=False)
    norms = np.linalg.norm(matrix, axis=1)
    nonzero_mask = norms > 0
    skipped_zero = int(np.sum(~nonzero_mask))
    if skipped_zero:
        matrix = matrix[nonzero_mask]
        kept = [row for row, keep in zip(kept, nonzero_mask) if keep]
        norms = norms[nonzero_mask]
    if matrix.shape[0] == 0:
        raise RuntimeError("No non-zero embeddings available after filtering.")

    matrix = matrix / norms[:, None]
    return (
        matrix,
        kept,
        {
            "missing": skipped_missing,
            "empty": skipped_empty,
            "zero": skipped_zero,
        },
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Retrieve nearest train ligand in embedding space and output its binder(s)."
        )
    )
    parser.add_argument(
        "--embedding-dir",
        type=Path,
        default=None,
        help="Embedding output directory (defaults to latest under embeddings root).",
    )
    parser.add_argument(
        "--embedding-root",
        type=Path,
        default=DEFAULT_EMBEDDING_ROOT,
        help="Root directory to search for embedding manifests.",
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default=DEFAULT_INDEX_PATH,
        help="Path to ligand index CSV.",
    )
    parser.add_argument(
        "--split-path",
        type=Path,
        default=DEFAULT_SPLIT_PATH,
        help="Path to train/test split CSV.",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help="Path to LP_PDBBind.csv with sequences.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for results.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=DEFAULT_OUTPUT_NAME,
        help="Output CSV filename.",
    )
    parser.add_argument(
        "--yaml-output-dir",
        type=Path,
        default=None,
        help="Output directory for Boltz YAML inputs.",
    )
    parser.add_argument(
        "--match-by",
        choices=("complex_id", "row_idx"),
        default="complex_id",
        help="Match embedding rows to index by complex_id or row_idx.",
    )
    parser.add_argument(
        "--max-test",
        type=int,
        default=None,
        help="Optional cap on number of test ligands.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    embedding_dir = args.embedding_dir
    if embedding_dir is None:
        embedding_dir = _find_latest_embedding_dir(args.embedding_root)
    if not embedding_dir.exists():
        raise FileNotFoundError(f"Embedding directory not found: {embedding_dir}")

    manifest = _load_embedding_manifest(embedding_dir / "manifest.json")
    if manifest is not None:
        task = str(manifest.get("task", ""))
        if task and task != "molecule":
            raise ValueError(
                f"Embedding task is {task}; this baseline requires molecule embeddings."
            )

    embedding_index_path = embedding_dir / "index.csv"
    if not embedding_index_path.exists():
        raise FileNotFoundError(
            f"Embedding index CSV not found: {embedding_index_path}"
        )

    embedding_records = _load_embedding_index(embedding_index_path, embedding_dir)
    if not embedding_records:
        raise RuntimeError("No embeddings found in index.")

    if not args.index_path.exists():
        raise FileNotFoundError(f"Index CSV not found: {args.index_path}")
    if not args.split_path.exists():
        raise FileNotFoundError(f"Split CSV not found: {args.split_path}")
    if not args.csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv_path}")

    index_rows = load_index(args.index_path)
    validate_index_rows(index_rows)
    split_rows = _load_split_assignments(args.split_path)
    split_labels = _build_split_labels(index_rows, split_rows)
    sequences = _load_sequences(args.csv_path)

    embedding_lookup, duplicates = _build_embedding_lookup(
        embedding_records,
        args.match_by,
    )

    train_records: list[LigandRecord] = []
    test_records: list[LigandRecord] = []
    missing_embeddings = 0
    for idx, index_row in enumerate(index_rows):
        split = split_labels[idx]
        complex_id = index_row["complex_id"]
        key: str | int
        if args.match_by == "complex_id":
            key = complex_id
        else:
            key = idx
        embedding_path = embedding_lookup.get(key)
        if embedding_path is None:
            missing_embeddings += 1
            continue
        record = LigandRecord(
            row_idx=idx,
            complex_id=complex_id,
            smiles=index_row.get("smiles", ""),
            seq=sequences.get(complex_id, ""),
            embedding_path=embedding_path,
        )
        if split == "train":
            train_records.append(record)
        elif split == "test":
            test_records.append(record)
        else:
            raise ValueError(f"Unknown split label: {split}")

    if not train_records or not test_records:
        raise RuntimeError("Train/test split is empty after filtering.")

    train_matrix, train_meta, train_skips = _load_embedding_matrix(
        train_records,
        desc="load train embeddings",
    )
    test_matrix, test_meta, test_skips = _load_embedding_matrix(
        test_records,
        desc="load test embeddings",
    )

    if args.max_test is not None and test_matrix.shape[0] > args.max_test:
        test_matrix = test_matrix[: args.max_test]
        test_meta = test_meta[: args.max_test]

    results = []
    for test_vector, test_record in tqdm(
        zip(test_matrix, test_meta, strict=True),
        total=len(test_meta),
        desc="nearest neighbor",
    ):
        sims = train_matrix @ test_vector
        nn_idx = int(np.argmax(sims))
        nn_sim = float(sims[nn_idx])
        train_record = train_meta[nn_idx]
        results.append(
            {
                "test_row_idx": test_record.row_idx,
                "test_complex_id": test_record.complex_id,
                "test_smiles": test_record.smiles,
                "nearest_train_row_idx": train_record.row_idx,
                "nearest_train_complex_id": train_record.complex_id,
                "nearest_train_smiles": train_record.smiles,
                "nearest_train_seq": train_record.seq,
                "embedding_similarity": f"{nn_sim:.6f}",
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    yaml_output_dir = args.yaml_output_dir
    if yaml_output_dir is None:
        yaml_output_dir = args.output_dir / "boltz_inputs"
    yaml_output_dir.mkdir(parents=True, exist_ok=True)

    yaml_counts: dict[str, int] = {}
    for row in results:
        yaml_path, status = write_boltz_yaml_for_row(
            row,
            output_dir=yaml_output_dir,
            overwrite=True,
            include_affinity=True,
        )
        yaml_counts[status] = yaml_counts.get(status, 0) + 1
        if yaml_path is None:
            row["yaml_path"] = ""
        else:
            try:
                row["yaml_path"] = str(yaml_path.relative_to(args.output_dir))
            except ValueError:
                row["yaml_path"] = str(yaml_path)
        row["yaml_status"] = status

    output_path = args.output_dir / args.output_name
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "test_row_idx",
                "test_complex_id",
                "test_smiles",
                "nearest_train_row_idx",
                "nearest_train_complex_id",
                "nearest_train_smiles",
                "nearest_train_seq",
                "embedding_similarity",
                "yaml_path",
                "yaml_status",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print(
        "Generated embedding baseline results. "
        f"train: {len(train_meta)}, test: {len(test_meta)}, "
        f"missing_embeddings: {missing_embeddings}, duplicates: {duplicates}, "
        f"output: {output_path}"
    )
    if yaml_counts:
        yaml_summary = ", ".join(
            f"{key}: {value}" for key, value in sorted(yaml_counts.items())
        )
        print(
            "Generated Boltz YAML inputs. "
            f"output_dir: {yaml_output_dir}, status: {yaml_summary}"
        )
    print(
        "Embedding skips. "
        f"train_missing: {train_skips['missing']}, train_empty: {train_skips['empty']}, "
        f"train_zero: {train_skips['zero']}, test_missing: {test_skips['missing']}, "
        f"test_empty: {test_skips['empty']}, test_zero: {test_skips['zero']}"
    )


if __name__ == "__main__":
    main()
