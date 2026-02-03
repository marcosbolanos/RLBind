from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "plots" / "lppdbb_train_test_tanimoto.png"
)


def _require_rdkit():
    try:
        from rdkit import DataStructs  # noqa: WPS433  # pyright: ignore[reportMissingImports]
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "RDKit is required for Tanimoto similarity plots. "
            "Install it with `uv sync --extra rdkit`."
        ) from exc
    return DataStructs


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot train/test or train/train Tanimoto similarity distribution."
    )
    parser.add_argument(
        "--split-path",
        type=Path,
        default=DEFAULT_SPLIT_PATH,
        help="Path to train/test split CSV.",
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
        default=DEFAULT_OUTPUT_PATH,
        help="Output plot path.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=50,
        help="Number of histogram bins.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=("train-test", "train-train"),
        default="train-test",
        help="Pairing mode for similarity computation.",
    )
    parser.add_argument(
        "--max-train",
        type=int,
        default=None,
        help="Optional cap on train set size.",
    )
    parser.add_argument(
        "--max-test",
        type=int,
        default=None,
        help="Optional cap on test set size.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.split_path.exists():
        raise FileNotFoundError(f"Split CSV not found: {args.split_path}")
    if not args.index_path.exists():
        raise FileNotFoundError(f"Index CSV not found: {args.index_path}")
    if not args.fingerprints_path.exists():
        raise FileNotFoundError(
            "Fingerprints .npy not found. Run the distances module to generate it."
        )

    index_rows = load_index(args.index_path)
    validate_index_rows(index_rows)
    split_rows = load_split_assignments(args.split_path)

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

    if args.manifest_path.exists():
        manifest = load_manifest(args.manifest_path)
        validate_manifest(
            manifest,
            num_items=len(index_rows),
            index_path=args.index_path,
            fingerprints_path=args.fingerprints_path,
        )

    fingerprints = load_fingerprints(args.fingerprints_path)
    train_fps = []
    test_fps = []
    for idx in range(len(fingerprints)):
        split = split_rows[idx]["split"]
        if split == "train":
            train_fps.append(fingerprints[idx])
        elif split == "test":
            test_fps.append(fingerprints[idx])
        else:
            raise ValueError(f"Unknown split label: {split}")

    if args.max_train is not None:
        train_fps = train_fps[: args.max_train]
    if args.max_test is not None:
        test_fps = test_fps[: args.max_test]

    if args.mode == "train-test":
        if not train_fps or not test_fps:
            raise RuntimeError("Train/test split is empty after filtering.")
    elif len(train_fps) < 2:
        raise RuntimeError("Train split needs at least two items for train/train.")

    DataStructs = _require_rdkit()
    counts = np.zeros(args.bins, dtype=np.int64)
    bin_edges = np.linspace(0.0, 1.0, args.bins + 1)
    if args.mode == "train-test":
        for test_fp in test_fps:
            sims = DataStructs.BulkTanimotoSimilarity(test_fp, train_fps)
            hist, _ = np.histogram(sims, bins=bin_edges)
            counts += hist
        total_pairs = len(train_fps) * len(test_fps)
        title = "Train/Test Tanimoto Similarity"
    else:
        for idx in range(1, len(train_fps)):
            sims = DataStructs.BulkTanimotoSimilarity(train_fps[idx], train_fps[:idx])
            hist, _ = np.histogram(sims, bins=bin_edges)
            counts += hist
        total_pairs = len(train_fps) * (len(train_fps) - 1) // 2
        title = "Train/Train Tanimoto Similarity"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    widths = np.diff(bin_edges)
    ax.bar(bin_edges[:-1], counts, width=widths, align="edge", edgecolor="black")
    ax.set_title(title)
    ax.set_xlabel("Tanimoto similarity")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(args.output, dpi=200)
    plt.close(fig)

    print(
        "Generated similarity plot. "
        f"mode: {args.mode}, "
        f"train: {len(train_fps)}, "
        f"test: {len(test_fps)}, "
        f"pairs: {total_pairs}, "
        f"bins: {args.bins}, "
        f"output: {args.output}"
    )


if __name__ == "__main__":
    main()
