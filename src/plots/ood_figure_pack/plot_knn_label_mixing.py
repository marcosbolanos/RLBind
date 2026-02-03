from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.data_utils.lppdbb.artifacts import (
    DEFAULT_INDEX_PATH,
    DEFAULT_MANIFEST_PATH,
    load_index,
    load_manifest,
    validate_index_rows,
    validate_manifest,
)
from src.data_utils.lppdbb.rdkit.distances import (
    DEFAULT_DISTANCES_PATH,
    load_pairwise_distances,
)
from src.plots.ood_figure_pack.utils import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SPLIT_PATH,
    load_split_assignments,
    validate_split_matches_index,
)

DEFAULT_OUTPUT_PATH = DEFAULT_OUTPUT_DIR / "lppdbb_knn_label_mixing.png"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot kNN label mixing curve for train/test split."
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
        "--distances-path",
        type=Path,
        default=DEFAULT_DISTANCES_PATH,
        help="Path to cached condensed distances (.npy).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output plot path.",
    )
    parser.add_argument(
        "--k-max",
        type=int,
        default=50,
        help="Maximum k to evaluate.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Optional cap on number of molecules for performance.",
    )
    return parser.parse_args()


def _condensed_index(i: int, j: int) -> int:
    if i == j:
        raise ValueError("Condensed index undefined for diagonal.")
    if i < j:
        i, j = j, i
    return i * (i - 1) // 2 + j


def _subset_square(distances: np.ndarray, indices: np.ndarray) -> np.ndarray:
    size = len(indices)
    square = np.zeros((size, size), dtype=distances.dtype)
    for i in range(1, size):
        orig_i = int(indices[i])
        for j in range(i):
            orig_j = int(indices[j])
            dist = distances[_condensed_index(orig_i, orig_j)]
            square[i, j] = dist
            square[j, i] = dist
    return square


def main() -> None:
    args = _parse_args()
    if not args.split_path.exists():
        raise FileNotFoundError(f"Split CSV not found: {args.split_path}")
    if not args.index_path.exists():
        raise FileNotFoundError(f"Index CSV not found: {args.index_path}")
    if not args.distances_path.exists():
        raise FileNotFoundError(
            "Distances .npy not found. Run the distances module to generate it."
        )

    index_rows = load_index(args.index_path)
    validate_index_rows(index_rows)
    split_rows = load_split_assignments(args.split_path)
    validate_split_matches_index(index_rows, split_rows)

    if args.manifest_path.exists():
        manifest = load_manifest(args.manifest_path)
        validate_manifest(
            manifest,
            num_items=len(index_rows),
            index_path=args.index_path,
            distances_path=args.distances_path,
        )

    distances = load_pairwise_distances(
        args.distances_path,
        num_items=len(index_rows),
    )

    num_items = len(index_rows)
    if args.max_items is not None and args.max_items < num_items:
        subset_indices = np.arange(args.max_items)
    else:
        subset_indices = np.arange(num_items)

    square = _subset_square(distances, subset_indices)
    labels = np.array(
        [split_rows[int(idx)]["split"] for idx in subset_indices], dtype=object
    )
    label_binary = np.where(labels == "train", 1, 0)

    k_max = min(args.k_max, len(subset_indices) - 1)
    if k_max <= 0:
        raise RuntimeError("Need at least two molecules for kNN mixing.")

    neighbors = np.empty((len(subset_indices), k_max), dtype=np.int64)
    for idx in range(len(subset_indices)):
        row = square[idx]
        candidates = np.argpartition(row, k_max + 1)[: k_max + 1]
        candidates = candidates[candidates != idx]
        order = candidates[np.argsort(row[candidates])]
        neighbors[idx] = order[:k_max]

    neighbor_labels = label_binary[neighbors]
    mismatch = neighbor_labels != label_binary[:, None]
    cumulative = np.cumsum(mismatch, axis=1, dtype=np.int64)
    mix_k = [float(np.mean(cumulative[:, k - 1] / k)) for k in range(1, k_max + 1)]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, k_max + 1), mix_k, marker="o")
    ax.set_xlabel("k")
    ax.set_ylabel("Pr(neighbor label != self label)")
    ax.set_title("kNN label mixing curve")
    ax.set_ylim(0.0, 1.0)
    fig.tight_layout()
    fig.savefig(args.output, dpi=200)
    plt.close(fig)

    print(
        "Generated kNN label mixing plot. "
        f"items: {len(subset_indices)}, "
        f"k_max: {k_max}, "
        f"output: {args.output}"
    )


if __name__ == "__main__":
    main()
