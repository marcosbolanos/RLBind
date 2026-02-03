from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.plots.ood_figure_pack.utils import (
    DEFAULT_FINGERPRINTS_PATH,
    DEFAULT_INDEX_PATH,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SPLIT_PATH,
    load_fingerprints_checked,
    load_index_rows,
    load_split_assignments,
    maybe_validate_manifest,
    require_rdkit_data_structs,
    split_fingerprints,
    validate_split_matches_index,
)

DEFAULT_OUTPUT_PATH = DEFAULT_OUTPUT_DIR / "lppdbb_test_nn_heatmap.png"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot top-k train neighbor similarities for each test ligand."
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
        "--k",
        type=int,
        default=10,
        help="Number of nearest neighbors to include.",
    )
    parser.add_argument(
        "--sort",
        choices=("ascending", "descending"),
        default="ascending",
        help="Sort rows by NN similarity.",
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
        default=2000,
        help="Optional cap on test set size for plotting.",
    )
    return parser.parse_args()


def _cap_list(values: list[object], limit: int | None) -> list[object]:
    if limit is None:
        return values
    return values[:limit]


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

    index_rows = load_index_rows(args.index_path)
    split_rows = load_split_assignments(args.split_path)
    validate_split_matches_index(index_rows, split_rows)
    maybe_validate_manifest(
        args.manifest_path,
        index_path=args.index_path,
        fingerprints_path=args.fingerprints_path,
    )

    fingerprints = load_fingerprints_checked(
        args.fingerprints_path,
        index_rows=index_rows,
    )
    train_fps, test_fps = split_fingerprints(fingerprints, split_rows)
    train_fps = _cap_list(train_fps, args.max_train)
    test_fps = _cap_list(test_fps, args.max_test)

    if not train_fps or not test_fps:
        raise RuntimeError("Train/test split is empty after filtering.")

    DataStructs = require_rdkit_data_structs()

    k = min(args.k, len(train_fps))
    if k <= 0:
        raise ValueError("k must be >= 1 and <= number of train molecules.")

    topk_matrix = np.empty((len(test_fps), k), dtype=np.float32)
    nn_scores = np.empty(len(test_fps), dtype=np.float32)
    for idx, test_fp in enumerate(test_fps):
        sims = np.asarray(
            DataStructs.BulkTanimotoSimilarity(test_fp, train_fps),
            dtype=np.float32,
        )
        if sims.size == 0:
            topk_matrix[idx] = 0.0
            nn_scores[idx] = 0.0
            continue
        if sims.size <= k:
            top = np.sort(sims)[::-1]
            if top.size < k:
                padded = np.zeros(k, dtype=np.float32)
                padded[: top.size] = top
                top = padded
        else:
            top = np.partition(sims, -k)[-k:]
            top.sort()
            top = top[::-1]
        topk_matrix[idx] = top
        nn_scores[idx] = top[0]

    if args.sort == "ascending":
        order = np.argsort(nn_scores)
    else:
        order = np.argsort(nn_scores)[::-1]
    topk_matrix = topk_matrix[order]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(
        topk_matrix,
        aspect="auto",
        interpolation="nearest",
        vmin=0.0,
        vmax=1.0,
        cmap="viridis",
    )
    ax.set_title("Test ligands: top-k train neighbor similarities")
    ax.set_xlabel("Neighbor rank")
    ax.set_ylabel("Test molecules (sorted)")
    ax.set_xticks(range(k))
    ax.set_xticklabels([str(idx + 1) for idx in range(k)])
    fig.colorbar(im, ax=ax, label="Tanimoto similarity")
    fig.tight_layout()
    fig.savefig(args.output, dpi=200)
    plt.close(fig)

    print(
        "Generated NN heatmap. "
        f"test: {len(test_fps)}, "
        f"train: {len(train_fps)}, "
        f"k: {k}, "
        f"output: {args.output}"
    )


if __name__ == "__main__":
    main()
