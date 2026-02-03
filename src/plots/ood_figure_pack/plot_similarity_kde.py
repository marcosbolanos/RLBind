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
    require_scipy_stats,
    split_fingerprints,
    validate_split_matches_index,
)

DEFAULT_OUTPUT_PATH = DEFAULT_OUTPUT_DIR / "lppdbb_similarity_kde.png"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot KDEs for train/train vs train/test similarity distributions."
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
        "--grid-size",
        type=int,
        default=400,
        help="Number of evaluation points for KDE.",
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


def _cap_list(values: list[object], limit: int | None) -> list[object]:
    if limit is None:
        return values
    return values[:limit]


def _pairwise_similarities(
    DataStructs,
    lhs: list[object],
    rhs: list[object],
) -> np.ndarray:
    sims: list[float] = []
    for fp in lhs:
        sims.extend(DataStructs.BulkTanimotoSimilarity(fp, rhs))
    return np.asarray(sims, dtype=np.float32)


def _train_train_similarities(DataStructs, train_fps: list[object]) -> np.ndarray:
    sims: list[float] = []
    for idx in range(1, len(train_fps)):
        sims.extend(DataStructs.BulkTanimotoSimilarity(train_fps[idx], train_fps[:idx]))
    return np.asarray(sims, dtype=np.float32)


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

    if len(train_fps) < 2 or not test_fps:
        raise RuntimeError("Train/test split is empty after filtering.")

    DataStructs = require_rdkit_data_structs()
    stats = require_scipy_stats()

    train_train = _train_train_similarities(DataStructs, train_fps)
    train_test = _pairwise_similarities(DataStructs, test_fps, train_fps)

    delta_mu = float(np.mean(train_train) - np.mean(train_test))
    ks_stat = float(stats.ks_2samp(train_train, train_test).statistic)

    x_grid = np.linspace(0.0, 1.0, args.grid_size)
    kde_tt = stats.gaussian_kde(train_train)
    kde_te = stats.gaussian_kde(train_test)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x_grid, kde_tt(x_grid), label="train-train", linewidth=2)
    ax.plot(x_grid, kde_te(x_grid), label="train-test", linewidth=2)
    ax.fill_between(x_grid, kde_tt(x_grid), alpha=0.15)
    ax.fill_between(x_grid, kde_te(x_grid), alpha=0.15)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Tanimoto similarity")
    ax.set_ylabel("Density")
    ax.set_title("Train/Train vs Train/Test similarity")
    ax.legend(frameon=False)
    ax.annotate(
        f"Δμ = {delta_mu:.3f}\nKS = {ks_stat:.3f}",
        xy=(0.02, 0.96),
        xycoords="axes fraction",
        va="top",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(args.output, dpi=200)
    plt.close(fig)

    print(
        "Generated similarity KDE plot. "
        f"train: {len(train_fps)}, "
        f"test: {len(test_fps)}, "
        f"delta_mu: {delta_mu:.3f}, "
        f"ks: {ks_stat:.3f}, "
        f"output: {args.output}"
    )


if __name__ == "__main__":
    main()
