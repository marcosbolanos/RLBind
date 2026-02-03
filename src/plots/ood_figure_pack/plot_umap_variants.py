from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
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
from src.data_utils.lppdbb.rdkit.clustering import butina_cluster_distances
from src.data_utils.lppdbb.rdkit.distances import (
    DEFAULT_DISTANCES_PATH,
    DEFAULT_FINGERPRINTS_PATH,
    compute_pairwise_distances,
    load_pairwise_distances,
)
from src.data_utils.lppdbb.rdkit.preprocessing import preprocess_ligands_with_report
from src.general_utils.timer import Timer
from src.plots.ood_figure_pack.utils import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SPLIT_PATH,
    load_split_assignments,
    require_umap,
    validate_split_matches_index,
)

DEFAULT_OUTPUT_NN = DEFAULT_OUTPUT_DIR / "lppdbb_umap_nn_similarity.png"
DEFAULT_OUTPUT_DENSITY = DEFAULT_OUTPUT_DIR / "lppdbb_umap_test_density.png"
DEFAULT_OUTPUT_MEDOID = DEFAULT_OUTPUT_DIR / "lppdbb_umap_cluster_medoids.png"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot UMAP variants for OOD diagnostics."
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
        default=(
            PROJECT_ROOT / "data" / "interim" / "lppdbb" / "lppdbb_preprocess_log.csv"
        ),
        help="Output CSV path for preprocessing failures.",
    )
    parser.add_argument(
        "--split-path",
        type=Path,
        default=DEFAULT_SPLIT_PATH,
        help="Path to train/test split CSV.",
    )
    parser.add_argument(
        "--distances-path",
        type=Path,
        default=DEFAULT_DISTANCES_PATH,
        help="Path to cached condensed distances (.npy).",
    )
    parser.add_argument(
        "--fingerprints-path",
        type=Path,
        default=DEFAULT_FINGERPRINTS_PATH,
        help="Path to cached fingerprints (.npy).",
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default=DEFAULT_INDEX_PATH,
        help="Path to cached ligand index CSV.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Path to cached artifact manifest JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for plots.",
    )
    parser.add_argument(
        "--variants",
        type=str,
        default="all",
        help="Comma-separated variants: nn-sim,test-density,medoid or 'all'.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Tanimoto similarity threshold for Butina clustering.",
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=2,
        help="Morgan fingerprint radius (when computing distances).",
    )
    parser.add_argument(
        "--n-bits",
        type=int,
        default=2048,
        help="Morgan fingerprint bit size (when computing distances).",
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=15,
        help="UMAP neighborhood size.",
    )
    parser.add_argument(
        "--min-dist",
        type=float,
        default=0.1,
        help="UMAP minimum distance in the embedding.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for UMAP.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=6.0,
        help="Scatter point size.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.8,
        help="Scatter point alpha.",
    )
    parser.add_argument(
        "--density-bins",
        type=int,
        default=60,
        help="Bins per axis for train density contour.",
    )
    parser.add_argument(
        "--max-clusters",
        type=int,
        default=None,
        help="Optional cap on number of clusters for medoid embedding.",
    )
    return parser.parse_args()


def _condensed_to_square(distances: np.ndarray, num_items: int) -> np.ndarray:
    square = np.zeros((num_items, num_items), dtype=distances.dtype)
    tri_i, tri_j = np.triu_indices(num_items, k=1)
    square[tri_i, tri_j] = distances
    square[tri_j, tri_i] = distances
    return square


def _condensed_index(i: int, j: int) -> int:
    if i == j:
        raise ValueError("Condensed index undefined for diagonal.")
    if i < j:
        i, j = j, i
    return i * (i - 1) // 2 + j


def _load_or_compute_distances(
    args: argparse.Namespace,
) -> tuple[list[dict[str, str]], np.ndarray]:
    record_rows: list[dict[str, str]] = []
    distances = None

    manifest = None
    if args.manifest_path.exists():
        manifest = load_manifest(args.manifest_path)

    index_rows = None
    if args.index_path.exists():
        index_rows = load_index(args.index_path)
        validate_index_rows(index_rows)

    use_cache = args.distances_path.exists() and index_rows is not None
    if use_cache:
        assert index_rows is not None
        if manifest is not None:
            try:
                validate_manifest(
                    manifest,
                    csv_path=args.csv_path,
                    num_items=len(index_rows),
                    index_path=args.index_path,
                    distances_path=args.distances_path,
                )
            except ValueError:
                use_cache = False

    if use_cache:
        assert index_rows is not None
        record_rows = index_rows
        distances = load_pairwise_distances(
            args.distances_path,
            num_items=len(record_rows),
        )
    else:
        records, _, report = preprocess_ligands_with_report(
            args.csv_path,
            report_path=args.log_output,
        )
        if not records:
            raise RuntimeError("No ligands were loaded. Check the dataset path.")

        record_rows = index_rows_from_records(records)
        if index_rows is not None:
            validate_index_matches_records(index_rows, records)
        else:
            write_index(record_rows, args.index_path)

        num_mols = len(records)
        distances = compute_pairwise_distances(
            [record.mol for record in records],
            radius=args.radius,
            n_bits=args.n_bits,
            output_path=args.distances_path,
            fingerprints_path=args.fingerprints_path,
        )

        num_pairs = num_mols * (num_mols - 1) // 2
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
            index_path=args.index_path,
            fingerprints_path=args.fingerprints_path,
            distances_path=args.distances_path,
        )
        write_manifest(args.manifest_path, manifest)

    if distances is None:
        raise RuntimeError("Distances were not computed or loaded.")
    return record_rows, distances


def _parse_variants(text: str) -> set[str]:
    if text.strip().lower() == "all":
        return {"nn-sim", "test-density", "medoid"}
    return {item.strip() for item in text.split(",") if item.strip()}


def _plot_nn_similarity(
    embedding: np.ndarray,
    nn_similarity: np.ndarray,
    output_path: Path,
    *,
    point_size: float,
    alpha: float,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=nn_similarity,
        s=point_size,
        alpha=alpha,
        cmap="viridis",
        linewidths=0,
        rasterized=True,
    )
    ax.set_title("UMAP colored by NN similarity to train")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(scatter, ax=ax, label="NN similarity")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_test_density(
    embedding: np.ndarray,
    train_mask: np.ndarray,
    output_path: Path,
    *,
    point_size: float,
    alpha: float,
    bins: int,
) -> None:
    train_points = embedding[train_mask]
    test_points = embedding[~train_mask]

    fig, ax = plt.subplots(figsize=(10, 8))
    hist, xedges, yedges = np.histogram2d(
        train_points[:, 0],
        train_points[:, 1],
        bins=bins,
        density=True,
    )
    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2
    xx, yy = np.meshgrid(xcenters, ycenters)
    ax.contour(xx, yy, hist.T, levels=6, cmap="Blues")
    ax.scatter(
        test_points[:, 0],
        test_points[:, 1],
        s=point_size,
        alpha=alpha,
        color="#F58518",
        linewidths=0,
        rasterized=True,
        label="test",
    )
    ax.set_title("UMAP: train density + test points")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _cluster_medoids(
    clusters: tuple[tuple[int, ...], ...],
    distances: np.ndarray,
) -> list[int]:
    medoids: list[int] = []
    for cluster in clusters:
        if len(cluster) == 1:
            medoids.append(cluster[0])
            continue
        indices = list(cluster)
        sums = np.zeros(len(indices), dtype=np.float64)
        for i in range(1, len(indices)):
            for j in range(i):
                dist = distances[_condensed_index(indices[i], indices[j])]
                sums[i] += dist
                sums[j] += dist
        medoids.append(indices[int(np.argmin(sums))])
    return medoids


def _plot_medoids(
    distances: np.ndarray,
    clusters: tuple[tuple[int, ...], ...],
    split_labels: np.ndarray,
    output_path: Path,
    *,
    threshold: float,
    max_clusters: int | None,
    n_neighbors: int,
    min_dist: float,
    random_state: int,
) -> None:
    sorted_clusters = sorted(clusters, key=len, reverse=True)
    if max_clusters is not None:
        sorted_clusters = sorted_clusters[:max_clusters]

    medoids = _cluster_medoids(tuple(sorted_clusters), distances)
    cluster_sizes = [len(cluster) for cluster in sorted_clusters]

    if len(medoids) < 2:
        raise RuntimeError("Need at least two medoids for UMAP.")

    num_medoids = len(medoids)
    medoid_square = np.zeros((num_medoids, num_medoids), dtype=np.float32)
    for i in range(1, num_medoids):
        for j in range(i):
            dist = distances[_condensed_index(medoids[i], medoids[j])]
            medoid_square[i, j] = dist
            medoid_square[j, i] = dist

    umap_module = require_umap()
    reducer = umap_module.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="precomputed",
        random_state=random_state,
    )
    embedding: np.ndarray = np.asarray(reducer.fit_transform(medoid_square))

    labels = []
    for cluster in sorted_clusters:
        train_count = sum(1 for idx in cluster if split_labels[idx] == "train")
        test_count = sum(1 for idx in cluster if split_labels[idx] == "test")
        if train_count and test_count:
            labels.append("mixed")
        elif train_count:
            labels.append("train")
        else:
            labels.append("test")

    color_map = {
        "train": "#4C78A8",
        "test": "#F58518",
        "mixed": "#54A24B",
    }
    sizes = np.array(cluster_sizes, dtype=np.float32)
    sizes = 30 + 120 * (sizes / sizes.max())

    fig, ax = plt.subplots(figsize=(10, 8))
    for label in ("train", "test", "mixed"):
        mask = np.array([lab == label for lab in labels])
        if not np.any(mask):
            continue
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=sizes[mask],
            alpha=0.8,
            color=color_map[label],
            label=label,
            linewidths=0,
            rasterized=True,
        )
    ax.set_title(f"Cluster medoid UMAP (threshold={threshold})")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    variants = _parse_variants(args.variants)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.split_path.exists():
        raise FileNotFoundError(f"Split CSV not found: {args.split_path}")

    total_timer = Timer()
    record_rows, distances = _load_or_compute_distances(args)
    num_mols = len(record_rows)

    split_rows = load_split_assignments(args.split_path)
    validate_split_matches_index(record_rows, split_rows)
    split_labels = np.array(
        [split_rows[idx]["split"] for idx in range(len(record_rows))]
    )
    train_mask = split_labels == "train"
    if not np.any(train_mask):
        raise RuntimeError("No train molecules found in split.")

    distances = distances.astype(np.float32, copy=False)
    square = _condensed_to_square(distances, num_mols)
    umap_module = require_umap()
    reducer = umap_module.UMAP(
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric="precomputed",
        random_state=args.random_state,
    )
    embedding: np.ndarray = np.asarray(reducer.fit_transform(square))

    if "nn-sim" in variants:
        train_indices = np.where(train_mask)[0]
        train_distances = square[:, train_indices]
        if train_indices.size < 2:
            raise RuntimeError("Need at least two train molecules for NN similarity.")
        train_distances[train_indices, np.arange(train_indices.size)] = np.inf
        min_dist = np.min(train_distances, axis=1)
        nn_similarity = 1.0 - min_dist
        _plot_nn_similarity(
            embedding,
            nn_similarity,
            args.output_dir / DEFAULT_OUTPUT_NN.name,
            point_size=args.point_size,
            alpha=args.alpha,
        )

    if "test-density" in variants:
        _plot_test_density(
            embedding,
            train_mask,
            args.output_dir / DEFAULT_OUTPUT_DENSITY.name,
            point_size=args.point_size,
            alpha=args.alpha,
            bins=args.density_bins,
        )

    if "medoid" in variants:
        clusters = butina_cluster_distances(
            distances,
            num_mols,
            threshold=args.threshold,
        )
        _plot_medoids(
            distances,
            clusters,
            split_labels,
            args.output_dir / DEFAULT_OUTPUT_MEDOID.name,
            threshold=args.threshold,
            max_clusters=args.max_clusters,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            random_state=args.random_state,
        )

    total_time = total_timer.elapsed()
    print(
        "Generated UMAP variants. "
        f"variants: {sorted(variants)}, "
        f"ligands: {num_mols}, "
        f"output_dir: {args.output_dir}, "
        f"time: {total_time:.2f}s"
    )


if __name__ == "__main__":
    main()
