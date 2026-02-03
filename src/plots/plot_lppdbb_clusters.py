from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

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
from src.data_utils.lppdbb.rdkit.clustering import (
    butina_cluster_distances,
    cluster_membership,
)
from src.data_utils.lppdbb.rdkit.distances import (
    DEFAULT_DISTANCES_PATH,
    DEFAULT_FINGERPRINTS_PATH,
    compute_pairwise_distances,
    load_pairwise_distances,
)
from src.data_utils.lppdbb.rdkit.preprocessing import (
    preprocess_ligands_with_report,
)
from src.general_utils.timer import Timer


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot LP-PDBBind Butina clusters for ligand similarity."
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help="Path to LP_PDBBind.csv for SMILES-based clustering.",
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
        "--threshold",
        type=float,
        default=0.5,
        help="Tanimoto similarity threshold for Butina clustering.",
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
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "plots" / "lppdbb_clusters.png",
        help="Output plot path.",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=(
            PROJECT_ROOT / "data" / "interim" / "lppdbb" / "lppdbb_ligand_clusters.csv"
        ),
        help="Output CSV path for ligand cluster assignments.",
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
    return parser.parse_args()


def plot_cluster_sizes(cluster_sizes: list[int], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(
        range(1, len(cluster_sizes) + 1),
        cluster_sizes,
        marker="o",
        markersize=2,
        linewidth=1,
    )
    axes[0].set_title("Cluster sizes (sorted)")
    axes[0].set_xlabel("Cluster rank")
    axes[0].set_ylabel("Ligands per cluster")

    axes[1].hist(cluster_sizes, bins=30)
    axes[1].set_title("Cluster size distribution")
    axes[1].set_xlabel("Cluster size")
    axes[1].set_ylabel("Count")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    total_timer = Timer()
    stage_timer = Timer()
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
        preprocess_time = stage_timer.elapsed()
        num_mols = len(record_rows)
        skipped = []
    else:
        records, skipped, report = preprocess_ligands_with_report(
            args.csv_path,
            report_path=args.log_output,
        )
        preprocess_time = stage_timer.elapsed()
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

    num_pairs = num_mols * (num_mols - 1) // 2
    est_list_gb = num_pairs * 32 / 1e9
    est_array_gb = num_pairs * 8 / 1e9
    print(
        "Pairwise estimate. "
        f"mols: {num_mols}, "
        f"pairs: {num_pairs}, "
        f"list_mem~: {est_list_gb:.2f}GB, "
        f"array_mem~: {est_array_gb:.2f}GB"
    )

    stage_timer.reset()
    if distances is None:
        raise RuntimeError("Distances were not computed or loaded.")
    clusters = butina_cluster_distances(
        distances,
        num_mols,
        threshold=args.threshold,
    )
    clustering_time = stage_timer.elapsed()

    stage_timer.reset()
    membership = cluster_membership(
        clusters,
        [row["complex_id"] for row in record_rows],
    )
    cluster_sizes = {idx: len(cluster) for idx, cluster in enumerate(clusters)}
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "complex_id",
                "ligand_path",
                "smiles",
                "cluster_id",
                "cluster_size",
            ],
        )
        writer.writeheader()
        for row in record_rows:
            cluster_id = membership[row["complex_id"]]
            writer.writerow(
                {
                    "complex_id": row["complex_id"],
                    "ligand_path": row["ligand_path"],
                    "smiles": row["smiles"],
                    "cluster_id": cluster_id,
                    "cluster_size": cluster_sizes[cluster_id],
                }
            )
    csv_write_time = stage_timer.elapsed()
    if manifest is not None:
        manifest["clusters"] = {
            "path": str(args.csv_output),
            "threshold": args.threshold,
        }
        write_manifest(args.manifest_path, manifest)

    stage_timer.reset()
    cluster_sizes_sorted = sorted(
        (len(cluster) for cluster in clusters),
        reverse=True,
    )
    plot_cluster_sizes(cluster_sizes_sorted, args.output)
    plot_time = stage_timer.elapsed()

    total_time = total_timer.elapsed()

    print(
        "Generated cluster plot. "
        f"Ligands: {num_mols}, "
        f"clusters: {len(clusters)}, "
        f"skipped: {len(skipped)}, "
        f"output: {args.output}, "
        f"csv: {args.csv_output}"
    )
    print(
        "Preprocess summary. "
        f"total: {num_mols + len(skipped)}, "
        f"retained: {num_mols}, "
        f"skipped: {len(skipped)}, "
        f"log: {args.log_output}"
    )
    print(
        "Timing. "
        f"preprocess: {preprocess_time:.2f}s, "
        f"clustering: {clustering_time:.2f}s, "
        f"csv_write: {csv_write_time:.2f}s, "
        f"plot: {plot_time:.2f}s, "
        f"total: {total_time:.2f}s"
    )


if __name__ == "__main__":
    main()
