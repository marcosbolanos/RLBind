from __future__ import annotations

import argparse
import colorsys
import csv
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

DEFAULT_SPLIT_PATH = (
    PROJECT_ROOT / "data" / "interim" / "lppdbb" / "lppdbb_train_test_split.csv"
)


def _require_umap():
    try:
        import umap  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "UMAP is required for LP-PDBBind UMAP plotting. "
            "Install it with `uv add umap-learn`."
        ) from exc
    return umap


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a UMAP embedding colored by train/test split."
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
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "processed"
        / "plots"
        / "lppdbb_umap_train_test.png",
        help="Output plot path.",
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
        default=5,
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
        default=8.0,
        help="Scatter point size.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.85,
        help="Scatter point alpha.",
    )
    return parser.parse_args()


def _condensed_to_square(distances: np.ndarray, num_items: int) -> np.ndarray:
    square = np.zeros((num_items, num_items), dtype=distances.dtype)
    tri_i, tri_j = np.triu_indices(num_items, k=1)
    square[tri_i, tri_j] = distances
    square[tri_j, tri_i] = distances
    return square


def _cluster_colors(
    num_colors: int,
    *,
    saturation: float = 0.62,
    lightness: float = 0.52,
    hue_offset: float = 0.07,
) -> list[tuple[float, float, float]]:
    if num_colors <= 0:
        return []
    golden_ratio = 0.61803398875
    return [
        colorsys.hls_to_rgb(
            (hue_offset + idx * golden_ratio) % 1.0,
            lightness,
            saturation,
        )
        for idx in range(num_colors)
    ]


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


def _split_color_map(labels: list[str]) -> dict[str, tuple[float, float, float]]:
    colors: dict[str, tuple[float, float, float]] = {
        "train": (0.121, 0.466, 0.705),
        "test": (1.0, 0.498, 0.055),
    }
    missing = [label for label in labels if label not in colors]
    if missing:
        palette = _cluster_colors(len(missing))
        for label, color in zip(missing, palette, strict=False):
            colors[label] = color
    return colors


def _ordered_labels(labels: list[str]) -> list[str]:
    unique = list(dict.fromkeys(labels))
    preferred = [label for label in ("train", "test") if label in unique]
    return preferred + [label for label in unique if label not in preferred]


def _plot_embedding(
    embedding: np.ndarray,
    split_labels: list[str],
    output_path: Path,
    *,
    point_size: float,
    alpha: float,
    title: str,
) -> None:
    labels = np.asarray(split_labels)
    color_map = _split_color_map(_ordered_labels(split_labels))
    fig, ax = plt.subplots(figsize=(10, 8))
    for label in _ordered_labels(split_labels):
        mask = labels == label
        if not np.any(mask):
            continue
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=point_size,
            alpha=alpha,
            color=color_map[label],
            label=label,
            linewidths=0,
            rasterized=True,
        )
    ax.set_title(title)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(frameon=False, markerscale=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    if not args.split_path.exists():
        raise FileNotFoundError(f"Split CSV not found: {args.split_path}")

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

    split_rows = _load_split_assignments(args.split_path)
    if len(split_rows) != len(record_rows):
        raise ValueError(
            "Split CSV count mismatch with index. "
            f"Index rows: {len(record_rows)}, split rows: {len(split_rows)}."
        )
    split_labels: list[str] = []
    for idx, index_row in enumerate(record_rows):
        split_row = split_rows.get(idx)
        if split_row is None:
            raise ValueError(f"Split CSV missing row_idx {idx}.")
        if split_row["complex_id"] != index_row["complex_id"]:
            raise ValueError(
                "Split complex_id mismatch at row "
                f"{idx}: {split_row['complex_id']} != {index_row['complex_id']}"
            )
        split_labels.append(split_row["split"])

    num_pairs = num_mols * (num_mols - 1) // 2
    est_list_gb = num_pairs * 32 / 1e9
    est_array_gb = num_pairs * 8 / 1e9
    square_gb = num_mols * num_mols * 4 / 1e9
    print(
        "Pairwise estimate. "
        f"mols: {num_mols}, "
        f"pairs: {num_pairs}, "
        f"list_mem~: {est_list_gb:.2f}GB, "
        f"array_mem~: {est_array_gb:.2f}GB, "
        f"square_mem~: {square_gb:.2f}GB"
    )

    stage_timer.reset()
    if distances is None:
        raise RuntimeError("Distances were not computed or loaded.")
    distances = distances.astype(np.float32, copy=False)
    square = _condensed_to_square(distances, num_mols)
    umap_module = _require_umap()
    reducer = umap_module.UMAP(
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric="precomputed",
        random_state=args.random_state,
    )
    embedding = reducer.fit_transform(square)
    umap_time = stage_timer.elapsed()

    stage_timer.reset()
    labels = _ordered_labels(split_labels)
    counts = {label: split_labels.count(label) for label in labels}
    label_summary = ", ".join(f"{label}: {counts[label]}" for label in labels)
    title = (
        "LP-PDBBind ligand UMAP (train/test split) "
        f"[n_neighbors: {args.n_neighbors}]\n{label_summary}"
    )
    _plot_embedding(
        embedding,
        split_labels,
        args.output,
        point_size=args.point_size,
        alpha=args.alpha,
        title=title,
    )
    plot_time = stage_timer.elapsed()

    total_time = total_timer.elapsed()

    print(f"Generated UMAP split plot. ligands: {num_mols}, output: {args.output}")
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
        f"umap: {umap_time:.2f}s, "
        f"plot: {plot_time:.2f}s, "
        f"total: {total_time:.2f}s"
    )


if __name__ == "__main__":
    main()
