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
    PROJECT_ROOT
    / "data"
    / "processed"
    / "plots"
    / "ood_figure_pack"
    / "lppdbb_nn_ecdf.png"
)


def _require_rdkit():
    try:
        from rdkit import DataStructs  # noqa: WPS433  # pyright: ignore[reportMissingImports]
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "RDKit is required for NN similarity ECDF plots. "
            "Install it with `uv sync --extra rdkit`."
        ) from exc
    return DataStructs


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot ECDF or survival of NN similarity for test ligands."
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
        "--curve",
        choices=("ecdf", "survival"),
        default="survival",
        help="Plot ECDF or survival curve.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="Similarity threshold to report in the annotation.",
    )
    return parser.parse_args()


def _compute_nn_similarities(
    train_fps: list[object],
    test_fps: list[object],
) -> np.ndarray:
    DataStructs = _require_rdkit()
    nn_sims = np.empty(len(test_fps), dtype=np.float32)
    for idx, test_fp in enumerate(test_fps):
        sims = DataStructs.BulkTanimotoSimilarity(test_fp, train_fps)
        nn_sims[idx] = float(max(sims)) if sims else 0.0
    return nn_sims


def _split_fingerprints(
    fingerprints: list[object],
    split_rows: dict[int, dict[str, str]],
) -> tuple[list[object], list[object]]:
    train_fps: list[object] = []
    test_fps: list[object] = []
    for idx, fp in enumerate(fingerprints):
        split = split_rows[idx]["split"]
        if split == "train":
            train_fps.append(fp)
        elif split == "test":
            test_fps.append(fp)
        else:
            raise ValueError(f"Unknown split label: {split}")
    return train_fps, test_fps


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
    split_rows = _load_split_assignments(args.split_path)

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
    train_fps, test_fps = _split_fingerprints(fingerprints, split_rows)
    if not train_fps or not test_fps:
        raise RuntimeError("Train/test split is empty after filtering.")

    nn_sims = _compute_nn_similarities(train_fps, test_fps)
    sorted_sims = np.sort(nn_sims)
    n = len(sorted_sims)
    ecdf = np.arange(1, n + 1, dtype=np.float32) / n
    if args.curve == "survival":
        curve = 1.0 - ecdf
        ylabel = "Pr(NN similarity ≥ t)"
    else:
        curve = ecdf
        ylabel = "Pr(NN similarity ≤ t)"

    above_thresh = float(np.mean(nn_sims >= args.threshold))
    below_thresh = 1.0 - above_thresh
    if args.curve == "survival":
        report_value = above_thresh
        report_label = "≥"
    else:
        report_value = below_thresh
        report_label = "≤"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(sorted_sims, curve, linewidth=2)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Test NN similarity to train")
    ax.set_xlabel("Nearest-neighbor Tanimoto similarity")
    ax.set_ylabel(ylabel)
    ax.annotate(
        f"{report_value * 100:.1f}% of test with NN {report_label} {args.threshold}",
        xy=(0.02, 0.96),
        xycoords="axes fraction",
        va="top",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(args.output, dpi=200)
    plt.close(fig)

    print(
        "Generated NN ECDF plot. "
        f"test: {len(test_fps)}, "
        f"curve: {args.curve}, "
        f"threshold: {args.threshold}, "
        f"output: {args.output}"
    )


if __name__ == "__main__":
    main()
