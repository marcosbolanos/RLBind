from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src import PROJECT_ROOT
from src.data_utils.lppdbb.records import DEFAULT_CSV_PATH

DEFAULT_SPLIT_PATH = (
    PROJECT_ROOT / "data" / "interim" / "lppdbb" / "lppdbb_train_test_split.csv"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "processed" / "plots" / "ood_figure_pack_v2"
)
DEFAULT_OVERALL_NAME = "lppdbb_binder_length_distribution.png"
DEFAULT_SPLIT_NAME = "lppdbb_binder_length_distribution_train_test.png"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot binder sequence length distributions."
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help="Path to LP_PDBBind.csv with sequences.",
    )
    parser.add_argument(
        "--split-path",
        type=Path,
        default=DEFAULT_SPLIT_PATH,
        help="Path to train/test split CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for plots.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=50,
        help="Number of histogram bins.",
    )
    return parser.parse_args()


def _sequence_length(seq: str) -> int:
    return sum(1 for ch in seq if ch.isalpha())


def _load_sequences(csv_path: Path) -> dict[str, str]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {csv_path}")
        id_column = "" if "" in reader.fieldnames else reader.fieldnames[0]
        if "seq" not in reader.fieldnames:
            raise ValueError("CSV missing 'seq' column.")
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


def _load_split_assignments(path: Path) -> dict[str, str]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Split CSV has no header: {path}")
        required = {"complex_id", "split"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Split CSV missing columns: {sorted(missing)}")
        assignments: dict[str, str] = {}
        for row in reader:
            complex_id = (row.get("complex_id") or "").strip()
            split = (row.get("split") or "").strip()
            if not complex_id or not split:
                continue
            assignments[complex_id] = split
    return assignments


def _plot_overall(lengths: np.ndarray, output_path: Path, bins: int) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(lengths, bins=bins, edgecolor="black")
    ax.set_title("Distribution of Binder Sequence Lengths")
    ax.set_xlabel("Sequence Length (Amino Acids)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_split(
    train_lengths: np.ndarray,
    test_lengths: np.ndarray,
    output_path: Path,
    bins: int,
) -> None:
    combined = np.concatenate([train_lengths, test_lengths])
    min_len = float(np.min(combined))
    max_len = float(np.max(combined))
    bin_edges = np.linspace(min_len, max_len, bins + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(
        train_lengths,
        bins=bin_edges,
        alpha=0.6,
        label="train",
        edgecolor="black",
        color="#4C78A8",
    )
    ax.hist(
        test_lengths,
        bins=bin_edges,
        alpha=0.6,
        label="test",
        edgecolor="black",
        color="#F58518",
    )
    ax.set_title("Distribution of Binder Sequence Lengths")
    ax.set_xlabel("Sequence Length (Amino Acids)")
    ax.set_ylabel("Count")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    if not args.csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv_path}")
    if not args.split_path.exists():
        raise FileNotFoundError(f"Split CSV not found: {args.split_path}")

    sequences = _load_sequences(args.csv_path)
    if not sequences:
        raise RuntimeError("No sequences found in CSV.")

    all_lengths = np.asarray(
        [_sequence_length(seq) for seq in sequences.values()], dtype=np.float32
    )
    all_lengths = all_lengths[all_lengths > 0]
    if all_lengths.size == 0:
        raise RuntimeError("No valid sequence lengths computed.")

    split_assignments = _load_split_assignments(args.split_path)
    train_lengths: list[int] = []
    test_lengths: list[int] = []
    for complex_id, seq in sequences.items():
        split = split_assignments.get(complex_id)
        if split is None:
            continue
        length = _sequence_length(seq)
        if length <= 0:
            continue
        if split == "train":
            train_lengths.append(length)
        elif split == "test":
            test_lengths.append(length)
        else:
            raise ValueError(f"Unknown split label: {split}")

    if not train_lengths or not test_lengths:
        raise RuntimeError("Train/test split is empty after filtering.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    overall_path = args.output_dir / DEFAULT_OVERALL_NAME
    split_path = args.output_dir / DEFAULT_SPLIT_NAME

    _plot_overall(all_lengths, overall_path, args.bins)
    _plot_split(
        np.asarray(train_lengths, dtype=np.float32),
        np.asarray(test_lengths, dtype=np.float32),
        split_path,
        args.bins,
    )

    print(
        "Generated binder length distributions. "
        f"total: {all_lengths.size}, "
        f"train: {len(train_lengths)}, "
        f"test: {len(test_lengths)}, "
        f"output_dir: {args.output_dir}"
    )


if __name__ == "__main__":
    main()
