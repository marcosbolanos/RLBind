from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.data_utils.lppdbb.records import DEFAULT_CSV_PATH

DEFAULT_OUTPUT_PATH = (
    Path("data")
    / "processed"
    / "plots"
    / "ood_figure_pack_v2"
    / "lppdbb_ligands_per_target.png"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot distribution of ligand counts per protein target."
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help="Path to LP_PDBBind.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output plot path.",
    )
    parser.add_argument(
        "--target-field",
        type=str,
        default="seq",
        help="CSV column to use as protein target identifier.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=50,
        help="Number of histogram bins.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv_path}")

    counts: Counter[str] = Counter()
    with args.csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {args.csv_path}")
        if args.target_field not in reader.fieldnames:
            raise ValueError(
                f"CSV missing target field '{args.target_field}'. "
                f"Available columns: {reader.fieldnames}"
            )
        for row in reader:
            target = (row.get(args.target_field) or "").strip()
            if not target:
                continue
            counts[target] += 1

    if not counts:
        raise RuntimeError("No targets found after filtering.")

    values = np.asarray(list(counts.values()), dtype=np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(values, bins=args.bins, edgecolor="black")
    ax.set_title("Distribution of Ligand Counts per Protein Target")
    ax.set_xlabel("Number of Ligands per Protein")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(args.output, dpi=200)
    plt.close(fig)

    print(
        "Generated ligands per target distribution. "
        f"targets: {len(values)}, "
        f"bins: {args.bins}, "
        f"output: {args.output}"
    )


if __name__ == "__main__":
    main()
