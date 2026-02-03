from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

from src.plots.ood_figure_pack.utils import (
    DEFAULT_INDEX_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SPLIT_PATH,
    load_index_rows,
    load_split_assignments,
    require_rdkit_chem,
    require_rdkit_scaffolds,
    split_smiles,
    validate_split_matches_index,
)

OVERLAP_OUTPUT = "lppdbb_scaffold_overlap.png"
RANK_OUTPUT = "lppdbb_scaffold_rank.png"
TOP_OUTPUT = "lppdbb_scaffold_top.png"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot scaffold shift diagnostics.")
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
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for output plots.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of top scaffolds to show in the bar plots.",
    )
    return parser.parse_args()


def _scaffold_smiles(smiles: str, Chem, MurckoScaffold) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "UNKNOWN"
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    return scaffold or "UNKNOWN"


def _short_label(text: str, max_len: int = 28) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _plot_overlap(
    train_scaffolds: list[str],
    test_scaffolds: list[str],
    output_path: Path,
) -> None:
    train_set = set(train_scaffolds)
    test_set = set(test_scaffolds)
    if not test_set:
        raise RuntimeError("No test scaffolds found.")
    unseen = test_set - train_set
    frac_scaffolds = len(unseen) / len(test_set)
    frac_molecules = sum(1 for scaf in test_scaffolds if scaf in unseen) / len(
        test_scaffolds
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    values = [frac_scaffolds, frac_molecules]
    labels = ["Test scaffolds unseen", "Test molecules on unseen scaffolds"]
    bars = ax.bar(labels, values, color=["#4C78A8", "#F58518"])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Fraction")
    ax.set_title("Scaffold overlap")
    ax.bar_label(bars, labels=[f"{v * 100:.1f}%" for v in values], padding=3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_rank(
    train_counts: Counter[str],
    test_counts: Counter[str],
    output_path: Path,
) -> None:
    ranked = train_counts.most_common()
    if not ranked:
        raise RuntimeError("No train scaffolds found.")
    ranks = list(range(1, len(ranked) + 1))
    train_vals = [count for _, count in ranked]
    test_vals = [test_counts.get(scaf, 0) for scaf, _ in ranked]
    test_mask = [val > 0 for val in test_vals]
    test_ranks = [rank for rank, keep in zip(ranks, test_mask) if keep]
    test_plot = [val for val in test_vals if val > 0]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(ranks, train_vals, label="train", linewidth=2)
    if test_plot:
        ax.loglog(test_ranks, test_plot, label="test", linewidth=2)
    ax.set_xlabel("Scaffold rank (train)")
    ax.set_ylabel("Count")
    ax.set_title("Scaffold frequency rank plot")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_top_scaffolds(
    train_counts: Counter[str],
    test_counts: Counter[str],
    output_path: Path,
    *,
    top_n: int,
) -> None:
    train_top = train_counts.most_common(top_n)
    test_top = test_counts.most_common(top_n)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    train_labels = [_short_label(scaf) for scaf, _ in train_top]
    train_vals = [count for _, count in train_top]
    axes[0].barh(train_labels, train_vals, color="#4C78A8")
    axes[0].invert_yaxis()
    axes[0].set_title("Top train scaffolds")
    axes[0].set_xlabel("Count")

    test_labels = [_short_label(scaf) for scaf, _ in test_top]
    test_vals = [count for _, count in test_top]
    axes[1].barh(test_labels, test_vals, color="#F58518")
    axes[1].invert_yaxis()
    axes[1].set_title("Top test scaffolds")
    axes[1].set_xlabel("Count")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    if not args.split_path.exists():
        raise FileNotFoundError(f"Split CSV not found: {args.split_path}")
    if not args.index_path.exists():
        raise FileNotFoundError(f"Index CSV not found: {args.index_path}")

    index_rows = load_index_rows(args.index_path)
    split_rows = load_split_assignments(args.split_path)
    validate_split_matches_index(index_rows, split_rows)

    train_smiles, test_smiles = split_smiles(index_rows, split_rows)
    if not train_smiles or not test_smiles:
        raise RuntimeError("Train/test split is empty after filtering.")

    Chem = require_rdkit_chem()
    MurckoScaffold = require_rdkit_scaffolds()
    train_scaffolds = [
        _scaffold_smiles(smiles, Chem, MurckoScaffold) for smiles in train_smiles
    ]
    test_scaffolds = [
        _scaffold_smiles(smiles, Chem, MurckoScaffold) for smiles in test_smiles
    ]
    train_counts = Counter(train_scaffolds)
    test_counts = Counter(test_scaffolds)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    overlap_path = args.output_dir / OVERLAP_OUTPUT
    rank_path = args.output_dir / RANK_OUTPUT
    top_path = args.output_dir / TOP_OUTPUT

    _plot_overlap(train_scaffolds, test_scaffolds, overlap_path)
    _plot_rank(train_counts, test_counts, rank_path)
    _plot_top_scaffolds(train_counts, test_counts, top_path, top_n=args.top_n)

    print(
        "Generated scaffold shift plots. "
        f"train_scaffolds: {len(train_counts)}, "
        f"test_scaffolds: {len(test_counts)}, "
        f"output_dir: {args.output_dir}"
    )


if __name__ == "__main__":
    main()
