from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.plots.ood_figure_pack.utils import (
    DEFAULT_INDEX_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SPLIT_PATH,
    load_index_rows,
    load_split_assignments,
    require_rdkit_chem,
    require_rdkit_descriptors,
    require_scipy_stats,
    split_smiles,
    validate_split_matches_index,
)

DEFAULT_OUTPUT_PATH = DEFAULT_OUTPUT_DIR / "lppdbb_property_shift.png"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot property distribution shift between train and test."
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
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output plot path.",
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


def _cap_list(values: list[str], limit: int | None) -> list[str]:
    if limit is None:
        return values
    return values[:limit]


def _compute_descriptors(smiles_list: list[str]) -> dict[str, np.ndarray]:
    Chem = require_rdkit_chem()
    Descriptors, Crippen, Lipinski, rdMolDescriptors = require_rdkit_descriptors()

    values: dict[str, list[float]] = {
        "MW": [],
        "cLogP": [],
        "HBD": [],
        "HBA": [],
        "TPSA": [],
        "RotB": [],
        "Rings": [],
        "FormalCharge": [],
    }

    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        values["MW"].append(Descriptors.MolWt(mol))
        values["cLogP"].append(Crippen.MolLogP(mol))
        values["HBD"].append(Lipinski.NumHDonors(mol))
        values["HBA"].append(Lipinski.NumHAcceptors(mol))
        values["TPSA"].append(rdMolDescriptors.CalcTPSA(mol))
        values["RotB"].append(Lipinski.NumRotatableBonds(mol))
        values["Rings"].append(rdMolDescriptors.CalcNumRings(mol))
        values["FormalCharge"].append(rdMolDescriptors.CalcFormalCharge(mol))

    return {key: np.asarray(vals, dtype=np.float32) for key, vals in values.items()}


def _plot_panel(
    ax: plt.Axes,
    train_vals: np.ndarray,
    test_vals: np.ndarray,
    title: str,
) -> None:
    stats = require_scipy_stats()
    if train_vals.size == 0 or test_vals.size == 0:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return

    combined = np.concatenate([train_vals, test_vals])
    x_min = float(np.min(combined))
    x_max = float(np.max(combined))
    if x_min == x_max:
        x_min -= 1.0
        x_max += 1.0
    x_grid = np.linspace(x_min, x_max, 200)

    try:
        kde_train = stats.gaussian_kde(train_vals)
        kde_test = stats.gaussian_kde(test_vals)
        train_density = kde_train(x_grid)
        test_density = kde_test(x_grid)
        ax.plot(x_grid, train_density, color="#4C78A8", label="train")
        ax.plot(x_grid, test_density, color="#F58518", label="test")
        ax.fill_between(x_grid, train_density, color="#4C78A8", alpha=0.15)
        ax.fill_between(x_grid, test_density, color="#F58518", alpha=0.15)
    except Exception:
        ax.hist(train_vals, bins=30, density=True, alpha=0.4, color="#4C78A8")
        ax.hist(test_vals, bins=30, density=True, alpha=0.4, color="#F58518")

    ax.set_title(title)


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
    train_smiles = _cap_list(train_smiles, args.max_train)
    test_smiles = _cap_list(test_smiles, args.max_test)

    if not train_smiles or not test_smiles:
        raise RuntimeError("Train/test split is empty after filtering.")

    train_desc = _compute_descriptors(train_smiles)
    test_desc = _compute_descriptors(test_smiles)

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes = axes.flatten()
    for idx, key in enumerate(train_desc.keys()):
        _plot_panel(axes[idx], train_desc[key], test_desc[key], key)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", frameon=False)
    fig.suptitle("Property distribution shift")
    fig.tight_layout(rect=(0, 0, 0.95, 0.95))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    plt.close(fig)

    print(
        "Generated property shift plot. "
        f"train: {len(train_smiles)}, "
        f"test: {len(test_smiles)}, "
        f"output: {args.output}"
    )


if __name__ == "__main__":
    main()
