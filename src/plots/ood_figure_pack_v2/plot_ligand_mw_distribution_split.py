from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src import PROJECT_ROOT
from src.data_utils.lppdbb.artifacts import DEFAULT_INDEX_PATH, load_index

DEFAULT_SPLIT_PATH = (
    PROJECT_ROOT / "data" / "interim" / "lppdbb" / "lppdbb_train_test_split.csv"
)
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "plots"
    / "ood_figure_pack_v2"
    / "lppdbb_mw_distribution_train_test.png"
)


def _require_rdkit():
    try:
        from rdkit.Chem import Descriptors  # noqa: WPS433  # pyright: ignore[reportMissingImports]
        from rdkit import Chem  # noqa: WPS433, TID252  # pyright: ignore[reportMissingImports]
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "RDKit is required for molecular weight plots. "
            "Install it with `uv sync --extra rdkit`."
        ) from exc
    return Chem, Descriptors


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
        description="Plot ligand molecular weight distribution for train/test."
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default=DEFAULT_INDEX_PATH,
        help="Path to ligand index CSV.",
    )
    parser.add_argument(
        "--split-path",
        type=Path,
        default=DEFAULT_SPLIT_PATH,
        help="Path to train/test split CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output plot path.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=50,
        help="Number of histogram bins.",
    )
    return parser.parse_args()


def _compute_molecular_weights(smiles_list: list[str]) -> np.ndarray:
    Chem, Descriptors = _require_rdkit()
    weights: list[float] = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        weights.append(float(Descriptors.MolWt(mol)))
    return np.asarray(weights, dtype=np.float32)


def main() -> None:
    args = _parse_args()
    if not args.index_path.exists():
        raise FileNotFoundError(f"Index CSV not found: {args.index_path}")
    if not args.split_path.exists():
        raise FileNotFoundError(f"Split CSV not found: {args.split_path}")

    index_rows = load_index(args.index_path)
    split_rows = _load_split_assignments(args.split_path)

    if len(split_rows) != len(index_rows):
        raise ValueError(
            "Split CSV count mismatch with index. "
            f"Index rows: {len(index_rows)}, split rows: {len(split_rows)}."
        )

    train_smiles: list[str] = []
    test_smiles: list[str] = []
    for idx, row in enumerate(index_rows):
        split_row = split_rows.get(idx)
        if split_row is None:
            raise ValueError(f"Split CSV missing row_idx {idx}.")
        if split_row["complex_id"] != row["complex_id"]:
            raise ValueError(
                "Split complex_id mismatch at row "
                f"{idx}: {split_row['complex_id']} != {row['complex_id']}"
            )
        split = split_row["split"]
        smiles = row.get("smiles")
        if not smiles:
            continue
        if split == "train":
            train_smiles.append(smiles)
        elif split == "test":
            test_smiles.append(smiles)
        else:
            raise ValueError(f"Unknown split label: {split}")

    if not train_smiles or not test_smiles:
        raise RuntimeError("Train/test split is empty after filtering.")

    train_weights = _compute_molecular_weights(train_smiles)
    test_weights = _compute_molecular_weights(test_smiles)
    if train_weights.size == 0 or test_weights.size == 0:
        raise RuntimeError("No valid molecular weights computed.")

    combined = np.concatenate([train_weights, test_weights])
    min_w = float(np.min(combined))
    max_w = float(np.max(combined))
    bins = np.linspace(min_w, max_w, args.bins + 1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(
        train_weights,
        bins=bins,
        alpha=0.6,
        label="train",
        edgecolor="black",
        color="#4C78A8",
    )
    ax.hist(
        test_weights,
        bins=bins,
        alpha=0.6,
        label="test",
        edgecolor="black",
        color="#F58518",
    )
    ax.set_title("Distribution of Ligand Molecular Weights")
    ax.set_xlabel("Molecular Weight (Da)")
    ax.set_ylabel("Count")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.output, dpi=200)
    plt.close(fig)

    print(
        "Generated train/test molecular weight distribution. "
        f"train: {train_weights.size}, "
        f"test: {test_weights.size}, "
        f"bins: {args.bins}, "
        f"output: {args.output}"
    )


if __name__ == "__main__":
    main()
