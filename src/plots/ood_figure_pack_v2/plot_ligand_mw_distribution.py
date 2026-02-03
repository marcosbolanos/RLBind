from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src import PROJECT_ROOT
from src.data_utils.lppdbb.artifacts import DEFAULT_INDEX_PATH, load_index

DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "plots"
    / "ood_figure_pack_v2"
    / "lppdbb_mw_distribution.png"
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot ligand molecular weight distribution."
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

    index_rows = load_index(args.index_path)
    smiles_list = [row["smiles"] for row in index_rows if row.get("smiles")]
    if not smiles_list:
        raise RuntimeError("No SMILES found in index.")

    weights = _compute_molecular_weights(smiles_list)
    if weights.size == 0:
        raise RuntimeError("No valid molecular weights computed.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(weights, bins=args.bins, edgecolor="black")
    ax.set_title("Distribution of Ligand Molecular Weights")
    ax.set_xlabel("Molecular Weight (Da)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(args.output, dpi=200)
    plt.close(fig)

    print(
        "Generated molecular weight distribution. "
        f"ligands: {weights.size}, "
        f"bins: {args.bins}, "
        f"output: {args.output}"
    )


if __name__ == "__main__":
    main()
