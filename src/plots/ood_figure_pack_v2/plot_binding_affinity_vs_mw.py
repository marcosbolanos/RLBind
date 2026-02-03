from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.data_utils.lppdbb.records import DEFAULT_CSV_PATH

DEFAULT_OUTPUT_PATH = (
    Path("data")
    / "processed"
    / "plots"
    / "ood_figure_pack_v2"
    / "lppdbb_binding_free_energy_vs_mw.png"
)

R_GAS_CONSTANT = 1.987204258e-3


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
        description="Plot binding free energy vs ligand molecular weight."
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
        "--temperature",
        type=float,
        default=298.15,
        help="Temperature in Kelvin for ΔG conversion.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.6,
        help="Scatter point alpha.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=10.0,
        help="Scatter point size.",
    )
    return parser.parse_args()


def _delta_g_from_pvalue(p_value: float, temperature: float) -> float:
    return -R_GAS_CONSTANT * temperature * np.log(10.0) * p_value


def main() -> None:
    args = _parse_args()
    if not args.csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv_path}")

    Chem, Descriptors = _require_rdkit()
    weights: list[float] = []
    delta_g: list[float] = []

    with args.csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {args.csv_path}")
        if "smiles" not in reader.fieldnames:
            raise ValueError("CSV missing 'smiles' column.")
        if "value" not in reader.fieldnames:
            raise ValueError("CSV missing 'value' column for binding affinity.")

        for row in reader:
            smiles = (row.get("smiles") or "").strip()
            value = (row.get("value") or "").strip()
            if not smiles or not value:
                continue
            try:
                p_value = float(value)
            except ValueError:
                continue
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            weights.append(float(Descriptors.MolWt(mol)))
            delta_g.append(_delta_g_from_pvalue(p_value, args.temperature))

    if not weights:
        raise RuntimeError("No valid ligand weights computed.")
    if not delta_g:
        raise RuntimeError("No valid binding affinity values computed.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(weights, delta_g, s=args.point_size, alpha=args.alpha, linewidths=0)
    ax.set_title("Binding Free Energy vs Ligand Molecular Weight")
    ax.set_xlabel("Molecular Weight (Da)")
    ax.set_ylabel("Binding Free Energy ΔG (kcal/mol)")
    fig.tight_layout()
    fig.savefig(args.output, dpi=200)
    plt.close(fig)

    print(
        "Generated binding free energy vs MW scatter. "
        f"points: {len(weights)}, "
        f"output: {args.output}"
    )


if __name__ == "__main__":
    main()
