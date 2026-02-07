from __future__ import annotations

import argparse
import csv
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

from src import PROJECT_ROOT

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "processed" / "generator" / "embedding_baseline1"
)
DEFAULT_INPUT_CSV = DEFAULT_OUTPUT_DIR / "nearest_train_binders.csv"
DEFAULT_YAML_DIR = DEFAULT_OUTPUT_DIR / "boltz_inputs"

SAFE_NAME_PATTERN = re.compile(r"[^A-Za-z0-9._-]")


def _safe_name(value: str, fallback: str, *, max_len: int = 80) -> str:
    cleaned = SAFE_NAME_PATTERN.sub("_", value).strip("._-")
    if max_len > 0:
        cleaned = cleaned[:max_len]
    return cleaned or fallback


def _clean_protein_sequence(sequence: str) -> str:
    return "".join(ch for ch in sequence if ch.isalpha()).upper()


def _split_protein_sequence(sequence: str) -> list[str]:
    parts = [part.strip() for part in sequence.split(":")]
    cleaned = [_clean_protein_sequence(part) for part in parts]
    return [part for part in cleaned if part]


def _chain_ids(num_chains: int) -> list[str]:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    ids: list[str] = []
    for idx in range(num_chains):
        letter = alphabet[idx % len(alphabet)]
        suffix = idx // len(alphabet)
        if suffix == 0:
            ids.append(letter)
        else:
            ids.append(f"{letter}{suffix}")
    return ids


def _quote_smiles(smiles: str) -> str:
    return smiles.replace("'", "''")


def _format_row_idx(row_idx: str | int | None) -> str | None:
    if row_idx is None:
        return None
    try:
        return f"{int(row_idx):06d}"
    except (TypeError, ValueError):
        return str(row_idx).strip() or None


def build_boltz_yaml(
    protein_sequences: Sequence[str],
    *,
    ligand_smiles: str,
    ligand_id: str = "L",
    include_affinity: bool = True,
) -> str:
    if not protein_sequences:
        raise ValueError("At least one protein sequence is required.")
    chain_ids = _chain_ids(len(protein_sequences))
    lines = ["version: 1", "sequences:"]
    for chain_id, sequence in zip(chain_ids, protein_sequences, strict=True):
        lines.extend(
            [
                "  - protein:",
                f"      id: {chain_id}",
                f"      sequence: {sequence}",
            ]
        )
    lines.extend(
        [
            "  - ligand:",
            f"      id: {ligand_id}",
            f"      smiles: '{_quote_smiles(ligand_smiles)}'",
        ]
    )
    if include_affinity:
        lines.extend(
            [
                "properties:",
                "  - affinity:",
                f"      binder: {ligand_id}",
            ]
        )
    return "\n".join(lines) + "\n"


def write_boltz_yaml_file(
    output_path: Path,
    *,
    protein_sequences: Sequence[str],
    ligand_smiles: str,
    include_affinity: bool = True,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_text = build_boltz_yaml(
        protein_sequences,
        ligand_smiles=ligand_smiles,
        include_affinity=include_affinity,
    )
    output_path.write_text(yaml_text, encoding="utf-8")


def write_boltz_yaml_for_row(
    row: Mapping[str, str | int],
    *,
    output_dir: Path,
    overwrite: bool = True,
    include_affinity: bool = True,
) -> tuple[Path | None, str]:
    test_smiles = str(row.get("test_smiles", "")).strip()
    if not test_smiles:
        return None, "missing_smiles"
    sequence = str(row.get("nearest_train_seq", "")).strip()
    if not sequence:
        return None, "missing_sequence"

    protein_sequences = _split_protein_sequence(sequence)
    if not protein_sequences:
        return None, "empty_sequence"

    complex_id = str(row.get("test_complex_id", "")).strip()
    row_idx = row.get("test_row_idx")
    row_label = _format_row_idx(row_idx)
    fallback = f"row_{row_label}" if row_label else "row"
    safe_label = _safe_name(complex_id, fallback)
    if row_label:
        filename = f"{row_label}_{safe_label}.yaml"
    else:
        filename = f"{safe_label}.yaml"
    output_path = output_dir / filename

    if output_path.exists() and not overwrite:
        return output_path, "exists"

    write_boltz_yaml_file(
        output_path,
        protein_sequences=protein_sequences,
        ligand_smiles=test_smiles,
        include_affinity=include_affinity,
    )
    return output_path, "ok"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Boltz-2 YAML inputs from baseline output CSV."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help="Input CSV from the embedding baseline.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_YAML_DIR,
        help="Directory to write YAML inputs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite YAML files if they already exist.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of rows to convert.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")

    required = {"test_complex_id", "test_smiles", "nearest_train_seq"}
    rows: list[dict[str, str]] = []
    with args.input_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {args.input_csv}")
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"CSV missing columns: {sorted(missing)}")
        for idx, row in enumerate(reader):
            if args.limit is not None and idx >= args.limit:
                break
            rows.append(dict(row))

    if not rows:
        raise RuntimeError("No rows found in input CSV.")

    counts: dict[str, int] = {}
    for row in rows:
        _, status = write_boltz_yaml_for_row(
            row,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
            include_affinity=True,
        )
        counts[status] = counts.get(status, 0) + 1

    summary = ", ".join(f"{key}: {value}" for key, value in sorted(counts.items()))
    print(
        "Generated Boltz YAML inputs. "
        f"rows: {len(rows)}, output_dir: {args.output_dir}, status: {summary}"
    )


if __name__ == "__main__":
    main()
