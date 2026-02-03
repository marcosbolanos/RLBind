from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from src.data_utils.lppdbb.records import DEFAULT_CSV_PATH
from src.general_utils.download_data import (
    delete_files,
    download_datasets,
    extract_files,
)

LP_PDBB_DATASETS = {
    "LP_PDBBind.csv": "https://github.com/THGLab/LP-PDBBind/raw/refs/heads/master/dataset/LP_PDBBind.csv",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed LP-PDBBind sequences and smiles with shared ordering."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help="Input CSV with seq/smiles columns.",
    )
    parser.add_argument(
        "--protein-backend",
        choices=["esm", "hf"],
        default="esm",
        help="Backend for protein embeddings.",
    )
    parser.add_argument(
        "--protein-model",
        default=None,
        help="Model name for protein embeddings (optional).",
    )
    parser.add_argument(
        "--molecule-backend",
        choices=["hf"],
        default="hf",
        help="Backend for molecule embeddings.",
    )
    parser.add_argument(
        "--molecule-model",
        default=None,
        help="Model name for molecule embeddings (optional).",
    )
    parser.add_argument(
        "--sequence-column",
        default="seq",
        help="CSV column containing protein sequences.",
    )
    parser.add_argument(
        "--smiles-column",
        default="smiles",
        help="CSV column containing molecule SMILES.",
    )
    parser.add_argument(
        "--id-column",
        default=None,
        help="CSV column to use as the item id (optional).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for model forward passes.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to run on (auto, cpu, cuda).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Optional max sequence length for tokenizer truncation.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of items to embed.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow transformers to load custom model code.",
    )
    return parser.parse_args()


def _build_command(
    *,
    task: str,
    backend: str,
    model: str | None,
    args: argparse.Namespace,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "src.data_utils.lppdbb.embed_data",
        "--task",
        task,
        "--backend",
        backend,
        "--input",
        str(args.input),
        "--input-format",
        "csv",
        "--sequence-column",
        args.sequence_column,
        "--smiles-column",
        args.smiles_column,
        "--batch-size",
        str(args.batch_size),
        "--device",
        args.device,
    ]
    if args.id_column:
        cmd.extend(["--id-column", args.id_column])
    if args.max_length is not None:
        cmd.extend(["--max-length", str(args.max_length)])
    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")
    if model:
        cmd.extend(["--model", model])
    return cmd


def _run_command(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def _ensure_input_available(path: Path) -> None:
    if path.exists():
        return
    if path.resolve() == DEFAULT_CSV_PATH.resolve():
        print(f"Input missing at {path}. Downloading LP-PDBBind CSV...")
        download_datasets(LP_PDBB_DATASETS)
        extract_files(LP_PDBB_DATASETS)
        delete_files(LP_PDBB_DATASETS)
    if not path.exists():
        raise FileNotFoundError(f"Input path not found after download: {path}")


def main() -> None:
    args = _parse_args()
    _ensure_input_available(args.input)

    protein_cmd = _build_command(
        task="protein",
        backend=args.protein_backend,
        model=args.protein_model,
        args=args,
    )
    molecule_cmd = _build_command(
        task="molecule",
        backend=args.molecule_backend,
        model=args.molecule_model,
        args=args,
    )

    _run_command(protein_cmd)
    _run_command(molecule_cmd)


if __name__ == "__main__":
    main()
