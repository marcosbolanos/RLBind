from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
from tqdm import tqdm

from src import PROJECT_ROOT
from src.data_utils.lppdbb.embeddings_esm import ESMEmbedder
from src.data_utils.lppdbb.embeddings_hf import HFEmbedder
from src.data_utils.lppdbb.records import DEFAULT_CSV_PATH

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "interim" / "lppdbb" / "embeddings"
DEFAULT_PROTEIN_MODEL_ESM = "esmc_300m"
DEFAULT_PROTEIN_MODEL_HF = "EvolutionaryScale/esm3-sm-open-v1"
DEFAULT_MOLECULE_MODEL_HF = "DeepChem/ChemBERTa-77M-MLM"

PROTEIN_FASTA_SUFFIXES = {".fa", ".faa", ".fasta", ".fna"}
SMILES_SUFFIXES = {".smi", ".smiles"}

SAFE_NAME_PATTERN = re.compile(r"[^A-Za-z0-9._-]")
PROTEIN_CLEAN_PATTERN = re.compile(r"[^A-Za-z]")


@dataclass(frozen=True)
class InputRecord:
    row_idx: int
    item_id: str
    text: str
    source: str
    length: int


def _safe_name(value: str, fallback: str) -> str:
    cleaned = SAFE_NAME_PATTERN.sub("_", value).strip("._-")
    return cleaned or fallback


def _clean_protein_sequence(sequence: str) -> str:
    cleaned = PROTEIN_CLEAN_PATTERN.sub("", sequence).upper()
    return cleaned


def _resolve_id_column(fieldnames: list[str], id_column: str | None) -> str:
    if id_column:
        return id_column
    for candidate in ("complex_id", "pdb_id", "id", "name"):
        if candidate in fieldnames:
            return candidate
    return fieldnames[0]


def _iter_csv_records(
    path: Path,
    *,
    text_column: str,
    id_column: str | None,
    clean_fn,
    limit: int | None,
) -> Iterator[InputRecord]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        resolved_id_column = _resolve_id_column(list(reader.fieldnames), id_column)
        for row_idx, row in enumerate(reader):
            if limit is not None and row_idx >= limit:
                break
            raw_text = (row.get(text_column) or "").strip()
            if not raw_text:
                continue
            text = clean_fn(raw_text)
            if not text:
                continue
            item_id = (row.get(resolved_id_column) or "").strip()
            if not item_id:
                item_id = f"row_{row_idx}"
            source = f"{path}:{row_idx + 1}"
            yield InputRecord(
                row_idx=row_idx,
                item_id=item_id,
                text=text,
                source=source,
                length=len(text),
            )


def _iter_fasta_records(
    path: Path,
    *,
    clean_fn,
    limit: int | None,
) -> Iterator[InputRecord]:
    current_id: str | None = None
    parts: list[str] = []
    row_idx = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    text = clean_fn("".join(parts))
                    if text:
                        yield InputRecord(
                            row_idx=row_idx,
                            item_id=current_id,
                            text=text,
                            source=f"{path}:{current_id}",
                            length=len(text),
                        )
                        row_idx += 1
                        if limit is not None and row_idx >= limit:
                            return
                header = line[1:].strip()
                current_id = header.split()[0] if header else f"seq_{row_idx}"
                parts = []
                continue
            parts.append(line)
        if current_id is not None:
            text = clean_fn("".join(parts))
            if text:
                yield InputRecord(
                    row_idx=row_idx,
                    item_id=current_id,
                    text=text,
                    source=f"{path}:{current_id}",
                    length=len(text),
                )


def _iter_smiles_records(
    path: Path,
    *,
    limit: int | None,
) -> Iterator[InputRecord]:
    row_idx = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                continue
            parts = line.split()
            if not parts:
                continue
            smiles = parts[0].strip()
            if not smiles:
                continue
            item_id = parts[1].strip() if len(parts) > 1 else f"row_{row_idx}"
            yield InputRecord(
                row_idx=row_idx,
                item_id=item_id,
                text=smiles,
                source=f"{path}:{row_idx + 1}",
                length=len(smiles),
            )
            row_idx += 1
            if limit is not None and row_idx >= limit:
                return


def _iter_records_from_directory(
    path: Path,
    *,
    task: str,
    clean_fn,
    limit: int | None,
) -> Iterator[InputRecord]:
    if task == "protein":
        suffixes = PROTEIN_FASTA_SUFFIXES
    else:
        suffixes = SMILES_SUFFIXES
    matched_files = [p for p in path.rglob("*") if p.suffix.lower() in suffixes]
    matched_files.sort()
    if not matched_files:
        raise ValueError(f"No input files found in {path} for task={task}.")
    row_idx = 0
    for file_path in matched_files:
        if task == "protein":
            record_iter = _iter_fasta_records(file_path, clean_fn=clean_fn, limit=None)
        else:
            record_iter = _iter_smiles_records(file_path, limit=None)
        for record in record_iter:
            if limit is not None and row_idx >= limit:
                return
            yield InputRecord(
                row_idx=row_idx,
                item_id=record.item_id,
                text=record.text,
                source=record.source,
                length=record.length,
            )
            row_idx += 1


def _resolve_input_format(input_path: Path, input_format: str | None, task: str) -> str:
    if input_format:
        return input_format
    if input_path.is_dir():
        return "dir"
    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix in PROTEIN_FASTA_SUFFIXES:
        return "fasta"
    if suffix in SMILES_SUFFIXES:
        return "smiles"
    if task == "protein":
        return "csv"
    return "csv"


def _iter_records(
    *,
    input_path: Path,
    input_format: str,
    task: str,
    text_column: str,
    id_column: str | None,
    limit: int | None,
) -> Iterator[InputRecord]:
    if task == "protein":
        clean_fn = _clean_protein_sequence
    else:
        clean_fn = lambda text: text.strip()
    if input_format == "csv":
        yield from _iter_csv_records(
            input_path,
            text_column=text_column,
            id_column=id_column,
            clean_fn=clean_fn,
            limit=limit,
        )
        return
    if input_format == "fasta":
        yield from _iter_fasta_records(input_path, clean_fn=clean_fn, limit=limit)
        return
    if input_format == "smiles":
        yield from _iter_smiles_records(input_path, limit=limit)
        return
    if input_format == "dir":
        yield from _iter_records_from_directory(
            input_path,
            task=task,
            clean_fn=clean_fn,
            limit=limit,
        )
        return
    raise ValueError(f"Unsupported input format: {input_format}")


def _batched(
    records: Iterable[InputRecord], batch_size: int
) -> Iterator[list[InputRecord]]:
    batch: list[InputRecord] = []
    for record in records:
        batch.append(record)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def embed_records(
    records: Iterable[InputRecord],
    *,
    embedder,
    output_dir: Path,
    batch_size: int,
    log_every: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir = output_dir / "vectors"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "index.csv"

    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "row_idx",
                "item_id",
                "source",
                "input_length",
                "embedding_path",
            ],
        )
        writer.writeheader()
        total = 0
        progress = tqdm(_batched(records, batch_size), desc="Embedding", unit="batch")
        for batch_idx, batch in enumerate(progress):
            texts = [record.text for record in batch]
            pooled = embedder.embed_texts(texts)
            for record, embedding in zip(batch, pooled, strict=True):
                safe_id = _safe_name(record.item_id, f"row_{record.row_idx}")
                filename = f"{record.row_idx:08d}_{safe_id}.npy"
                out_path = embeddings_dir / filename
                np.save(out_path, embedding)
                writer.writerow(
                    {
                        "row_idx": record.row_idx,
                        "item_id": record.item_id,
                        "source": record.source,
                        "input_length": record.length,
                        "embedding_path": str(out_path.relative_to(output_dir)),
                    }
                )
                total += 1
            if log_every > 0 and (batch_idx + 1) % log_every == 0:
                progress.set_postfix_str(f"items={total}")


def _build_manifest(
    *,
    task: str,
    backend: str,
    model_name: str,
    input_path: Path,
    input_format: str,
    output_dir: Path,
    pooling: str,
    batch_size: int,
    max_length: int | None,
    device: str,
    trust_remote_code: bool,
) -> dict[str, str | int | None]:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": task,
        "backend": backend,
        "model": model_name,
        "input_path": str(input_path),
        "input_format": input_format,
        "output_dir": str(output_dir),
        "pooling": pooling,
        "batch_size": batch_size,
        "max_length": max_length,
        "device": device,
        "trust_remote_code": trust_remote_code,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed protein or molecule inputs into vectors using HF or ESM."
    )
    parser.add_argument(
        "--task",
        choices=["protein", "molecule"],
        default="protein",
        help="Type of input to embed.",
    )
    parser.add_argument(
        "--backend",
        choices=["hf", "esm"],
        default=None,
        help="Embedding backend (defaults to esm for protein, hf for molecule).",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help="Input data file or directory.",
    )
    parser.add_argument(
        "--input-format",
        choices=["auto", "csv", "fasta", "smiles", "dir"],
        default="auto",
        help="Format of the input data.",
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
        help="CSV column to use as the item id (defaults to first column).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name/id to use for embeddings.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for embeddings and index.",
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
        "--pooling",
        choices=["mean", "cls"],
        default="mean",
        help="Pooling strategy for token embeddings.",
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
    parser.add_argument(
        "--log-every",
        type=int,
        default=25,
        help="Log progress every N batches.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    task = args.task
    backend = args.backend
    if backend is None:
        backend = "esm" if task == "protein" else "hf"
    if backend == "esm" and task != "protein":
        raise ValueError("ESM backend is only supported for protein embeddings.")
    if args.model:
        model_name = args.model
    else:
        if backend == "esm":
            model_name = DEFAULT_PROTEIN_MODEL_ESM
        elif task == "protein":
            model_name = DEFAULT_PROTEIN_MODEL_HF
        else:
            model_name = DEFAULT_MOLECULE_MODEL_HF
    input_path = args.input
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")
    input_format = args.input_format
    if input_format == "auto":
        input_format = _resolve_input_format(input_path, None, task)

    text_column = args.sequence_column if task == "protein" else args.smiles_column

    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        model_slug = _safe_name(model_name, task)
        output_dir = DEFAULT_OUTPUT_DIR / task / backend / model_slug

    records = _iter_records(
        input_path=input_path,
        input_format=input_format,
        task=task,
        text_column=text_column,
        id_column=args.id_column,
        limit=args.limit,
    )

    manifest = _build_manifest(
        task=task,
        backend=backend,
        model_name=model_name,
        input_path=input_path,
        input_format=input_format,
        output_dir=output_dir,
        pooling=args.pooling,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    if backend == "esm":
        embedder = ESMEmbedder(
            model_name=model_name,
            device=args.device,
            pooling=args.pooling,
        )
    else:
        embedder = HFEmbedder(
            model_name=model_name,
            device=args.device,
            pooling=args.pooling,
            max_length=args.max_length,
            trust_remote_code=args.trust_remote_code,
        )

    embed_records(
        records,
        embedder=embedder,
        output_dir=output_dir,
        batch_size=args.batch_size,
        log_every=args.log_every,
    )


if __name__ == "__main__":
    main()
