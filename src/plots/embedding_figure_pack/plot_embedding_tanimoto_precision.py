from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, cast

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src import PROJECT_ROOT
from src.data_utils.lppdbb.artifacts import (
    DEFAULT_INDEX_PATH,
    load_index,
    validate_index_rows,
)
from src.data_utils.lppdbb.rdkit.distances import (
    DEFAULT_FINGERPRINTS_PATH,
    load_fingerprints,
)

DEFAULT_EMBEDDING_ROOT = PROJECT_ROOT / "data" / "interim" / "lppdbb" / "embeddings"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "processed" / "plots" / "embedding_figure_pack"
)
DEFAULT_SIMILARITY_OUTPUT = (
    DEFAULT_OUTPUT_DIR / "lppdbb_embedding_similarity_vs_tanimoto.png"
)
DEFAULT_PRECISION_OUTPUT = DEFAULT_OUTPUT_DIR / "lppdbb_embedding_precision_at_k.png"


@dataclass(frozen=True)
class EmbeddingRecord:
    row_idx: int
    item_id: str
    embedding_path: Path


def _require_rdkit():
    try:
        from rdkit import DataStructs  # noqa: WPS433  # pyright: ignore[reportMissingImports]
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "RDKit is required for Tanimoto computations. "
            "Install it with `uv sync --extra rdkit`."
        ) from exc
    return DataStructs


def _maybe_require_scipy():
    try:
        from scipy import stats  # type: ignore[import-not-found]
    except ImportError:  # pragma: no cover - optional dependency
        return None
    return stats


def _find_latest_embedding_dir(root: Path) -> Path:
    if not root.exists():
        raise FileNotFoundError(f"Embedding root not found: {root}")
    manifests = list(root.rglob("manifest.json"))
    if not manifests:
        raise FileNotFoundError(
            "No embedding manifests found. Provide --embedding-dir explicitly."
        )
    latest = max(manifests, key=lambda path: path.stat().st_mtime)
    return latest.parent


def _load_embedding_manifest(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_embedding_index(path: Path, output_dir: Path) -> list[EmbeddingRecord]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Embedding index CSV has no header: {path}")
        required = {"row_idx", "item_id", "embedding_path"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"Embedding index missing columns: {sorted(missing)} in {path}"
            )
        records = []
        for row in reader:
            record_path = Path(row["embedding_path"])
            if not record_path.is_absolute():
                record_path = output_dir / record_path
            records.append(
                EmbeddingRecord(
                    row_idx=int(row["row_idx"]),
                    item_id=row["item_id"],
                    embedding_path=record_path,
                )
            )
    return records


def _parse_k_values(raw: str) -> list[int]:
    values = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    if not values:
        raise ValueError("k-values cannot be empty")
    return values


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot embedding similarity vs Tanimoto similarity and precision@k diagnostics."
        )
    )
    parser.add_argument(
        "--embedding-dir",
        type=Path,
        default=None,
        help="Embedding output directory (defaults to latest under embeddings root).",
    )
    parser.add_argument(
        "--embedding-root",
        type=Path,
        default=DEFAULT_EMBEDDING_ROOT,
        help="Root directory to search for embedding manifests.",
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default=DEFAULT_INDEX_PATH,
        help="Path to ligand index CSV for fingerprints.",
    )
    parser.add_argument(
        "--fingerprints-path",
        type=Path,
        default=DEFAULT_FINGERPRINTS_PATH,
        help="Path to cached fingerprints (.npy).",
    )
    parser.add_argument(
        "--match-by",
        choices=("complex_id", "row_idx"),
        default="complex_id",
        help="Match embedding rows to index by complex_id or row_idx.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=2000,
        help="Maximum number of matched items to evaluate.",
    )
    parser.add_argument(
        "--pair-samples",
        type=int,
        default=100000,
        help="Number of random pairs for the similarity vs Tanimoto plot.",
    )
    parser.add_argument(
        "--k-values",
        type=str,
        default="1,5,10,25,50",
        help="Comma-separated k values for precision@k.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for plots.",
    )
    parser.add_argument(
        "--similarity-output",
        type=Path,
        default=DEFAULT_SIMILARITY_OUTPUT,
        help="Output path for similarity vs Tanimoto plot.",
    )
    parser.add_argument(
        "--precision-output",
        type=Path,
        default=DEFAULT_PRECISION_OUTPUT,
        help="Output path for precision@k plot.",
    )
    parser.add_argument(
        "--plot-style",
        choices=("hexbin", "scatter"),
        default="hexbin",
        help="Plot style for similarity vs Tanimoto.",
    )
    return parser.parse_args()


def _map_index_by_complex_id(
    index_rows: list[dict[str, str]],
) -> tuple[dict[str, int], int]:
    mapping: dict[str, int] = {}
    duplicates = 0
    for idx, row in enumerate(index_rows):
        complex_id = row.get("complex_id")
        if not complex_id:
            continue
        if complex_id in mapping:
            duplicates += 1
            continue
        mapping[complex_id] = idx
    return mapping, duplicates


def _match_embeddings(
    embedding_records: Iterable[EmbeddingRecord],
    index_rows: list[dict[str, str]],
    match_by: str,
) -> tuple[list[tuple[EmbeddingRecord, int]], int]:
    matches: list[tuple[EmbeddingRecord, int]] = []
    missing = 0
    if match_by == "complex_id":
        mapping, _ = _map_index_by_complex_id(index_rows)
        for record in embedding_records:
            idx = mapping.get(record.item_id)
            if idx is None:
                missing += 1
                continue
            matches.append((record, idx))
        return matches, missing

    for record in embedding_records:
        if 0 <= record.row_idx < len(index_rows):
            matches.append((record, record.row_idx))
        else:
            missing += 1
    return matches, missing


def _sample_pairs(
    num_items: int, num_pairs: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    if num_items < 2:
        raise RuntimeError("Need at least two items to sample pairs.")
    if num_pairs <= 0:
        raise ValueError("pair-samples must be positive")
    pairs_i = []
    pairs_j = []
    while len(pairs_i) < num_pairs:
        batch = max(10000, num_pairs - len(pairs_i))
        i = rng.integers(0, num_items, size=batch)
        j = rng.integers(0, num_items, size=batch)
        mask = i != j
        pairs_i.extend(i[mask].tolist())
        pairs_j.extend(j[mask].tolist())
    return np.array(pairs_i[:num_pairs]), np.array(pairs_j[:num_pairs])


def _precision_at_k(
    emb_sim: np.ndarray,
    tan_sim: np.ndarray,
    k_values: list[int],
) -> dict[int, np.ndarray]:
    num_items = emb_sim.shape[0]
    max_k = max(k_values)
    if max_k >= num_items:
        max_k = num_items - 1
    if max_k <= 0:
        raise RuntimeError("Need at least two items for precision@k.")
    k_values = [k for k in k_values if k <= max_k]
    if not k_values:
        raise RuntimeError("No valid k values after applying dataset size.")

    precision: dict[int, list[float]] = {k: [] for k in k_values}
    for idx in tqdm(range(num_items), desc="precision@k"):
        emb_candidates = np.argpartition(-emb_sim[idx], max_k)[:max_k]
        emb_ranked = emb_candidates[np.argsort(-emb_sim[idx, emb_candidates])]
        tan_candidates = np.argpartition(-tan_sim[idx], max_k)[:max_k]
        tan_ranked = tan_candidates[np.argsort(-tan_sim[idx, tan_candidates])]
        for k in k_values:
            emb_top = set(emb_ranked[:k])
            tan_top = set(tan_ranked[:k])
            precision[k].append(len(emb_top & tan_top) / k)
    return {k: np.asarray(values, dtype=np.float32) for k, values in precision.items()}


def main() -> None:
    args = _parse_args()

    if args.similarity_output == DEFAULT_SIMILARITY_OUTPUT:
        args.similarity_output = args.output_dir / DEFAULT_SIMILARITY_OUTPUT.name
    if args.precision_output == DEFAULT_PRECISION_OUTPUT:
        args.precision_output = args.output_dir / DEFAULT_PRECISION_OUTPUT.name

    embedding_dir = args.embedding_dir
    if embedding_dir is None:
        embedding_dir = _find_latest_embedding_dir(args.embedding_root)
    if not embedding_dir.exists():
        raise FileNotFoundError(f"Embedding directory not found: {embedding_dir}")

    manifest = _load_embedding_manifest(embedding_dir / "manifest.json")
    if manifest is not None:
        task = str(manifest.get("task", ""))
        if task and task != "molecule":
            raise ValueError(
                f"Embedding task is {task}; Tanimoto diagnostics require molecule embeddings."
            )

    embedding_index_path = embedding_dir / "index.csv"
    if not embedding_index_path.exists():
        raise FileNotFoundError(
            f"Embedding index CSV not found: {embedding_index_path}"
        )

    embedding_records = _load_embedding_index(embedding_index_path, embedding_dir)
    if not embedding_records:
        raise RuntimeError("No embeddings found in index.")

    if not args.index_path.exists():
        raise FileNotFoundError(f"Index CSV not found: {args.index_path}")
    if not args.fingerprints_path.exists():
        raise FileNotFoundError(
            "Fingerprints .npy not found. Run the distances module to generate it."
        )

    index_rows = load_index(args.index_path)
    validate_index_rows(index_rows)
    fingerprints = load_fingerprints(args.fingerprints_path)
    if len(fingerprints) != len(index_rows):
        raise ValueError(
            "Fingerprint count mismatch with index. "
            f"Index rows: {len(index_rows)}, fingerprints: {len(fingerprints)}."
        )

    matches, missing = _match_embeddings(
        embedding_records,
        index_rows,
        args.match_by,
    )
    if len(matches) < 2:
        raise RuntimeError(
            "Not enough matched embeddings for analysis. "
            f"Matched: {len(matches)}, missing: {missing}."
        )

    rng = np.random.default_rng(args.seed)
    if args.max_items is not None and len(matches) > args.max_items:
        selected = rng.choice(len(matches), size=args.max_items, replace=False)
        matches = [matches[idx] for idx in selected]

    embeddings = []
    matched_fps = []
    for record, idx in tqdm(matches, desc="load embeddings"):
        vector = np.load(record.embedding_path)
        vector = np.ravel(vector).astype(np.float32)
        if vector.size == 0:
            continue
        embeddings.append(vector)
        matched_fps.append(fingerprints[idx])

    if not embeddings:
        raise RuntimeError("No valid embeddings loaded.")

    dim = embeddings[0].shape[0]
    for vector in embeddings:
        if vector.shape[0] != dim:
            raise ValueError("Embedding dimensionality mismatch across items.")

    emb_matrix = np.vstack(embeddings).astype(np.float32, copy=False)
    norms = np.linalg.norm(emb_matrix, axis=1)
    nonzero_mask = norms > 0
    if not np.all(nonzero_mask):
        emb_matrix = emb_matrix[nonzero_mask]
        matched_fps = [fp for fp, keep in zip(matched_fps, nonzero_mask) if keep]
        norms = norms[nonzero_mask]
    if emb_matrix.shape[0] < 2:
        raise RuntimeError("Not enough non-zero embeddings for analysis.")

    emb_matrix = emb_matrix / norms[:, None]
    emb_sim = emb_matrix @ emb_matrix.T
    np.fill_diagonal(emb_sim, -np.inf)

    DataStructs = _require_rdkit()
    num_items = emb_matrix.shape[0]
    tan_sim = np.empty((num_items, num_items), dtype=np.float32)
    for idx, fp in enumerate(tqdm(matched_fps, desc="tanimoto")):
        sims = DataStructs.BulkTanimotoSimilarity(fp, matched_fps)
        tan_sim[idx] = np.asarray(sims, dtype=np.float32)
    np.fill_diagonal(tan_sim, -np.inf)

    pair_count = min(args.pair_samples, num_items * (num_items - 1))
    pair_i, pair_j = _sample_pairs(num_items, pair_count, rng)
    emb_vals = emb_sim[pair_i, pair_j]
    tan_vals = tan_sim[pair_i, pair_j]

    pearson = np.corrcoef(emb_vals, tan_vals)[0, 1]
    spearman = None
    stats = _maybe_require_scipy()
    if stats is not None:
        result = stats.spearmanr(emb_vals, tan_vals)
        result_any = cast(Any, result)
        spearman = float(getattr(result_any, "correlation", result_any[0]))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.similarity_output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    if args.plot_style == "hexbin":
        hb = ax.hexbin(
            emb_vals,
            tan_vals,
            gridsize=60,
            cmap="viridis",
            mincnt=1,
        )
        fig.colorbar(hb, ax=ax, label="Count")
    else:
        ax.scatter(emb_vals, tan_vals, s=6, alpha=0.25, linewidths=0)
    ax.set_title("Embedding similarity vs Tanimoto similarity")
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Tanimoto similarity")
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    annotation = f"pairs: {pair_count}, items: {num_items}, pearson: {pearson:.3f}"
    if spearman is not None:
        annotation += f", spearman: {spearman:.3f}"
    ax.text(
        0.02,
        0.98,
        annotation,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(args.similarity_output, dpi=200)
    plt.close(fig)

    k_values = _parse_k_values(args.k_values)
    precisions = _precision_at_k(emb_sim, tan_sim, k_values)
    k_values = sorted(precisions.keys())
    mean_precision = np.array([np.mean(precisions[k]) for k in k_values])

    args.precision_output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(k_values, mean_precision, marker="o", linewidth=2)
    ax.set_title("Embedding precision@k (Tanimoto neighbors)")
    ax.set_xlabel("k")
    ax.set_ylabel("Precision@k")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(k_values)
    fig.tight_layout()
    fig.savefig(args.precision_output, dpi=200)
    plt.close(fig)

    print(
        "Generated embedding similarity vs Tanimoto plot. "
        f"items: {num_items}, pairs: {pair_count}, output: {args.similarity_output}"
    )
    print(
        "Generated precision@k plot. "
        f"items: {num_items}, k_values: {k_values}, output: {args.precision_output}"
    )


if __name__ == "__main__":
    main()
