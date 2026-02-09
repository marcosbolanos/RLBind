from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src import PROJECT_ROOT


DEFAULT_BOLTZ_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "generator"
    / "embedding_baseline1"
    / "boltz_outputs"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "processed" / "plots" / "embedding_baseline1_figure_pack"
)


# Supported score keys.
#
# We support two common output formats:
# - `boltz predict ...`: confidence + affinity scores are written as JSON under
#   `predictions/<input_name>/confidence_*.json` and `affinity_*.json`.
# - `boltzgen` (fork vendored in this repo): some steps write score arrays into
#   `.npz` files (e.g. fold_out_npz / affinity_out_npz).
#
# The plotting code is resilient to missing keys.
DEFAULT_SCORE_KEYS: tuple[str, ...] = (
    # folding/confidence (boltzgen + sometimes boltz-derived)
    "iptm",
    "ptm",
    "ligand_iptm",
    "protein_iptm",
    "interaction_pae",
    "min_interaction_pae",
    "min_design_to_target_pae",
    "design_iptm",
    "design_iiptm",
    "design_to_target_iptm",
    "target_ptm",
    "design_ptm",
    # boltz (JSON confidence)
    "confidence_score",
    "complex_plddt",
    "complex_iplddt",
    "complex_pde",
    "complex_ipde",
    # affinity
    "affinity_pred_value",
    "affinity_probability_binary",
    "affinity_probability_binary1",
    "affinity_probability_binary2",
    "affinity_pred_value1",
    "affinity_pred_value2",
)


def _maybe_tqdm(iterable, *, desc: str):
    try:
        from tqdm import tqdm  # type: ignore[import-not-found]
    except ImportError:  # pragma: no cover
        return iterable
    return tqdm(iterable, desc=desc)


def _is_prob_like(metric: str) -> bool:
    if metric.endswith("_iptm") or metric in {
        "confidence_score",
        "iptm",
        "ptm",
        "target_ptm",
        "design_ptm",
        "complex_plddt",
        "complex_iplddt",
    }:
        return True
    if metric.startswith("affinity_probability_"):
        return True
    return False


def _safe_float(value: object) -> float | None:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _load_json(path: Path) -> dict[str, object] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    out: dict[str, object] = {}
    for key, value in data.items():
        if not isinstance(key, str):
            continue
        out[key] = value
    return out


def _extract_scalar_scores(data: dict[str, object]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for key, value in data.items():
        if not isinstance(key, str):
            continue
        val = _safe_float(value)
        if val is None:
            continue
        scores[key] = val
    return scores


def _compute_best_idx(arr_iptm: np.ndarray, arr_ptm: np.ndarray) -> int | None:
    iptm = np.asarray(arr_iptm)
    ptm = np.asarray(arr_ptm)

    # Accept either (n,) or (n, 1); anything else is ambiguous.
    if iptm.ndim == 2 and iptm.shape[1] == 1:
        iptm = iptm[:, 0]
    if ptm.ndim == 2 and ptm.shape[1] == 1:
        ptm = ptm[:, 0]

    if iptm.ndim != 1 or ptm.ndim != 1:
        return None
    if iptm.shape[0] == 0 or ptm.shape[0] == 0:
        return None
    if iptm.shape[0] != ptm.shape[0]:
        return None

    confidence = 0.8 * iptm + 0.2 * ptm
    if confidence.size == 0:
        return None
    return int(np.nanargmax(confidence))


def _extract_scalar_metric(
    arr: np.ndarray,
    *,
    best_idx: int | None,
    n_samples: int | None,
) -> float | None:
    a = np.asarray(arr)
    if a.dtype == object:
        return None

    if a.ndim == 0:
        return _safe_float(a.item())

    if a.size == 1:
        return _safe_float(np.ravel(a)[0].item())

    if best_idx is not None and n_samples is not None and a.shape[0] == n_samples:
        sample_val = a[best_idx]
        if np.ndim(sample_val) == 0:
            return _safe_float(sample_val.item())
        if np.size(sample_val) == 1:
            return _safe_float(np.ravel(sample_val)[0].item())
        return None

    # Fall back to a simple first element for 1D arrays.
    if a.ndim == 1 and a.shape[0] > 0:
        return _safe_float(a[0].item())

    return None


@dataclass(frozen=True)
class ScoreRow:
    item_id: str
    scores: dict[str, float]


def _find_score_npzs(root: Path, score_keys: set[str]) -> list[Path]:
    npz_paths = sorted(root.rglob("*.npz"))
    score_npzs: list[Path] = []
    for path in _maybe_tqdm(npz_paths, desc="Scanning npz files"):
        try:
            with np.load(path, allow_pickle=False) as z:
                keys = set(z.files)
        except Exception:
            continue
        if keys & score_keys:
            score_npzs.append(path)
    return score_npzs


def _find_boltz_score_jsons(root: Path) -> tuple[list[Path], list[Path]]:
    confidence = sorted(root.rglob("confidence_*.json"))
    affinity = sorted(root.rglob("affinity_*.json"))
    return confidence, affinity


def _rows_from_boltz_jsons(
    confidence_jsons: list[Path],
    affinity_jsons: list[Path],
    *,
    score_keys: set[str],
) -> list[ScoreRow]:
    # Group by per-input directory (preferred) to avoid brittle filename parsing.
    best_conf_by_item: dict[str, tuple[float | None, dict[str, float]]] = {}
    for path in confidence_jsons:
        item_id = path.parent.name
        payload = _load_json(path)
        if payload is None:
            continue
        scores = _extract_scalar_scores(payload)
        # Keep only the keys we know about, but allow new scalar metrics too.
        filtered = {k: v for k, v in scores.items() if k in score_keys}
        if not filtered:
            continue
        conf = filtered.get("confidence_score")
        prev = best_conf_by_item.get(item_id)
        if prev is None:
            best_conf_by_item[item_id] = (conf, filtered)
            continue
        prev_conf, _ = prev
        if conf is None and prev_conf is not None:
            continue
        if prev_conf is None and conf is not None:
            best_conf_by_item[item_id] = (conf, filtered)
            continue
        if conf is not None and prev_conf is not None and conf > prev_conf:
            best_conf_by_item[item_id] = (conf, filtered)

    affinity_by_item: dict[str, dict[str, float]] = {}
    for path in affinity_jsons:
        item_id = path.parent.name
        payload = _load_json(path)
        if payload is None:
            continue
        scores = _extract_scalar_scores(payload)
        filtered = {k: v for k, v in scores.items() if k in score_keys}
        if not filtered:
            continue
        # If multiple affinity jsons exist, keep the one with most keys.
        prev = affinity_by_item.get(item_id)
        if prev is None or len(filtered) > len(prev):
            affinity_by_item[item_id] = filtered

    item_ids = sorted(set(best_conf_by_item) | set(affinity_by_item))
    rows: list[ScoreRow] = []
    for item_id in item_ids:
        scores: dict[str, float] = {}
        conf_entry = best_conf_by_item.get(item_id)
        if conf_entry is not None:
            _, conf_scores = conf_entry
            scores.update(conf_scores)
        aff_scores = affinity_by_item.get(item_id)
        if aff_scores is not None:
            scores.update(aff_scores)
        if not scores:
            continue
        rows.append(ScoreRow(item_id=item_id, scores=scores))
    return rows


def _load_scores_from_npz(
    path: Path, *, score_keys: tuple[str, ...]
) -> ScoreRow | None:
    try:
        with np.load(path, allow_pickle=False) as z:
            available = set(z.files)

            # Determine best sample index if multi-sample folding metrics are present.
            best_idx: int | None = None
            n_samples: int | None = None

            if "iptm" in available and "ptm" in available:
                best_idx = _compute_best_idx(z["iptm"], z["ptm"])
                try:
                    n_samples = int(np.asarray(z["iptm"]).shape[0])
                except Exception:
                    n_samples = None
            elif "design_to_target_iptm" in available and "design_ptm" in available:
                best_idx = _compute_best_idx(
                    z["design_to_target_iptm"],
                    z["design_ptm"],
                )
                try:
                    n_samples = int(np.asarray(z["design_to_target_iptm"]).shape[0])
                except Exception:
                    n_samples = None

            scores: dict[str, float] = {}
            for key in score_keys:
                if key not in available:
                    continue
                val = _extract_scalar_metric(
                    z[key],
                    best_idx=best_idx,
                    n_samples=n_samples,
                )
                if val is None:
                    continue
                scores[key] = val

            # Derived confidence score (matching the writer convention).
            if "iptm" in available and "ptm" in available:
                iptm = np.asarray(z["iptm"]).reshape(-1)
                ptm = np.asarray(z["ptm"]).reshape(-1)
                if iptm.size == ptm.size and iptm.size > 0:
                    conf = 0.8 * iptm + 0.2 * ptm
                    conf_idx = (
                        best_idx if best_idx is not None else int(np.nanargmax(conf))
                    )
                    conf_val = _safe_float(conf[conf_idx].item())
                    if conf_val is not None:
                        scores["confidence"] = conf_val
            elif "design_to_target_iptm" in available and "design_ptm" in available:
                iptm = np.asarray(z["design_to_target_iptm"]).reshape(-1)
                ptm = np.asarray(z["design_ptm"]).reshape(-1)
                if iptm.size == ptm.size and iptm.size > 0:
                    conf = 0.8 * iptm + 0.2 * ptm
                    conf_idx = (
                        best_idx if best_idx is not None else int(np.nanargmax(conf))
                    )
                    conf_val = _safe_float(conf[conf_idx].item())
                    if conf_val is not None:
                        scores["confidence"] = conf_val

    except Exception:
        return None

    if not scores:
        return None

    return ScoreRow(item_id=path.stem, scores=scores)


def _write_scores_csv(rows: list[ScoreRow], output_path: Path) -> None:
    keys: set[str] = set()
    for row in rows:
        keys.update(row.scores.keys())

    fieldnames = ["item_id", *sorted(keys)]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {"item_id": row.item_id, **row.scores}
            writer.writerow(out)


def _plot_histogram(
    values: np.ndarray,
    *,
    metric: str,
    bins: int,
    output_path: Path,
) -> None:
    clean = values[np.isfinite(values)]
    if clean.size == 0:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(clean, bins=bins, edgecolor="black", linewidth=0.5, color="#4C78A8")

    mean = float(np.mean(clean))
    median = float(np.median(clean))
    ax.axvline(
        mean, color="#F58518", linestyle="--", linewidth=1.25, label=f"mean={mean:.3g}"
    )
    ax.axvline(
        median,
        color="#54A24B",
        linestyle=":",
        linewidth=1.25,
        label=f"median={median:.3g}",
    )

    title_metric = metric.replace("_", " ")
    ax.set_title(f"Boltz-2 {title_metric} distribution (n={clean.size})")
    ax.set_xlabel(title_metric)
    ax.set_ylabel("count")

    if _is_prob_like(metric):
        ax.set_xlim(0.0, 1.0)

    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot dataset-wide distributions for key Boltz-2 scores "
            "(e.g., iptm/ptm and affinity predictions)."
        )
    )
    parser.add_argument(
        "--boltz-output-dir",
        type=Path,
        default=DEFAULT_BOLTZ_OUTPUT_DIR,
        help="Root directory containing Boltz outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for plots and the score CSV.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=50,
        help="Number of histogram bins.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of score npz files to load.",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Disable writing the aggregated scores CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.boltz_output_dir.exists():
        raise FileNotFoundError(f"Boltz output dir not found: {args.boltz_output_dir}")

    score_keys = set(DEFAULT_SCORE_KEYS)

    # Prefer boltz JSON outputs if present.
    confidence_jsons, affinity_jsons = _find_boltz_score_jsons(args.boltz_output_dir)
    if confidence_jsons or affinity_jsons:
        if args.limit is not None:
            confidence_jsons = confidence_jsons[: args.limit]
            affinity_jsons = affinity_jsons[: args.limit]
        rows = _rows_from_boltz_jsons(
            confidence_jsons,
            affinity_jsons,
            score_keys=score_keys,
        )
    else:
        score_npzs = _find_score_npzs(args.boltz_output_dir, score_keys)
        if args.limit is not None:
            score_npzs = score_npzs[: args.limit]

        rows = []
        for path in _maybe_tqdm(score_npzs, desc="Loading score npz"):
            row = _load_scores_from_npz(path, score_keys=DEFAULT_SCORE_KEYS)
            if row is None:
                continue
            rows.append(row)

    if not rows:
        predictions_dir = args.boltz_output_dir / "predictions"
        processed_dir = args.boltz_output_dir / "processed"
        maybe_empty_predictions = (
            predictions_dir.exists()
            and predictions_dir.is_dir()
            and not any(predictions_dir.rglob("*"))
        )
        hint = ""
        if maybe_empty_predictions and processed_dir.exists():
            hint = (
                " It looks like preprocessing ran (processed/ exists) but predictions are missing "
                "(predictions/ is empty)."
            )
        raise RuntimeError(
            "No Boltz-2 scalar scores found to plot." + hint + " "
            "Expected either: (1) boltz outputs under predictions/*/confidence_*.json and/or affinity_*.json, "
            "or (2) score arrays inside .npz files (boltzgen fold_out_npz / affinity_out_npz). "
            f"Searched under: {args.boltz_output_dir}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.no_csv:
        _write_scores_csv(rows, args.output_dir / "boltz2_scores.csv")

    # Collect per-metric values.
    metric_values: dict[str, list[float]] = {}
    for row in rows:
        for key, val in row.scores.items():
            metric_values.setdefault(key, []).append(val)

    for metric, vals in sorted(metric_values.items()):
        arr = np.asarray(vals, dtype=np.float32)
        out_path = args.output_dir / f"boltz2_{metric}_hist.png"
        _plot_histogram(arr, metric=metric, bins=args.bins, output_path=out_path)

    print(
        "Generated Boltz-2 score distributions. "
        f"items: {len(rows)}, metrics: {len(metric_values)}, output_dir: {args.output_dir}"
    )


if __name__ == "__main__":
    main()
