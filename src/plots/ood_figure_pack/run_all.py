from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from src.plots.ood_figure_pack.utils import DEFAULT_OUTPUT_DIR

PLOTS = [
    ("ecdf", "src.plots.ood_figure_pack.plot_ecdf_nn_similarity"),
    ("kde", "src.plots.ood_figure_pack.plot_similarity_kde"),
    ("heatmap", "src.plots.ood_figure_pack.plot_nn_heatmap"),
    ("scaffold", "src.plots.ood_figure_pack.plot_scaffold_shift"),
    ("umap", "src.plots.ood_figure_pack.plot_umap_variants"),
    ("mixing", "src.plots.ood_figure_pack.plot_knn_label_mixing"),
    ("properties", "src.plots.ood_figure_pack.plot_property_shift"),
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all OOD figure pack plots.")
    parser.add_argument(
        "--split-path",
        type=Path,
        default=None,
        help="Override split CSV path.",
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default=None,
        help="Override index CSV path.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="Override manifest JSON path.",
    )
    parser.add_argument(
        "--fingerprints-path",
        type=Path,
        default=None,
        help="Override fingerprints .npy path.",
    )
    parser.add_argument(
        "--distances-path",
        type=Path,
        default=None,
        help="Override distances .npy path.",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=None,
        help="Override source CSV path for distance computation.",
    )
    parser.add_argument(
        "--log-output",
        type=Path,
        default=None,
        help="Override preprocess log output path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for plots.",
    )
    parser.add_argument(
        "--variants",
        type=str,
        default=None,
        help="Override UMAP variants string (e.g., nn-sim,test-density).",
    )
    parser.add_argument(
        "--skip",
        type=str,
        default="",
        help="Comma-separated plot keys to skip.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use reduced sizes for faster inspection.",
    )
    return parser.parse_args()


def _extend_arg(cmd: list[str], name: str, value: Path | str | None) -> None:
    if value is None:
        return
    cmd.extend([name, str(value)])


def _build_common_args(args: argparse.Namespace) -> dict[str, str]:
    common: dict[str, str] = {}
    if args.split_path is not None:
        common["--split-path"] = str(args.split_path)
    if args.index_path is not None:
        common["--index-path"] = str(args.index_path)
    if args.manifest_path is not None:
        common["--manifest-path"] = str(args.manifest_path)
    if args.fingerprints_path is not None:
        common["--fingerprints-path"] = str(args.fingerprints_path)
    if args.distances_path is not None:
        common["--distances-path"] = str(args.distances_path)
    if args.csv_path is not None:
        common["--csv-path"] = str(args.csv_path)
    if args.log_output is not None:
        common["--log-output"] = str(args.log_output)
    return common


def _run_module(module: str, extra_args: list[str]) -> None:
    cmd = [sys.executable, "-m", module, *extra_args]
    subprocess.run(cmd, check=True)


def main() -> None:
    args = _parse_args()
    skip = {item.strip() for item in args.skip.split(",") if item.strip()}
    common_args = _build_common_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for key, module in PLOTS:
        if key in skip:
            continue
        extra_args: list[str] = []
        for name, value in common_args.items():
            extra_args.extend([name, value])

        if key == "ecdf":
            _extend_arg(extra_args, "--output", args.output_dir / "lppdbb_nn_ecdf.png")
        if key == "kde":
            _extend_arg(
                extra_args, "--output", args.output_dir / "lppdbb_similarity_kde.png"
            )
        if key == "heatmap":
            _extend_arg(
                extra_args, "--output", args.output_dir / "lppdbb_test_nn_heatmap.png"
            )
        if key == "mixing":
            _extend_arg(
                extra_args, "--output", args.output_dir / "lppdbb_knn_label_mixing.png"
            )
        if key == "properties":
            _extend_arg(
                extra_args, "--output", args.output_dir / "lppdbb_property_shift.png"
            )
        if key == "umap":
            _extend_arg(extra_args, "--output-dir", args.output_dir)
            if args.variants:
                _extend_arg(extra_args, "--variants", args.variants)
        if key == "scaffold":
            _extend_arg(extra_args, "--output-dir", args.output_dir)

        if args.fast:
            if key == "kde":
                extra_args.extend(["--max-train", "2000", "--max-test", "1000"])
            if key == "heatmap":
                extra_args.extend(["--max-train", "5000", "--max-test", "1000"])
            if key == "properties":
                extra_args.extend(["--max-train", "2000", "--max-test", "1000"])
            if key == "mixing":
                extra_args.extend(["--max-items", "3000"])
            if key == "umap":
                extra_args.extend(["--max-clusters", "300"])

        print(f"Running {key}...")
        _run_module(module, extra_args)

    print(f"OOD figure pack complete. Outputs in {args.output_dir}")


if __name__ == "__main__":
    main()
