import argparse
import os
import re
import subprocess
import tarfile
from pathlib import Path

import yaml
from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASELINE_YAML_DIR = (
    REPO_ROOT
    / "data"
    / "processed"
    / "generator"
    / "embedding_baseline1"
    / "boltz_inputs"
)
DEFAULT_BASELINE_OUT_DIR = (
    REPO_ROOT
    / "data"
    / "processed"
    / "generator"
    / "embedding_baseline1"
    / "boltz_outputs"
)


def _validate_protein_sequence(seq: str) -> None:
    if not seq:
        raise ValueError("Protein sequence is empty")
    if not seq.isascii():
        raise ValueError("Protein sequence contains non-ASCII characters")
    if not seq.isupper():
        raise ValueError("Protein sequence must be uppercase (A-Z)")
    if not re.fullmatch(r"[A-Z]+", seq):
        raise ValueError("Protein sequence must contain only letters A-Z")


def _slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = value.strip("-")
    return value or "run"


def _require_rowan_api_key() -> str:
    api_key = os.getenv("ROWAN_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing ROWAN_API_KEY. Set it in tools/boltz2_runner/.env "
            "(copy from .env.example)."
        )
    return api_key


def _read_fasta(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    seqs: list[str] = []
    buf: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if buf:
                seqs.append("".join(buf))
                buf = []
            continue
        buf.append(line)
    if buf:
        seqs.append("".join(buf))
    return seqs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a Boltz YAML and run `boltz predict` (optionally protein-ligand affinity)."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input YAML file or directory of YAMLs. If omitted and no --protein-* args are given, "
        f"defaults to {DEFAULT_BASELINE_YAML_DIR} if it exists.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_BASELINE_OUT_DIR,
        help="Output directory passed to `boltz predict --out_dir`.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="boltz-run",
        help="Run name (used for output folder naming).",
    )
    parser.add_argument(
        "--protein-seq",
        action="append",
        default=None,
        help="Protein sequence (repeatable). Example: --protein-seq MKT...",
    )
    parser.add_argument(
        "--protein-fasta",
        type=Path,
        default=None,
        help="FASTA file containing one or more protein sequences.",
    )
    parser.add_argument(
        "--ligand-smiles",
        type=str,
        default=None,
        help="Ligand SMILES. If provided, a ligand entity is added and affinity is requested.",
    )
    parser.add_argument(
        "--ligand-id",
        type=str,
        default="L",
        help="Ligand ID used in YAML (default: L).",
    )
    parser.add_argument(
        "--msa-mode",
        choices=("empty", "rowan"),
        default="empty",
        help="How to populate protein MSA fields. Default: empty.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("runs"),
        help="Root directory for runs (default: runs/).",
    )

    # Pass-through knobs for `boltz predict`.
    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="Boltz cache directory (passed to `boltz predict --cache`).",
    )
    parser.add_argument(
        "--override",
        action="store_true",
        help="Override existing predictions (passed to `boltz predict --override`).",
    )
    parser.add_argument(
        "--no-kernels",
        action="store_true",
        help="Disable cuEquivariance kernels (passed to `boltz predict --no_kernels`).",
    )

    return parser.parse_args()


def _gather_protein_sequences(args: argparse.Namespace) -> list[str]:
    seqs: list[str] = []
    if args.protein_seq:
        seqs.extend([str(s).strip() for s in args.protein_seq if str(s).strip()])
    if args.protein_fasta is not None:
        if not args.protein_fasta.exists():
            raise FileNotFoundError(f"FASTA not found: {args.protein_fasta}")
        seqs.extend(_read_fasta(args.protein_fasta))
    seqs = [s.replace(" ", "").replace("\t", "").replace("\r", "") for s in seqs]
    if not seqs:
        raise ValueError(
            "Provide at least one protein via --protein-seq and/or --protein-fasta"
        )
    return seqs


def _run_boltz_predict(
    *,
    input_path: Path,
    out_dir: Path,
    cache: Path | None,
    override: bool,
    no_kernels: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["boltz", "predict", str(input_path), "--out_dir", str(out_dir)]
    if cache is not None:
        cmd.extend(["--cache", str(cache)])
    if override:
        cmd.append("--override")
    if no_kernels:
        cmd.append("--no_kernels")
    subprocess.run(cmd, check=True)


def _download_rowan_msas(
    *, run_name: str, protein_sequences: list[str], msa_dir: Path
) -> list[Path]:
    """Submit Rowan MSA workflow and return extracted CSV paths."""
    import rowan  # type: ignore[import-not-found]
    from stjames import MSAFormat  # type: ignore[import-not-found]

    rowan.api_key = _require_rowan_api_key()  # type: ignore[attr-defined]

    # generate MSAs
    print("Submitting MSA workflow...")
    try:
        msa_workflow = rowan.submit_msa_workflow(
            initial_protein_sequences=protein_sequences,  # type: ignore[arg-type]
            output_formats=[MSAFormat.BOLTZ],
            name=run_name,
        )
    except Exception as e:
        # Rowan wraps HTTP errors via httpx.HTTPStatusError.
        response = getattr(e, "response", None)
        if response is not None:
            try:
                body_json = response.json()
            except Exception:
                body_json = None

            if isinstance(body_json, dict) and isinstance(body_json.get("detail"), str):
                detail = body_json["detail"].strip()
                raise RuntimeError(
                    "Rowan API request was rejected.\n"
                    f"HTTP {response.status_code} {response.reason_phrase}\n"
                    f"Detail: {detail}"
                ) from e

            content_type = response.headers.get("content-type", "")
            body_preview = response.text[:2000]
            raise RuntimeError(
                "Rowan API request failed. This is usually invalid input (name/sequences) "
                "or an API-side validation change.\n"
                f"HTTP {response.status_code} {response.reason_phrase}\n"
                f"Content-Type: {content_type}\n"
                f"Body (first 2000 chars):\n{body_preview}"
            ) from e
        raise

    print("Waiting for MSA results...")
    msa_workflow.wait_for_result().fetch_latest(in_place=True)

    print("Downloading MSA files...")
    msa_workflow.download_msa_files(MSAFormat.BOLTZ, path=msa_dir)

    # extract .tar.gz
    tar_path = next(msa_dir.glob("*.tar.gz"), None)
    if tar_path is None:
        raise FileNotFoundError(f"No .tar.gz found under {msa_dir}")

    print(f"Extracting {tar_path}...")
    with tarfile.open(tar_path, "r:gz") as tar_ref:
        tar_ref.extractall(msa_dir, filter="data")
    tar_path.unlink()

    csvs = sorted(msa_dir.glob("*.csv"))
    if len(csvs) != len(protein_sequences):
        raise RuntimeError(
            f"Expected {len(protein_sequences)} MSA CSVs, found {len(csvs)} in {msa_dir}"
        )

    return csvs


def run_boltz(
    name: str, protein_sequences: list[str], *, msa_mode: str = "empty"
) -> None:
    load_dotenv()
    msa_mode = msa_mode.strip().lower()
    if msa_mode not in {"empty", "rowan"}:
        raise ValueError("msa_mode must be one of: 'empty', 'rowan'")

    run_name = _slugify(name)
    run_dir = Path("runs") / run_name
    input_yaml = run_dir / "input.yaml"
    boltz_out_dir = run_dir / "boltz_out"
    msa_dir = run_dir / "msa"

    run_dir.mkdir(parents=True, exist_ok=True)
    boltz_out_dir.mkdir(parents=True, exist_ok=True)

    for seq in protein_sequences:
        _validate_protein_sequence(seq)

    if msa_mode == "rowan":
        msa_dir.mkdir(parents=True, exist_ok=True)
        csvs = _download_rowan_msas(
            run_name=run_name,
            protein_sequences=protein_sequences,
            msa_dir=msa_dir,
        )

        sequences = [
            {"protein": {"id": chr(65 + i), "sequence": s, "msa": str(f)}}
            for i, (s, f) in enumerate(zip(protein_sequences, csvs))
        ]
    else:
        print("Skipping MSA generation; writing YAML with msa: empty")
        sequences = [
            {"protein": {"id": chr(65 + i), "sequence": s, "msa": "empty"}}
            for i, s in enumerate(protein_sequences)
        ]

    data = {"version": 1, "sequences": sequences}

    input_yaml.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    print(f"Wrote YAML to {input_yaml.resolve()}")

    # run boltz
    print("Running Boltz prediction...")
    cmd = ["boltz", "predict", str(input_yaml), "--out_dir", str(boltz_out_dir)]
    subprocess.run(cmd, check=True)
    print("Done!")


if __name__ == "__main__":
    args = _parse_args()
    load_dotenv()

    # Prefer running on existing embedding_baseline1 YAMLs if no explicit inputs were provided.
    has_explicit_proteins = bool(args.protein_seq) or args.protein_fasta is not None
    input_path = args.input
    if (
        input_path is None
        and not has_explicit_proteins
        and DEFAULT_BASELINE_YAML_DIR.exists()
    ):
        input_path = DEFAULT_BASELINE_YAML_DIR

    if input_path is not None:
        if not input_path.exists():
            raise FileNotFoundError(f"Input path not found: {input_path}")
        print(f"Running Boltz on existing YAML input(s): {input_path}")
        _run_boltz_predict(
            input_path=input_path,
            out_dir=args.out_dir,
            cache=args.cache,
            override=args.override,
            no_kernels=args.no_kernels,
        )
        raise SystemExit(0)

    protein_sequences = _gather_protein_sequences(args)
    for seq in protein_sequences:
        _validate_protein_sequence(seq)

    run_name = _slugify(args.name)
    run_dir = args.out_root / run_name
    input_yaml = run_dir / "input.yaml"
    boltz_out_dir = run_dir / "boltz_out"
    msa_dir = run_dir / "msa"

    run_dir.mkdir(parents=True, exist_ok=True)
    boltz_out_dir.mkdir(parents=True, exist_ok=True)

    msa_mode = str(args.msa_mode).strip().lower()
    if msa_mode == "rowan":
        msa_dir.mkdir(parents=True, exist_ok=True)
        csvs = _download_rowan_msas(
            run_name=run_name,
            protein_sequences=protein_sequences,
            msa_dir=msa_dir,
        )
        sequences: list[dict[str, dict[str, object]]] = [
            {"protein": {"id": chr(65 + i), "sequence": s, "msa": str(f)}}
            for i, (s, f) in enumerate(zip(protein_sequences, csvs))
        ]
    else:
        print("Skipping MSA generation; writing YAML with msa: empty")
        sequences = [
            {"protein": {"id": chr(65 + i), "sequence": s, "msa": "empty"}}
            for i, s in enumerate(protein_sequences)
        ]

    ligand_smiles = (args.ligand_smiles or "").strip()
    include_affinity = bool(ligand_smiles)
    if ligand_smiles:
        sequences.append({"ligand": {"id": args.ligand_id, "smiles": ligand_smiles}})

    data: dict[str, object] = {"version": 1, "sequences": sequences}
    if include_affinity:
        data["properties"] = [{"affinity": {"binder": args.ligand_id}}]
    else:
        print(
            "No --ligand-smiles provided; running protein-only (no affinity outputs expected)"
        )

    input_yaml.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    print(f"Wrote YAML to {input_yaml.resolve()}")

    print("Running Boltz prediction...")
    _run_boltz_predict(
        input_path=input_yaml,
        out_dir=boltz_out_dir,
        cache=args.cache,
        override=args.override,
        no_kernels=args.no_kernels,
    )
    print("Done!")
