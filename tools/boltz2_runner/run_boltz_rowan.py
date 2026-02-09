import os
import re
import subprocess
import tarfile
from pathlib import Path

import yaml
from dotenv import load_dotenv


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


def _download_rowan_msas(
    *, run_name: str, protein_sequences: list[str], msa_dir: Path
) -> list[Path]:
    """Submit Rowan MSA workflow and return extracted CSV paths."""
    import rowan  # type: ignore[import-not-found]
    from stjames import MSAFormat  # type: ignore[import-not-found]

    rowan.api_key = _require_rowan_api_key()

    # generate MSAs
    print("Submitting MSA workflow...")
    try:
        msa_workflow = rowan.submit_msa_workflow(
            initial_protein_sequences=protein_sequences,
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

    data = {"sequences": sequences}

    input_yaml.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    print(f"Wrote YAML to {input_yaml.resolve()}")

    # run boltz
    print("Running Boltz prediction...")
    cmd = ["boltz", "predict", str(input_yaml), "--out_dir", str(boltz_out_dir)]
    subprocess.run(cmd, check=True)
    print("Done!")


if __name__ == "__main__":
    name = "barnase-barstar-complex"
    protein_sequences = [
        "AQVINTFDGVADYLQTYHKLPDNYITKSEAQALGWVASKGNLADVAPGKSIGGDIFSNREGKLPGKSGRTWREADINYTSGFRNSDRILYSSDWLIYATTDHYQTFTKIR",
        "MKKAVINGEQIRSISDLHQTLKKELALPEYYGENLDALWAALTGWVEYPLVLEWRQFEQSKQLTENGAESVLQVFREAKAEGADITIILS",
    ]

    msa_mode = os.getenv("BOLTZ2_RUNNER_MSA_MODE", "empty")
    run_boltz(name, protein_sequences, msa_mode=msa_mode)
