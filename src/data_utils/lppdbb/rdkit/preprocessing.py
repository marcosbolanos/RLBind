from __future__ import annotations

import csv
from pathlib import Path

from src.data_utils.lppdbb.records import (
    DEFAULT_CSV_PATH,
    LigandPreprocessReport,
    LigandRecord,
)


def _require_rdkit():
    try:
        from rdkit import Chem  # noqa: WPS433, TID252  # pyright: ignore[reportMissingImports]
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "RDKit is required for LP-PDBBind ligand preprocessing. "
            "Install it with `uv sync --extra rdkit`."
        ) from exc
    return Chem


def _count_csv_rows(csv_path: Path) -> int:
    with csv_path.open(encoding="utf-8") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def preprocess_ligands_from_csv(
    csv_path: str | Path = DEFAULT_CSV_PATH,
    *,
    smiles_column: str = "smiles",
    id_column: str | None = None,
    sanitize: bool = True,
    remove_hs: bool = True,
    report_path: str | Path | None = None,
    show_progress: bool = True,
) -> tuple[list[LigandRecord], list[str], LigandPreprocessReport]:
    Chem = _require_rdkit()
    try:
        from rdkit import RDLogger  # noqa: WPS433  # pyright: ignore[reportMissingImports]
    except ImportError:  # pragma: no cover - depends on environment
        RDLogger = None
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    if RDLogger is not None:
        RDLogger.DisableLog("rdApp.warning")  # type: ignore[attr-defined]
        RDLogger.DisableLog("rdApp.error")  # type: ignore[attr-defined]

    total_rows = None
    if show_progress:
        total_rows = _count_csv_rows(csv_path)

    records: list[LigandRecord] = []
    skipped: list[str] = []
    skipped_counts: dict[str, int] = {}
    log_entries: list[dict[str, str]] = []
    report_path = Path(report_path) if report_path is not None else None

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {csv_path}")
        if id_column is None:
            id_column = "" if "" in reader.fieldnames else reader.fieldnames[0]

        iterator = reader
        if show_progress:
            from tqdm import tqdm as _tqdm  # noqa: WPS433

            iterator = _tqdm(
                reader,
                total=total_rows,
                desc="Preprocessing ligands",
            )

        for idx, row in enumerate(iterator):
            smiles = (row.get(smiles_column) or "").strip()
            complex_id = (row.get(id_column) or "").strip() or f"row_{idx}"
            if not smiles:
                skipped.append(complex_id)
                skipped_counts["missing_smiles"] = (
                    skipped_counts.get("missing_smiles", 0) + 1
                )
                log_entries.append(
                    {
                        "complex_id": complex_id,
                        "reason": "missing_smiles",
                        "detail": "",
                        "smiles": "",
                    }
                )
                continue
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            if mol is None:
                skipped.append(complex_id)
                skipped_counts["parse_failed"] = (
                    skipped_counts.get("parse_failed", 0) + 1
                )
                log_entries.append(
                    {
                        "complex_id": complex_id,
                        "reason": "parse_failed",
                        "detail": "MolFromSmiles returned None",
                        "smiles": smiles,
                    }
                )
                continue
            if sanitize:
                try:
                    Chem.SanitizeMol(mol)
                except Exception as exc:  # pragma: no cover - depends on RDKit
                    skipped.append(complex_id)
                    skipped_counts["sanitize_failed"] = (
                        skipped_counts.get("sanitize_failed", 0) + 1
                    )
                    log_entries.append(
                        {
                            "complex_id": complex_id,
                            "reason": "sanitize_failed",
                            "detail": str(exc),
                            "smiles": smiles,
                        }
                    )
                    continue
            if remove_hs:
                mol = Chem.RemoveHs(mol)
            canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
            records.append(
                LigandRecord(
                    complex_id=complex_id,
                    ligand_path=csv_path,
                    smiles=canonical_smiles,
                    mol=mol,
                )
            )

    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["complex_id", "reason", "detail", "smiles"],
            )
            writer.writeheader()
            writer.writerows(log_entries)

    report = LigandPreprocessReport(
        total_rows=len(records) + len(skipped),
        retained=len(records),
        skipped_counts=skipped_counts,
        log_path=report_path,
    )

    return records, skipped, report


def preprocess_ligands(
    csv_path: str | Path = DEFAULT_CSV_PATH,
    *,
    smiles_column: str = "smiles",
    id_column: str | None = None,
    sanitize: bool = True,
    remove_hs: bool = True,
    show_progress: bool = True,
) -> tuple[list[LigandRecord], list[str]]:
    records, skipped, _ = preprocess_ligands_from_csv(
        csv_path,
        smiles_column=smiles_column,
        id_column=id_column,
        sanitize=sanitize,
        remove_hs=remove_hs,
        show_progress=show_progress,
    )
    return records, skipped


def preprocess_ligands_with_report(
    csv_path: str | Path = DEFAULT_CSV_PATH,
    *,
    smiles_column: str = "smiles",
    id_column: str | None = None,
    sanitize: bool = True,
    remove_hs: bool = True,
    report_path: str | Path | None = None,
    show_progress: bool = True,
) -> tuple[list[LigandRecord], list[str], LigandPreprocessReport]:
    return preprocess_ligands_from_csv(
        csv_path,
        smiles_column=smiles_column,
        id_column=id_column,
        sanitize=sanitize,
        remove_hs=remove_hs,
        report_path=report_path,
        show_progress=show_progress,
    )
