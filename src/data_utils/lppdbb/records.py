from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src import PROJECT_ROOT

DEFAULT_CSV_PATH = PROJECT_ROOT / "data" / "external" / "LP_PDBBind.csv"


@dataclass(frozen=True)
class LigandRecord:
    complex_id: str
    ligand_path: Path
    smiles: str
    mol: Any


@dataclass(frozen=True)
class LigandPreprocessReport:
    total_rows: int
    retained: int
    skipped_counts: dict[str, int]
    log_path: Path | None
