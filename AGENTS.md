# RLBind Agent Guide

Audience: agentic coding tools working in this repository.
Goal: be consistent with the current codebase and avoid surprises.

## Repository layout
- `src/`: main package (scripts, utilities, schemas, plotting).
- `data/`: datasets and outputs (external/interim/processed).
- `notebooks/`: exploratory work.
- `bindcraft-fork/`, `boltzgen-fork/`: vendored projects; see their own configs.

## Environment
- Python: 3.12+ (see `pyproject.toml`).
- Dependency manager: `uv` (use `uv sync`, `uv run`).
- Heavy deps: RDKit, JAX, PyRosetta, ColabDesign; import lazily.
- Workspace root is the repo root; avoid hardcoding relative paths.
- Optional extras: `uv sync --extra rdkit` for RDKit-backed utilities.

## Build / lint / test
- There is no root build or test config at the moment.
- Install deps: `uv sync`
- Run a module: `uv run -m src.plots.plot_lppdbb_clusters`
- Run a script file: `uv run python src/scripts/time_bindcraft_filtering.py`
- Linting: no root linter configured.
- Testing: no `tests/` directory in root; if adding tests, prefer `pytest`.
- Single test example (if pytest is added): `uv run pytest path/to/test_file.py::TestClass::test_name`
- Full test example: `uv run pytest`

## Subproject commands (vendored forks)
- `boltzgen-fork/` defines `ruff`, `pytest`, and `mypy` settings in its `pyproject.toml`.
- If working there, run commands from that directory and follow its config.
- `bindcraft-fork/` has a minimal `pyproject.toml` without lint/test tools.

## Code style (Python)
Formatting
- No formatter is enforced; follow the style of the file you touch.
- Use 4-space indents and standard PEP 8 spacing.
- Wrap long lines with parentheses, not backslashes.
- Keep top-level definitions separated by blank lines.

Imports
- Group imports: standard library, third-party, local `src`/project imports.
- Prefer absolute imports from `src` (e.g., `from src.data_utils...`).
- Use `pathlib.Path` for filesystem paths.
- Place `from __future__ import annotations` first when used.
- Keep optional dependencies in lazy import helpers (e.g., `_require_rdkit()`).

Typing
- Target Python 3.12 features: `list[int]`, `dict[str, int]`, `Path | None`.
- Add type hints to new functions when practical.
- Use `from __future__ import annotations` in new modules with heavy type hints.
- Use `Any` for third-party objects that lack stubs (e.g., RDKit, PyRosetta).
- For Pydantic models, use `.model_dump()` for serialization.
- Use `@dataclass(frozen=True)` for immutable record types.

Naming
- Modules and functions: `snake_case`.
- Classes: `PascalCase`.
- Constants: `UPPER_SNAKE_CASE`.
- Use descriptive names for data records (`LigandRecord`, `FilterSettings`).
- Avoid one-letter variables except for small loops or indices.

Error handling
- Raise explicit exceptions with context (`FileNotFoundError`, `ValueError`, `RuntimeError`).
- For optional dependencies, raise `ImportError` with a clear install hint.
- Avoid broad `except` unless it is truly necessary; document why.
- Preserve existing `try/except` patterns around RDKit sanitization.

Logging / output
- Use `print()` in CLI-style scripts and notebooks.
- Avoid global logging configuration unless needed.
- For library functions, return data and let callers decide how to log.

Data paths and I/O
- Use `PROJECT_ROOT` from `src/__init__.py` for absolute project paths.
- Create directories with `mkdir(parents=True, exist_ok=True)`.
- Use UTF-8 and `newline=''` for CSV writes.
- Use `csv.DictReader/DictWriter` with explicit `fieldnames`.
- Keep data files out of `src/` and inside `data/`.
- For LP-PDBBind artifacts, keep `lppdbb_index.csv` + `lppdbb_manifest.json` in `data/interim/lppdbb/` to pin ordering and parameters.

Performance / memory
- Be careful with quadratic pairwise computations (see `lppdbb` clustering).
- Prefer streaming or array-based computation where possible.
- Only materialize large lists when necessary.
- Consider progress bars (`tqdm`) for long loops.

Docstrings and comments
- Add docstrings for public APIs or complex logic.
- Avoid comments for obvious code; prefer clear names.
- TODOs are acceptable but should include context.
- Keep comments in ASCII unless the file already uses Unicode.

Linting / static analysis hints
- Some files use `# noqa` (e.g., `WPS433`, `TID252`) and `pyright` ignores.
- Preserve these if the import pattern stays the same.
- If adding new ignores, keep them minimal and local.

Optional dependencies
- RDKit and PyRosetta are not always available; import lazily.
- Use helper functions like `_require_rdkit()` to gate imports.
- Avoid module-level imports that break in minimal envs.
- Keep RDKit-dependent modules under `src/data_utils/lppdbb/rdkit/` and import them explicitly.

CLI modules
- Prefer `if __name__ == "__main__":` blocks for runnable scripts.
- Keep CLI argument parsing inside a `_parse_args()` helper.
- Use `argparse` with clear help strings.
- Provide sensible defaults rooted at `PROJECT_ROOT`.

Plotting
- Use `matplotlib` patterns from `src/plots` (figure/axes, tight_layout, savefig).
- Always close figures with `plt.close()`.
- Write output plots under `data/processed/plots/` when applicable.

Pydantic models
- Use `BaseModel` for structured config data.
- Prefer explicit fields over dynamic dicts.
- Keep `Config` or model config aligned with current usage.
- Use aliases for fields that are not valid identifiers (see `FilterSettings`).

Interacting with forks
- Treat `bindcraft-fork/` and `boltzgen-fork/` as upstream-ish code.
- Avoid cross-editing unless the task explicitly targets those folders.
- If you must edit, follow their local style and configs.

Cursor / Copilot rules
- No `.cursorrules`, `.cursor/rules/`, or `.github/copilot-instructions.md` found.
- If new rules are added later, update this file.

Agent checklist
- Confirm the task scope (root vs subproject).
- Use `uv` for dependency installation and running modules.
- Respect optional dependencies; avoid breaking imports.
- Prefer `Path` and `PROJECT_ROOT` for filesystem work.
- Keep edits consistent with the surrounding file style.
- Update `AGENTS.md` if tooling changes.

Common command snippets
- Install deps: `uv sync`
- Run a module: `uv run -m src.general_utils.download_data`
- Run a script: `uv run python src/scripts/time_bindcraft_filtering.py`
- Re-run a plot: `uv run -m src.plots.plot_lppdbb_clusters --help`
- Plot train/test Tanimoto: `uv run -m src.plots.plot_lppdbb_split_tanimoto`
- Compute LP-PDBBind distances: `uv run -m src.data_utils.lppdbb.rdkit.distances`
- Create LP-PDBBind splits: `uv run -m src.data_utils.lppdbb.splits`
- Single test (if pytest added): `uv run pytest tests/test_file.py::test_name`
- Run boltzgen lint (if working there): `cd boltzgen-fork && uv run ruff check .`
- Run boltzgen tests (if working there): `cd boltzgen-fork && uv run pytest`

Notes
- `README.md` in the repo root is currently empty.
- There is no CI config or default lint/test runner in the root.
- Keep output files in `data/` and do not commit large datasets.
- Prefer ASCII text in new files unless the existing file uses Unicode.

End
