# RLBind

Research codebase for experiments around few-shot learning for protein generation and optimization.

Most of the repo is small, script-style utilities:

- LP-PDBBind dataset download + preprocessing
- RDKit-backed ligand similarity artifacts (fingerprints, pairwise Tanimoto distances, Butina clusters)
- Train/test split generation from ligand clusters (+ a nearest-neighbor similarity filter)
- Plotting utilities for out-of-distribution (OOD) analysis
- Embedding utilities for protein sequences / ligand SMILES (ESM / HuggingFace)
- Embedding-space retrieval baseline approach benchmark
- RL environment with evaluations/reward computations from Boltz-2, AlphaFold2, Rosetta, and other fitness orcales

## Requirements

- Python >= 3.12 (see `.python-version`)
- Dependency manager: `uv` (recommended)
- Optional: RDKit for ligand distances/clustering (`uv sync --extra rdkit`)

Some parts of the repo pull in heavy scientific dependencies (Torch/JAX/ESM/PyRosetta). Install only what you need.

## Setup

If you cloned from git and want the vendored tooling repos, initialize submodules:

```bash
git submodule update --init --recursive
```

Install Python deps with `uv`:

```bash
uv sync
```

If you want RDKit-backed utilities (ligand preprocessing, fingerprints, Tanimoto distances, Butina clustering):

```bash
uv sync --extra rdkit
```

## Data layout

By convention:

- `data/external/`: raw downloaded datasets (e.g. `LP_PDBBind.csv`)
- `data/interim/`: cached artifacts (indexes, manifests, `.npy` distance vectors, embeddings)
- `data/processed/`: plots and downstream outputs

Paths are resolved via `src.PROJECT_ROOT` so scripts can be run from anywhere.

## LP-PDBBind quickstart

Download the LP-PDBBind CSV into `data/external/LP_PDBBind.csv`:

```bash
uv run -m src.general_utils.download_data
```

Compute RDKit artifacts (fingerprints + condensed pairwise Tanimoto distances) and write an index/manifest under `data/interim/lppdbb/`:

```bash
uv run -m src.data_utils.lppdbb.rdkit.distances
```

Cluster ligands with Butina clustering (also writes `data/interim/lppdbb/lppdbb_ligand_clusters.csv` and a plot by default):

```bash
uv run -m src.plots.plot_lppdbb_clusters
```

Generate a train/test split based on ligand clusters (+ a nearest-neighbor similarity filter that moves overly-similar test ligands into train):

```bash
uv run -m src.data_utils.lppdbb.splits
```

Generate the OOD figure pack (writes into `data/processed/plots/ood_figure_pack/` by default):

```bash
uv run -m src.plots.ood_figure_pack.run_all
```

Tip: most modules support `--help`.

## Embeddings

Embed LP-PDBBind protein sequences and ligand SMILES with a shared ordering:

```bash
uv run python src/scripts/embed_lppdbb_all.py
```

For more control (input format, model choice, pooling, device), use the underlying embedding CLI:

```bash
uv run -m src.data_utils.lppdbb.embed_data --help
```

Embeddings are written under `data/interim/lppdbb/embeddings/<task>/<backend>/<model_slug>/`.

## Generator baseline: embedding NN retrieval

`src/generator/embedding_baseline1/retrieve_nearest_train_binders.py` implements a simple baseline:

- load ligand embeddings (typically the molecule embeddings output above)
- for each test ligand, find the nearest train ligand in embedding space
- copy over the train ligand's protein sequence ("binder")
- write a results CSV and generate Boltz-style YAML inputs per test item

Run it after you have:

- a train/test split (`data/interim/lppdbb/lppdbb_train_test_split.csv`)
- a ligand index (`data/interim/lppdbb/lppdbb_index.csv`)
- molecule embeddings (`data/interim/lppdbb/embeddings/molecule/...`)

```bash
uv run python src/generator/embedding_baseline1/retrieve_nearest_train_binders.py
```

Outputs default to `data/processed/generator/embedding_baseline1/`.

## BindCraft integration (optional)

There is a thin wrapper around BindCraft validation filters in `src/bindcraft_utils/filters.py`.
This path is optional and requires extra setup (BindCraft code + AlphaFold/ColabDesign tooling + PyRosetta).

Example timing script:

```bash
uv run python src/scripts/time_bindcraft_filtering.py
```

## Repo structure

- `src/data_utils/lppdbb/`: LP-PDBBind helpers (records, artifacts, embeddings, splits)
- `src/plots/`: plotting utilities (cluster plots, UMAPs, OOD figure packs)
- `src/scripts/`: small orchestration scripts
- `src/generator/`: baseline generators / data export utilities
- `notebooks/`: exploratory notebooks
- `bindcraft-fork/`, `boltzgen-fork/`: vendored upstream-ish code (see their READMEs)

## Notes

- Large arrays (e.g. distance vectors, embeddings) live in `data/` and should generally not be committed.
- RDKit-dependent code is guarded; if you see an import error, install with `uv sync --extra rdkit`.
