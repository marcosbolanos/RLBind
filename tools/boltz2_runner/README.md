# boltz2-runner

Small, isolated `uv` project to:

1) submit an MSA workflow to Rowan
2) download/extract BOLTZ-format MSAs
3) run `boltz predict` on a generated YAML

## Setup

```bash
cd tools/boltz2_runner
cp .env.example .env
uv python install 3.12
uv sync
```

Edit `.env` and set `ROWAN_API_KEY`.

## Run

```bash
cd tools/boltz2_runner
uv run python run_boltz_rowan.py
```

Outputs are written under `runs/<name>/` (YAML, MSA files, and Boltz outputs).
