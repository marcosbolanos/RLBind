FROM nvidia/cuda:12.4.1-base-ubuntu22.04 AS base

# Instal uv for containers
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

RUN <<EOT 
groupadd devuser 
useradd -m -d /home/devuser -g devuser devuser
EOT

# Set uv to use the global python installation, among other things
# Using an absolue path /venv means there's only one venv in the container
ENV UV_LINK_MODE=copy \
  UV_COMPILE_BYTECODE=1 \
  UV_PYTHON=python3.12.12

## Pyrosetta installation, this is the longest step
## Install into the project environment that will be used later
#RUN uv pip install pip pyrosetta-installer==0.1.2 && \
#  uv run python -c 'import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()'



FROM base AS dev

USER devuser

WORKDIR /home/devuser
ENV VIRTUAL_ENV=/home/devuser/venv \
  UV_PROJECT_ENVIRONMENT=/home/devuser/venv

RUN uv venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV:$PATH"

COPY --chown=1000:1000 pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/home/devuser/.cache/uv,uid=1000,gid=1000 \
  uv sync \
  --locked \
  --no-dev \
  --no-install-project



FROM dev AS prod

USER devuser
WORKDIR /app

COPY --from=build  ${UV_PROJECT_ENVIRONMENT} ${UV_PROJECT_ENVIRONMENT}

COPY . .

ENV PATH="${UV_PROJECT_ENVIRONMENT}/bin:$PATH"

CMD ["python", "main.py"]
