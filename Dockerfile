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

RUN apt-get update && \
  apt-get install -y git



FROM base AS build

USER devuser

WORKDIR /home/devuser
ENV VIRTUAL_ENV=/home/devuser/venv
ENV UV_PROJECT_ENVIRONMENT=$VIRTUAL_ENV

RUN uv venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY --chown=1000:1000 pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/home/devuser/.cache/uv,uid=1000,gid=1000 \
  uv sync \
  --no-dev \
  --no-install-project

# Note: pyrosetta_installer uses pip internally, so we cache pip's directory
RUN --mount=type=cache,target=/home/devuser/.cache/pip,uid=1000,gid=1000 \
  --mount=type=cache,target=/home/devuser/.cache/uv,uid=1000,gid=1000 \
  uv pip install pip pyrosetta-installer==0.1.2 && \
  python -c 'import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()'



FROM build as dev

WORKDIR /home/devuser

COPY --from=build /home/devuser/venv ./venv

# Optionally install dev tools, defined on a local file
COPY --chown=1000:1000 .optional .optional
RUN if [ -f .optional/dev-setup.sh ]; then \
  echo 'Installing dev tools...'; \
  bash .optional/dev-setup.sh || echo 'Dev tools installation failed, continuing...'; \
  fi

USER devuser



FROM build AS prod

USER devuser
WORKDIR /app

COPY --from=dev ${UV_PROJECT_ENVIRONMENT} ${UV_PROJECT_ENVIRONMENT}

COPY . .

ENV PATH="${UV_PROJECT_ENVIRONMENT}/bin:$PATH"

CMD ["python", "main.py"]
