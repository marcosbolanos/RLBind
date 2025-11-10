FROM nvidia/cuda:12.4.1-base-ubuntu22.04 AS base

# Instal uv for containers
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

RUN <<EOT 
groupadd devuser 
useradd -m -d /home/devuser -g devuser devuser
EOT

USER devuser
WORKDIR /home/devuser

# Set uv to use the global python installation, among other things
# Using an absolue path /venv means there's only one venv in the container
ENV UV_LINK_MODE=copy \
  UV_COMPILE_BYTECODE=1 \
  UV_PYTHON=python3.11.13 \
  UV_PROJECT_ENVIRONMENT=/home/devuser/venv

# Pyrosetta installation, this is the longest step
# Install into the project environment that will be used later
RUN uv venv $UV_PROJECT_ENVIRONMENT --python $UV_PYTHON && \
  uv pip install --python $UV_PROJECT_ENVIRONMENT/bin/python pip pyrosetta-installer==0.1.2 && \
  PATH="$UV_PROJECT_ENVIRONMENT/bin:$PATH" $UV_PROJECT_ENVIRONMENT/bin/python -c 'import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()'



FROM base AS build

USER devuser
WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/home/devuser/.cache/uv \
  uv sync \
  --locked \
  --no-dev \
  --no-install-project



FROM build as dev

# I can later add dev-only code here, this part still serves as an alias



FROM base AS prod

USER devuser
WORKDIR /app

COPY --from=build  ${UV_PROJECT_ENVIRONMENT} ${UV_PROJECT_ENVIRONMENT}

COPY . .

ENV PATH="${UV_PROJECT_ENVIRONMENT}/bin:$PATH"

CMD ["python", "main.py"]
