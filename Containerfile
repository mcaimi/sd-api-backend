# SD API Backend container image
#
# Build:
#   podman build -t sd-api-backend -f Containerfile .
#
# Run:
#   podman run -p 8000:8000 \
#     -v ./models:/opt/app-root/data/models:Z \
#     -v ./outputs:/opt/app-root/data/outputs:Z \
#     sd-api-backend
#
# The image runs as an unprivileged user and is compatible with OpenShift's
# arbitrary UID assignment (all writable paths are owned by group 0).

ARG BASE_IMAGE=docker.io/library/debian:latest
ARG UV_IMAGE=ghcr.io/astral-sh/uv:latest

FROM ${UV_IMAGE} AS uv

# ---------------------------------------------------------------------------
# Builder: install a managed Python and sync the virtualenv with uv
# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS builder

# Python version installed by uv (project requires >= 3.12)
ARG PYTHON_VERSION=3.12
# Project file used for dependency resolution (e.g. pyproject.linux.toml)
ARG PYPROJECT=pyproject.toml
# Extra arguments for `uv sync` (e.g. "--extra cu134" with pyproject.linux.toml)
ARG UV_SYNC_ARGS=""

ENV APP_ROOT=/opt/app-root \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PYTHON_PREFERENCE=only-managed \
    UV_PROJECT_ENVIRONMENT=/opt/app-root/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /uvx /usr/local/bin/

RUN uv python install ${PYTHON_VERSION}

WORKDIR ${APP_ROOT}/src

# Dependencies first, so source changes don't invalidate this layer.
# uv.lock is optional: it is used when present and (re)generated otherwise.
COPY ${PYPROJECT} ./pyproject.toml
COPY README.md uv.lock* ./
RUN uv sync --no-dev --no-install-project --python ${PYTHON_VERSION} ${UV_SYNC_ARGS}

COPY main.py ./
COPY api ./api
COPY webui ./webui
RUN uv sync --no-dev --no-install-project --python ${PYTHON_VERSION} ${UV_SYNC_ARGS}

# ---------------------------------------------------------------------------
# Runtime image
# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE}

ARG APP_UID=1001
# Listening address and port (overridable at runtime with -e SD_API_HOST=... / -e SD_API_PORT=...)
ARG SD_API_HOST=0.0.0.0
ARG SD_API_PORT=8000
# Checkpoint (models) and output directories
ARG SD_API_MODELS_DIR=/opt/app-root/data/models
ARG SD_API_OUTPUTS_DIR=/opt/app-root/data/outputs

ENV APP_ROOT=/opt/app-root \
    HOME=/opt/app-root/home \
    PATH=/opt/app-root/venv/bin:${PATH} \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/opt/app-root/home/.cache/huggingface \
    SD_API_HOST=${SD_API_HOST} \
    SD_API_PORT=${SD_API_PORT} \
    SD_API_MODELS_DIR=${SD_API_MODELS_DIR} \
    SD_API_OUTPUTS_DIR=${SD_API_OUTPUTS_DIR} \
    CONFIG_FILE=/opt/app-root/config/parameters.yaml

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# The virtualenv links to the uv-managed interpreter, so both keep their paths
COPY --from=builder /opt/python /opt/python
COPY --from=builder ${APP_ROOT}/venv ${APP_ROOT}/venv
COPY --from=builder ${APP_ROOT}/src ${APP_ROOT}/src
COPY container/entrypoint.sh /usr/local/bin/entrypoint.sh

# OpenShift runs containers with a random UID that belongs to group 0:
# make every writable path group-owned by root with group permissions
# matching the owner's.
RUN useradd --uid ${APP_UID} --gid 0 --no-create-home --home-dir ${HOME} \
        --shell /usr/sbin/nologin app \
    && mkdir -p ${HOME} ${APP_ROOT}/config ${SD_API_MODELS_DIR} ${SD_API_OUTPUTS_DIR} \
    && chmod 0755 /usr/local/bin/entrypoint.sh \
    && chown -R ${APP_UID}:0 ${APP_ROOT} \
    && chmod -R g=u ${APP_ROOT}

WORKDIR ${APP_ROOT}/src

USER ${APP_UID}

EXPOSE ${SD_API_PORT}

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
