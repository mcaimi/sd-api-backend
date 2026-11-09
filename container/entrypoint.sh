#!/bin/sh
#
# Container entrypoint for the SD API Backend.
#
# Renders parameters.yaml from environment variables (unless CONFIG_FILE
# points to an existing, user-provided file) and starts uvicorn bound to
# SD_API_HOST (default 0.0.0.0) on SD_API_PORT.

set -eu

: "${SD_API_HOST:=0.0.0.0}"
: "${SD_API_PORT:=8000}"
: "${SD_API_MODELS_DIR:=${APP_ROOT}/data/models}"
: "${SD_API_OUTPUTS_DIR:=${APP_ROOT}/data/outputs}"
: "${SD_API_PERSIST_OUTPUTS:=true}"
: "${SD_API_DEVICE_BACKEND:=auto}"
: "${CONFIG_FILE:=${APP_ROOT}/config/parameters.yaml}"
export CONFIG_FILE

if [ -f "${CONFIG_FILE}" ]; then
    echo "Using existing configuration file: ${CONFIG_FILE}"
else
    echo "Generating configuration file: ${CONFIG_FILE}"
    mkdir -p "$(dirname "${CONFIG_FILE}")"
    cat > "${CONFIG_FILE}" <<YAML
---
checkpoints:
  sd15:
    path: '${SD_API_MODELS_DIR}/checkpoints/SD15'
  sdxl:
    path: '${SD_API_MODELS_DIR}/checkpoints/SDXL'
  illustrious:
    path: '${SD_API_MODELS_DIR}/checkpoints/Illustrious'
  flux:
    path: '${SD_API_MODELS_DIR}/checkpoints/Flux'

loras:
  sd15:
    path: '${SD_API_MODELS_DIR}/loras/SD15'
  sdxl:
    path: '${SD_API_MODELS_DIR}/loras/SDXL'
  illustrious:
    path: '${SD_API_MODELS_DIR}/loras/Illustrious'
  flux:
    path: '${SD_API_MODELS_DIR}/loras/Flux'

vaes:
  sd15:
    path: '${SD_API_MODELS_DIR}/vae/SD15'
  sdxl:
    path: '${SD_API_MODELS_DIR}/vae/SDXL'
  illustrious:
    path: '${SD_API_MODELS_DIR}/vae/Illustrious'
  flux:
    path: '${SD_API_MODELS_DIR}/vae/Flux'

flux:
  components_path: '${SD_API_MODELS_DIR}/flux_components'

device:
  backend: ${SD_API_DEVICE_BACKEND}
  index: 0
  dtype: auto

storage:
  persist_to_backend: ${SD_API_PERSIST_OUTPUTS}
  output_images: '${SD_API_OUTPUTS_DIR}/images'
  output_json: '${SD_API_OUTPUTS_DIR}/json'

api:
  host: "${SD_API_HOST}"
  port: ${SD_API_PORT}
  workers: 1
  job_queue:
    num_workers: 1
    result_ttl: 3600
  model_cache:
    max_models: 2
  cors:
    enabled: true
    allow_origins: ["*"]
YAML
fi

# The listening address and port come from the environment regardless of
# what the configuration file says.
exec uvicorn api.app:app --host "${SD_API_HOST}" --port "${SD_API_PORT}" "$@"
