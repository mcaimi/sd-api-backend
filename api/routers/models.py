#!/usr/bin/env python
"""
Models Router

Endpoints for listing and querying model metadata.
"""

import asyncio
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from api.models.system import ModelEntry, ModelListResponse, ModelMetadataResponse
from api.config import get_app_config
from api.constants import validate_model_type, validate_resource_type
from api.services import model_layout
from api.services.model_files import preview_path, read_sidecar
from metastable.models import enumerate_models, read_safetensors_header

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/{model_type}/{resource_type}", response_model=ModelListResponse)
async def list_models(model_type: str, resource_type: str):
    """
    List available models.

    Args:
        model_type: "sd15", "sdxl", "illustrious", or "flux"
        resource_type: "checkpoints", "loras", or "vaes"

    Returns:
        List of models with names, paths and, when known, Civitai details,
        preview availability and (Flux checkpoints) bundled components
    """
    validate_model_type(model_type)
    validate_resource_type(resource_type)

    config = get_app_config()
    paths = config.get_model_paths(model_type)
    path = paths[resource_type]
    is_flux_checkpoint = model_type == "flux" and resource_type == "checkpoints"

    def _entries():
        return [
            _model_entry(name, model_path, is_flux_checkpoint)
            for name, model_path in enumerate_models(path).items()
        ]

    return ModelListResponse(
        model_type=model_type,
        resource_type=resource_type,
        models=await asyncio.to_thread(_entries),
    )


def _model_entry(name: str, model_path: Path, is_flux_checkpoint: bool) -> ModelEntry:
    """A list entry enriched from the model's sidecar, preview and header."""
    sidecar = read_sidecar(model_path) or {}
    flux_layout = None
    if is_flux_checkpoint:
        flux_layout = sidecar.get("flux_layout")
        if flux_layout is None:
            try:
                flux_layout = model_layout.classify(model_path)
            except Exception as e:
                logger.warning(f"Could not read the Flux layout of {model_path}: {e}")
    return ModelEntry(
        name=name,
        path=str(model_path),
        civitai=sidecar if sidecar.get("source") == "civitai" else None,
        has_preview=preview_path(model_path).is_file(),
        flux_layout=flux_layout,
    )


def _find_model(model_type: str, resource_type: str, model_name: str) -> Path:
    validate_model_type(model_type)
    validate_resource_type(resource_type)
    path = get_app_config().get_model_paths(model_type)[resource_type]
    model_path = enumerate_models(path).get(model_name)
    if not model_path:
        raise HTTPException(status_code=404, detail="Model not found")
    return model_path


@router.get("/{model_type}/{resource_type}/{model_name}/preview")
async def get_model_preview(model_type: str, resource_type: str, model_name: str):
    """Preview image saved next to a model downloaded from (or identified on) Civitai."""
    preview = preview_path(_find_model(model_type, resource_type, model_name))
    if not preview.is_file():
        raise HTTPException(status_code=404, detail="No preview for this model")
    return FileResponse(preview, media_type="image/jpeg")


@router.get(
    "/{model_type}/{resource_type}/{model_name}/metadata",
    response_model=ModelMetadataResponse,
)
async def get_model_metadata(model_type: str, resource_type: str, model_name: str):
    """
    Get metadata for a specific model.

    Args:
        model_type: "sd15", "sdxl", "illustrious", or "flux"
        resource_type: "checkpoints", "loras", or "vaes"
        model_name: Model filename

    Returns:
        Model metadata from safetensors header
    """
    validate_model_type(model_type)
    validate_resource_type(resource_type)

    config = get_app_config()
    paths = config.get_model_paths(model_type)
    path = paths[resource_type]

    models = enumerate_models(path)
    model_path = models.get(model_name)

    if not model_path:
        raise HTTPException(status_code=404, detail="Model not found")

    try:
        metadata = read_safetensors_header(model_path)
        return ModelMetadataResponse(
            name=model_name, path=str(model_path), metadata=metadata
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read metadata: {e}")


@router.get("/{model_type}/{resource_type}/{model_name}/download")
async def download_model(model_type: str, resource_type: str, model_name: str):
    """
    Download a model file (e.g. a merged checkpoint).

    Args:
        model_type: "sd15", "sdxl", "illustrious", or "flux"
        resource_type: "checkpoints", "loras", or "vaes"
        model_name: Model filename
    """
    validate_model_type(model_type)
    validate_resource_type(resource_type)

    config = get_app_config()
    paths = config.get_model_paths(model_type)
    path = paths[resource_type]

    models = enumerate_models(path)
    model_path = models.get(model_name)

    if not model_path:
        raise HTTPException(status_code=404, detail="Model not found")

    return FileResponse(
        model_path, media_type="application/octet-stream", filename=model_name
    )
