#!/usr/bin/env python
"""
Models Router

Endpoints for listing and querying model metadata.
"""

from typing import List
from fastapi import APIRouter, HTTPException

from api.models.system import ModelListResponse, ModelMetadataResponse
from libs.shared.config import get_app_config
from libs.shared.utils import enumerate_models, read_safetensors_header

router = APIRouter()


@router.get("/{model_type}/{resource_type}", response_model=ModelListResponse)
async def list_models(model_type: str, resource_type: str):
    """
    List available models.

    Args:
        model_type: "sd15" or "sdxl"
        resource_type: "checkpoints", "loras", or "vaes"

    Returns:
        List of models with names and paths
    """
    if model_type not in ["sd15", "sdxl"]:
        raise HTTPException(
            status_code=400, detail="Invalid model_type. Must be 'sd15' or 'sdxl'"
        )

    if resource_type not in ["checkpoints", "loras", "vaes"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid resource_type. Must be 'checkpoints', 'loras', or 'vaes'",
        )

    config = get_app_config()
    paths = config.get_model_paths(model_type)
    path = paths[resource_type]

    models = enumerate_models(path)

    return ModelListResponse(
        model_type=model_type,
        resource_type=resource_type,
        models=[
            {"name": name, "path": str(model_path)}
            for name, model_path in models.items()
        ],
    )


@router.get(
    "/{model_type}/{resource_type}/{model_name}/metadata",
    response_model=ModelMetadataResponse,
)
async def get_model_metadata(model_type: str, resource_type: str, model_name: str):
    """
    Get metadata for a specific model.

    Args:
        model_type: "sd15" or "sdxl"
        resource_type: "checkpoints", "loras", or "vaes"
        model_name: Model filename

    Returns:
        Model metadata from safetensors header
    """
    if model_type not in ["sd15", "sdxl"]:
        raise HTTPException(status_code=400, detail="Invalid model_type")

    if resource_type not in ["checkpoints", "loras", "vaes"]:
        raise HTTPException(status_code=400, detail="Invalid resource_type")

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
