#!/usr/bin/env python
"""
Model Path Resolution

Resolves the checkpoint, LoRA and custom VAE file names of a generation
request to files under the configured model directories (subfolders
included), and the shared components directory of transformer-only
architectures, for the generation routers.
"""

from pathlib import Path
from typing import List, Optional, Tuple

from fastapi import HTTPException

from metastable import ConfigurationError

from api.config import get_app_config
from api.constants import TRANSFORMER_MODEL_TYPES
from api.services.model_files import resolve_checkpoint_from_registry


def _resolve_or_404(name: str, directory: str, kind: str) -> Path:
    path = resolve_checkpoint_from_registry(name, directory)
    if path is None:
        raise HTTPException(status_code=404, detail=f"{kind} not found: {name}")
    return path


def resolve_generation_paths(model_type: str, request) -> Tuple[Path, Optional[List[dict]], Optional[str]]:
    """
    Resolve the model files a generation/inpaint request names.

    Returns:
        (model_path, loras, custom_vae) where loras is the job's list of
        {"name": str, "path": str, "strength": float} (or None) and
        custom_vae the VAE path (or None); both are JSON-serializable, as
        they end up in the persisted generation metadata

    Raises:
        HTTPException 404 if any of the named files does not exist
    """
    paths = get_app_config().get_model_paths(model_type)
    model_path = _resolve_or_404(request.model_checkpoint, paths["checkpoints"], "Model checkpoint")

    loras = None
    if request.loras:
        loras = [
            {
                "name": lora.name,
                "path": str(_resolve_or_404(lora.name, paths["loras"], "LoRA")),
                "strength": lora.strength,
            }
            for lora in request.loras
        ]

    custom_vae = None
    if request.custom_vae:
        custom_vae = str(_resolve_or_404(request.custom_vae, paths["vaes"], "VAE"))

    return model_path, loras, custom_vae


def resolve_components_dir(model_type: str, variant: Optional[str] = None) -> Optional[str]:
    """
    The shared components directory a generation request needs: None for
    single-file architectures.

    Raises:
        HTTPException 400 if the directory is not configured
    """
    if model_type not in TRANSFORMER_MODEL_TYPES:
        return None
    try:
        return get_app_config().components_path(model_type, variant)
    except ConfigurationError as e:
        raise HTTPException(status_code=400, detail=str(e))
