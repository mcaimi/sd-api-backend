#!/usr/bin/env python
"""
Merge Router

API endpoints for model merging operations.
"""

from fastapi import APIRouter, HTTPException
from pathlib import Path
from typing import Dict, List, Tuple
import asyncio
import logging

from api.models.merge import (
    MergeRequest,
    BatchMergeRequest,
    MergeRecipeRequest,
    MergeResponse,
)
from api.models.generation import JobResponse
from api.services.merge import (
    merge_models_async,
    batch_merge_models_async,
    recipe_merge_models_async,
)
from api.services.job_queue import get_job_queue
from api.config import get_app_config
from metastable.exceptions import CheckpointLoadError, MergeError
from metastable.merge import ARCHITECTURES, check_compatible, inspect_checkpoint
from metastable.models import enumerate_models

router = APIRouter()
logger = logging.getLogger(__name__)


def _available_checkpoints(model_type: str) -> Tuple[Path, Dict[str, Path]]:
    """The model type's checkpoint directory and the checkpoints in it."""
    checkpoint_path = Path(get_app_config().get_model_paths(model_type)["checkpoints"])
    return checkpoint_path, enumerate_models(checkpoint_path)


def _require_models(available: Dict[str, Path], names: List[str], label: str) -> List[Path]:
    """Paths of the named checkpoints; HTTP 404 listing any that are missing."""
    missing = [name for name in names if name not in available]
    if missing:
        raise HTTPException(status_code=404, detail=f"{label} not found: {', '.join(missing)}")
    return [available[name] for name in names]


def _output_checkpoint(checkpoint_path: Path, output_name: str) -> Path:
    output_path = checkpoint_path / output_name
    if output_path.suffix != ".safetensors":
        output_path = output_path.with_suffix(".safetensors")
    return output_path


async def _check_mergeable(model_type: str, base: Path, targets: List[Path]) -> None:
    """
    Check from the safetensors headers alone (no weights are read) that the
    targets can be merged into the base: matching keys, shapes and key
    layouts, no fp8 weights. Raises HTTP 400 with the reason otherwise, so
    incompatible merges fail before they are queued.
    """
    architecture = ARCHITECTURES[model_type]

    def _check():
        base_info = inspect_checkpoint(base, architecture)
        for target in targets:
            check_compatible(base_info, inspect_checkpoint(target, architecture))

    try:
        await asyncio.to_thread(_check)
    except (MergeError, CheckpointLoadError) as e:
        raise HTTPException(status_code=400, detail=f"Checkpoints cannot be merged: {e}")


@router.post("/merge", response_model=JobResponse)
async def merge_models(request: MergeRequest):
    """
    Merge two models asynchronously.

    Validates model existence, determines model type, and submits
    merge job to the queue. The merged model will be saved to the
    same checkpoint directory as the source models.

    Args:
        request: MergeRequest with model_type, base_model, target_model,
                 method, alpha, output_name, output_dtype, preserve_metadata

    Returns:
        JobResponse with job_id, status, and message

    Raises:
        HTTPException 404: If base_model or target_model not found
        HTTPException 400: If the checkpoints cannot be merged
    """
    checkpoint_path, available_models = _available_checkpoints(request.model_type)
    [base_path] = _require_models(available_models, [request.base_model], "Base model")
    [target_path] = _require_models(available_models, [request.target_model], "Target model")
    await _check_mergeable(request.model_type, base_path, [target_path])

    output_path = _output_checkpoint(checkpoint_path, request.output_name)

    # Submit to job queue
    job_queue = get_job_queue()
    job_id = await job_queue.submit_job(
        job_type="model_merge",
        params={
            "model_type": request.model_type,
            "base_model_path": str(base_path),
            "target_model_path": str(target_path),
            "method": request.method,
            "alpha": request.alpha,
            "output_path": str(output_path),
            "output_dtype": request.output_dtype,
            "preserve_metadata": request.preserve_metadata,
        },
        callback=merge_models_async,
    )

    logger.info(f"Merge job {job_id} submitted: {request.output_name}")

    return JobResponse(
        job_id=job_id,
        status="queued",
        message=f"Merge job submitted: {request.output_name}"
    )


@router.post("/batch", response_model=JobResponse)
async def batch_merge_models(request: BatchMergeRequest):
    """
    Merge base model with multiple target models in batch.

    Creates a subdirectory for batch results and merges each
    target model with the base model independently.

    Args:
        request: BatchMergeRequest with model_type, base_model, target_models,
                 method, alpha, output_subdir, output_dtype, preserve_metadata

    Returns:
        JobResponse with job_id, status, and message

    Raises:
        HTTPException 404: If base_model or any target_model not found
        HTTPException 400: If any target cannot be merged into the base
    """
    checkpoint_path, available_models = _available_checkpoints(request.model_type)
    [base_path] = _require_models(available_models, [request.base_model], "Base model")
    target_paths = _require_models(available_models, request.target_models, "Target models")
    await _check_mergeable(request.model_type, base_path, target_paths)

    output_dir = checkpoint_path / request.output_subdir

    # Submit to job queue
    job_queue = get_job_queue()
    job_id = await job_queue.submit_job(
        job_type="batch_merge",
        params={
            "model_type": request.model_type,
            "base_model_path": str(base_path),
            "target_model_paths": [str(path) for path in target_paths],
            "method": request.method,
            "alpha": request.alpha,
            "output_dir": str(output_dir),
            "output_dtype": request.output_dtype,
            "preserve_metadata": request.preserve_metadata,
        },
        callback=batch_merge_models_async,
    )

    logger.info(f"Batch merge job {job_id} submitted: {len(request.target_models)} targets")

    return JobResponse(
        job_id=job_id,
        status="queued",
        message=f"Batch merge job submitted: {len(request.target_models)} targets"
    )


@router.post("/recipe", response_model=JobResponse)
async def recipe_merge_models(request: MergeRecipeRequest):
    """
    Execute a multi-step merge recipe.

    Applies sequential merge operations following the recipe steps.
    Each step merges the current result with a new target model.

    Args:
        request: MergeRecipeRequest with model_type, base_model, steps,
                 output_name, output_dtype

    Returns:
        JobResponse with job_id, status, and message

    Raises:
        HTTPException 404: If base_model or any step's target_model not found
        HTTPException 400: If any step's target cannot be merged into the base
    """
    checkpoint_path, available_models = _available_checkpoints(request.model_type)
    [base_path] = _require_models(available_models, [request.base_model], "Base model")
    target_paths = _require_models(
        available_models,
        [step.target_model for step in request.steps],
        "Target models in recipe steps",
    )
    await _check_mergeable(request.model_type, base_path, target_paths)

    output_path = _output_checkpoint(checkpoint_path, request.output_name)

    # Transform steps to include full paths
    recipe_steps = [
        {
            "target_model_path": str(target_path),
            "method": step.method,
            "alpha": step.alpha,
        }
        for step, target_path in zip(request.steps, target_paths)
    ]

    # Submit to job queue
    job_queue = get_job_queue()
    job_id = await job_queue.submit_job(
        job_type="recipe_merge",
        params={
            "model_type": request.model_type,
            "base_model_path": str(base_path),
            "recipe_steps": recipe_steps,
            "output_path": str(output_path),
            "output_dtype": request.output_dtype,
        },
        callback=recipe_merge_models_async,
    )

    logger.info(f"Recipe merge job {job_id} submitted: {len(request.steps)} steps")

    return JobResponse(
        job_id=job_id,
        status="queued",
        message=f"Recipe merge job submitted: {len(request.steps)} steps"
    )
