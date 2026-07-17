#!/usr/bin/env python
"""
Merge Router

API endpoints for model merging operations.
"""

from fastapi import APIRouter, HTTPException
from pathlib import Path
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
from libs.shared.config import get_app_config
from libs.shared.utils import enumerate_models

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/merge", response_model=JobResponse)
async def merge_models(request: MergeRequest):
    """
    Merge two models asynchronously.

    Validates model existence, determines model type, and submits
    merge job to the queue. The merged model will be saved to the
    same checkpoint directory as the source models.

    Args:
        request: MergeRequest with model_type, base_model, target_model,
                 method, alpha, output_name, preserve_metadata

    Returns:
        JobResponse with job_id, status, and message

    Raises:
        HTTPException 404: If base_model or target_model not found
    """
    config = get_app_config()
    paths = config.get_model_paths(request.model_type)
    checkpoint_path = Path(paths["checkpoints"])

    # Validate models exist
    available_models = enumerate_models(checkpoint_path)
    if request.base_model not in available_models:
        raise HTTPException(
            status_code=404,
            detail=f"Base model not found: {request.base_model}"
        )
    if request.target_model not in available_models:
        raise HTTPException(
            status_code=404,
            detail=f"Target model not found: {request.target_model}"
        )

    # Build output path
    output_path = checkpoint_path / request.output_name
    if output_path.suffix != ".safetensors":
        output_path = output_path.with_suffix(".safetensors")

    # Submit to job queue
    job_queue = get_job_queue()
    job_id = await job_queue.submit_job(
        job_type="model_merge",
        params={
            "model_type": request.model_type,
            "base_model_path": str(available_models[request.base_model]),
            "target_model_path": str(available_models[request.target_model]),
            "method": request.method,
            "alpha": request.alpha,
            "output_path": str(output_path),
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
                 method, alpha, output_subdir, preserve_metadata

    Returns:
        JobResponse with job_id, status, and message

    Raises:
        HTTPException 404: If base_model or any target_model not found
    """
    config = get_app_config()
    paths = config.get_model_paths(request.model_type)
    checkpoint_path = Path(paths["checkpoints"])

    # Validate models exist
    available_models = enumerate_models(checkpoint_path)
    if request.base_model not in available_models:
        raise HTTPException(
            status_code=404,
            detail=f"Base model not found: {request.base_model}"
        )

    # Validate all target models exist
    missing_targets = [
        target for target in request.target_models
        if target not in available_models
    ]
    if missing_targets:
        raise HTTPException(
            status_code=404,
            detail=f"Target models not found: {', '.join(missing_targets)}"
        )

    # Build output directory
    output_dir = checkpoint_path / request.output_subdir

    # Get full paths for target models
    target_model_paths = [
        str(available_models[target]) for target in request.target_models
    ]

    # Submit to job queue
    job_queue = get_job_queue()
    job_id = await job_queue.submit_job(
        job_type="batch_merge",
        params={
            "model_type": request.model_type,
            "base_model_path": str(available_models[request.base_model]),
            "target_model_paths": target_model_paths,
            "method": request.method,
            "alpha": request.alpha,
            "output_dir": str(output_dir),
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
                 output_name

    Returns:
        JobResponse with job_id, status, and message

    Raises:
        HTTPException 404: If base_model or any step's target_model not found
    """
    config = get_app_config()
    paths = config.get_model_paths(request.model_type)
    checkpoint_path = Path(paths["checkpoints"])

    # Validate models exist
    available_models = enumerate_models(checkpoint_path)
    if request.base_model not in available_models:
        raise HTTPException(
            status_code=404,
            detail=f"Base model not found: {request.base_model}"
        )

    # Validate all target models in steps exist
    missing_targets = []
    for step in request.steps:
        if step.target_model not in available_models:
            missing_targets.append(step.target_model)

    if missing_targets:
        raise HTTPException(
            status_code=404,
            detail=f"Target models not found in recipe steps: {', '.join(missing_targets)}"
        )

    # Build output path
    output_path = checkpoint_path / request.output_name
    if output_path.suffix != ".safetensors":
        output_path = output_path.with_suffix(".safetensors")

    # Transform steps to include full paths
    recipe_steps = [
        {
            "target_model_path": str(available_models[step.target_model]),
            "method": step.method,
            "alpha": step.alpha,
        }
        for step in request.steps
    ]

    # Submit to job queue
    job_queue = get_job_queue()
    job_id = await job_queue.submit_job(
        job_type="recipe_merge",
        params={
            "model_type": request.model_type,
            "base_model_path": str(available_models[request.base_model]),
            "recipe_steps": recipe_steps,
            "output_path": str(output_path),
        },
        callback=recipe_merge_models_async,
    )

    logger.info(f"Recipe merge job {job_id} submitted: {len(request.steps)} steps")

    return JobResponse(
        job_id=job_id,
        status="queued",
        message=f"Recipe merge job submitted: {len(request.steps)} steps"
    )
