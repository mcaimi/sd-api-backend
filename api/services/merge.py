#!/usr/bin/env python
"""
Merge Service

Async wrappers for model merging operations.
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any

from libs.stablediffusion.merge import SD15MergePipeline, SDXLMergePipeline
from libs.stablediffusion.mergeops import MergeConfig, batch_merge_checkpoints
from libs.globals.vars import MergeMethod

logger = logging.getLogger(__name__)


async def merge_models_async(
    model_type: str,
    base_model_path: str,
    target_model_path: str,
    method: str,
    alpha: float,
    output_path: str,
    preserve_metadata: bool = True,
) -> dict:
    """
    Async wrapper for single model merging.

    Args:
        model_type: "sd15" or "sdxl"
        base_model_path: Full path to base model
        target_model_path: Full path to target model
        method: Merge method (linear, slerp, additive, subtract)
        alpha: Merge strength (0.0-1.0)
        output_path: Full path for output model
        preserve_metadata: Whether to preserve metadata

    Returns:
        Dictionary with output_path and merge metadata
    """
    # Select pipeline based on model type
    if model_type == "sd15":
        pipeline = SD15MergePipeline(device="auto")
    else:  # sdxl
        pipeline = SDXLMergePipeline(device="auto")

    # Create merge config
    merge_method = MergeMethod.from_string(method)
    config = MergeConfig(
        method=merge_method,
        alpha=alpha,
        preserve_metadata=preserve_metadata,
        progress_callback=lambda p: logger.info(f"Merge progress: {p*100:.1f}%"),
    )

    # Run merge in thread pool (blocking operation)
    loop = asyncio.get_event_loop()

    def _merge():
        return pipeline.merge_for_pipeline_generator(
            base_model=base_model_path,
            target_model=target_model_path,
            config=config,
            output_path=output_path,
        )

    try:
        logger.info(f"Starting merge: {Path(base_model_path).name} + {Path(target_model_path).name}")
        result_path = await loop.run_in_executor(None, _merge)
        logger.info(f"Merge completed: {result_path}")

        return {
            "output_path": result_path,
            "base_model": base_model_path,
            "target_model": target_model_path,
            "method": method,
            "alpha": alpha,
            "model_type": model_type,
        }
    finally:
        # Clean up pipeline cache
        pipeline.clear_cache()


async def batch_merge_models_async(
    model_type: str,
    base_model_path: str,
    target_model_paths: List[str],
    method: str,
    alpha: float,
    output_dir: str,
    preserve_metadata: bool = True,
) -> dict:
    """
    Async wrapper for batch merging.

    Args:
        model_type: "sd15" or "sdxl"
        base_model_path: Full path to base model
        target_model_paths: List of full paths to target models
        method: Merge method (linear, slerp, additive, subtract)
        alpha: Merge strength (0.0-1.0)
        output_dir: Directory for output models
        preserve_metadata: Whether to preserve metadata

    Returns:
        Dictionary with list of output_paths and metadata
    """
    # Create merge config
    merge_method = MergeMethod.from_string(method)
    config = MergeConfig(
        method=merge_method,
        alpha=alpha,
        preserve_metadata=preserve_metadata,
        progress_callback=lambda p: logger.info(f"Batch merge progress: {p*100:.1f}%"),
    )

    # Run batch merge in thread pool (blocking operation)
    loop = asyncio.get_event_loop()

    def _batch_merge():
        return batch_merge_checkpoints(
            base_checkpoint=base_model_path,
            checkpoint_list=target_model_paths,
            output_dir=output_dir,
            config=config,
        )

    try:
        logger.info(f"Starting batch merge: {len(target_model_paths)} targets")
        output_paths = await loop.run_in_executor(None, _batch_merge)
        logger.info(f"Batch merge completed: {len(output_paths)} models created")

        return {
            "output_paths": output_paths,
            "base_model": base_model_path,
            "target_models": target_model_paths,
            "method": method,
            "alpha": alpha,
            "model_type": model_type,
            "count": len(output_paths),
        }
    except Exception as e:
        logger.error(f"Batch merge failed: {e}", exc_info=True)
        raise


async def recipe_merge_models_async(
    model_type: str,
    base_model_path: str,
    recipe_steps: List[Dict[str, Any]],
    output_path: str,
) -> dict:
    """
    Async wrapper for recipe-based merging.

    Args:
        model_type: "sd15" or "sdxl"
        base_model_path: Full path to base model
        recipe_steps: List of dicts with keys: target_model_path, method, alpha
        output_path: Full path for final merged model

    Returns:
        Dictionary with output_path and recipe execution metadata
    """
    # Select pipeline based on model type
    if model_type == "sd15":
        pipeline = SD15MergePipeline(device="auto")
    else:  # sdxl
        pipeline = SDXLMergePipeline(device="auto")

    # Transform recipe_steps into the format expected by create_merge_recipe
    # create_merge_recipe expects: List[Dict[str, Any]] with target_model, method, alpha
    formatted_steps = []
    for step in recipe_steps:
        formatted_steps.append({
            "target_model": step["target_model_path"],
            "method": step["method"],
            "alpha": step["alpha"],
            "show_progress": True,
        })

    # Create recipe
    recipe = pipeline.create_merge_recipe(
        base_model=base_model_path,
        merge_steps=formatted_steps,
    )

    # Run recipe execution in thread pool (blocking operation)
    loop = asyncio.get_event_loop()

    def _execute_recipe():
        return pipeline.execute_merge_recipe(
            recipe=recipe,
            output_path=output_path,
        )

    try:
        logger.info(f"Starting recipe merge: {len(recipe_steps)} steps")
        success = await loop.run_in_executor(None, _execute_recipe)

        if success:
            logger.info(f"Recipe merge completed: {output_path}")
        else:
            raise RuntimeError("Recipe merge failed")

        return {
            "output_path": output_path,
            "base_model": base_model_path,
            "steps": recipe_steps,
            "model_type": model_type,
            "step_count": len(recipe_steps),
        }
    finally:
        # Clean up pipeline cache
        pipeline.clear_cache()
