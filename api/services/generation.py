#!/usr/bin/env python
"""
Generation Service

Async wrapper around SD pipeline generators for image generation and inpainting.
"""

import asyncio
import logging
from typing import Optional
from pathlib import Path
import json

from api.services.model_cache import get_cache_manager
from api.utils.image_encoding import numpy_to_base64, base64_to_pil
from libs.stablediffusion.funcs import get_random_seed
from libs.globals.vars import RANDOM_BIT_LENGTH
from libs.shared.config import get_app_config
from uuid import uuid4

logger = logging.getLogger(__name__)


# persist generation data to backend
async def persist_to_backend(image_array, metadata) -> bool:
    """Persist image and metadata to disk.

    Returns:
        bool: True if both files saved successfully, False otherwise.
    """
    config = get_app_config()
    img_dir = Path(config.output_images_path)
    json_path = Path(config.output_json_path)
    outfile_name = str(uuid4())

    try:
        # Validate inputs
        if not image_array:
            logger.warning("Empty image_array provided")
            return False
        if not isinstance(metadata, dict):
            logger.warning("Metadata is not a dictionary")
            return False

        # Save image
        image_path = img_dir / f"{outfile_name}.png"
        output_image = base64_to_pil(image_array)
        output_image.save(image_path)
        logger.info(f"Image saved to {image_path}")

        # Save metadata
        json_filepath = json_path / f"{outfile_name}.json"
        with open(json_filepath, "w") as f:
            json.dump(metadata, f)
        logger.info(f"Metadata saved to {json_filepath}")

        return True

    except Exception as e:
        logger.error(f"Failed to persist to backend: {e}", exc_info=True)
        # Optional cleanup: delete partially created image
        try:
            if "image_path" in locals() and image_path.exists():
                image_path.unlink()
        except Exception as cleanup_err:
            logger.warning(f"Cleanup failed: {cleanup_err}")
        return False


async def generate_image_async(
    model_type: str,
    model_path: str,
    positive_prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    cfg: float,
    seed: int,
    scheduler: str,
    loras: Optional[list] = None,
    custom_vae: Optional[str] = None,
) -> dict:
    """
    Async wrapper for image generation.

    Args:
        model_type: "sd15" or "sdxl"
        model_path: Path to model checkpoint
        positive_prompt: Positive prompt text
        negative_prompt: Negative prompt text
        width: Image width
        height: Image height
        steps: Inference steps
        cfg: CFG scale
        seed: Random seed (-1 for random)
        scheduler: Scheduler type
        loras: List of LoRA configs
        custom_vae: Optional custom VAE path

    Returns:
        Dictionary with image_base64 and metadata
    """
    # Get cached pipeline
    cache_manager = get_cache_manager()
    pipeline_gen = await cache_manager.get_pipeline(model_type, model_path, "txt2img")

    # Apply LoRAs if specified
    lora_metadata = {}
    if loras:
        for i, lora_config in enumerate(loras):
            lora_metadata[f"lora_{i}"] = {
                "name": lora_config["name"],
                "lora_path": Path(lora_config.get("path", lora_config["name"])),
                "merge_strength": lora_config["strength"],
                "metadata": {},
            }
        pipeline_gen.addLorasToPipeline(lora_metadata)

    # Apply custom VAE if specified
    if custom_vae:
        pipeline_gen.loadCustomVAE(custom_vae)

    # Generate seed if random
    if seed < 0:
        seed = get_random_seed(RANDOM_BIT_LENGTH)

    # Run generation in thread pool (blocking operation)
    loop = asyncio.get_event_loop()

    def _generate():
        image_array, metadata = pipeline_gen.forward(
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
            scheduler_type=scheduler,
            steps=steps,
            width=width,
            height=height,
            cfg=cfg,
            seed=seed,
        )
        return image_array, metadata

    image_array, metadata = await loop.run_in_executor(None, _generate)

    # Convert to base64
    image_base64 = numpy_to_base64(image_array)

    # persist to backend
    if get_app_config().persist_to_backend:
        await persist_to_backend(
            image_array=image_base64,
            metadata={
                "generation": metadata,
                "loras": lora_metadata,
                "vae": custom_vae,
            },
        )

    return {
        "image_base64": image_base64,
        "metadata": metadata,
        "seed": seed,
    }


async def inpaint_image_async(
    model_type: str,
    model_path: str,
    positive_prompt: str,
    negative_prompt: str,
    image_base64: str,
    mask_base64: str,
    steps: int,
    cfg: float,
    seed: int,
    scheduler: str,
    loras: Optional[list] = None,
    custom_vae: Optional[str] = None,
) -> dict:
    """
    Async wrapper for inpainting.

    Args:
        model_type: "sd15" or "sdxl"
        model_path: Path to model checkpoint
        positive_prompt: Positive prompt text
        negative_prompt: Negative prompt text
        image_base64: Base64-encoded input image
        mask_base64: Base64-encoded mask image
        steps: Inference steps
        cfg: CFG scale
        seed: Random seed (-1 for random)
        scheduler: Scheduler type
        loras: List of LoRA configs
        custom_vae: Optional custom VAE path

    Returns:
        Dictionary with image_base64 and metadata
    """
    # Get cached pipeline
    cache_manager = get_cache_manager()
    pipeline_gen = await cache_manager.get_pipeline(model_type, model_path, "inpaint")

    # Apply LoRAs if specified
    lora_metadata = {}
    if loras:
        for i, lora_config in enumerate(loras):
            lora_metadata[f"lora_{i}"] = {
                "name": lora_config["name"],
                "lora_path": Path(lora_config.get("path", lora_config["name"])),
                "merge_strength": lora_config["strength"],
                "metadata": {},
            }
        pipeline_gen.addLorasToPipeline(lora_metadata)

    # Apply custom VAE if specified
    if custom_vae:
        pipeline_gen.loadCustomVAE(custom_vae)

    # Generate seed if random
    if seed < 0:
        seed = get_random_seed(RANDOM_BIT_LENGTH)

    # Decode images
    input_image = base64_to_pil(image_base64)
    mask_image = base64_to_pil(mask_base64)

    # Run inpainting in thread pool (blocking operation)
    loop = asyncio.get_event_loop()

    def _inpaint():
        image_array, metadata = pipeline_gen.forward_inpaint(
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
            image=input_image,
            mask_image=mask_image,
            scheduler_type=scheduler,
            steps=steps,
            cfg=cfg,
            seed=seed,
        )
        return image_array, metadata

    image_array, metadata = await loop.run_in_executor(None, _inpaint)

    # Convert to base64
    image_base64 = numpy_to_base64(image_array)

    # persist to backend
    if get_app_config().persist_to_backend:
        await persist_to_backend(
            image_array=image_base64,
            metadata={
                "generation": metadata,
                "loras": lora_metadata,
                "vae": custom_vae,
            },
        )

    return {
        "image_base64": image_base64,
        "metadata": metadata,
        "seed": seed,
    }
