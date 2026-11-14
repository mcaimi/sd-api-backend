#!/usr/bin/env python
"""
Generation Service

Async wrapper around SD pipeline generators for image generation and inpainting.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional
from pathlib import Path
import json

from metastable.pipelines import LoraSpec

from api.services.model_cache import get_cache_manager, DIRTY_STATE
from api.utils.image_encoding import numpy_to_base64, base64_to_pil
from api.config import get_app_config
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


def _requested_state(loras: Optional[list], custom_vae: Optional[str]) -> tuple:
    """The LoRA/VAE state a request needs applied, comparable to CLEAN_STATE."""
    lora_state = tuple(
        (str(lora_config.get("path", lora_config["name"])), float(lora_config["strength"]))
        for lora_config in (loras or [])
    )
    return (lora_state, str(custom_vae) if custom_vae else None)


@asynccontextmanager
async def _leased_pipeline(
    model_type: str,
    model_path: str,
    pipeline_type: str,
    loras: Optional[list],
    custom_vae: Optional[str],
):
    """
    Lease a cached pipeline with the requested LoRAs and custom VAE applied.

    The lease is exclusive: run inference inside the context so no other job
    can change the pipeline's LoRAs/VAE (or evict it) in the meantime.

    Yields:
        Tuple of (pipeline generator, JSON-serializable LoRA metadata)
    """
    cache_manager = get_cache_manager()
    state = _requested_state(loras, custom_vae)

    lora_specs = []
    lora_metadata = {}
    for i, lora_config in enumerate(loras or []):
        lora_path = lora_config.get("path", lora_config["name"])
        lora_specs.append(LoraSpec(path=lora_path, strength=lora_config["strength"]))
        lora_metadata[f"lora_{i}"] = {
            "name": lora_config["name"],
            "lora_path": str(lora_path),
            "merge_strength": lora_config["strength"],
        }

    def _apply(pipeline_gen):
        if lora_specs:
            pipeline_gen.add_loras(lora_specs)
        if custom_vae:
            pipeline_gen.load_custom_vae(custom_vae)

    loop = asyncio.get_event_loop()
    for attempt in (1, 2):
        async with cache_manager.acquire(model_type, model_path, pipeline_type, state) as entry:
            if entry.state != state:
                # Until _apply finishes the pipeline is in an unknown state
                entry.state = DIRTY_STATE
                try:
                    await loop.run_in_executor(None, _apply, entry.pipeline)
                except Exception as e:
                    if attempt == 2:
                        raise
                    # Leaving the lease with DIRTY_STATE makes the next acquire reload it
                    logger.warning(f"Failed to apply LoRAs/VAE, reloading pipeline: {e}")
                    continue
                entry.state = state

            yield entry.pipeline, lora_metadata
            return


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
    true_cfg_scale: float = 1.0,
) -> dict:
    """
    Async wrapper for image generation.

    Args:
        model_type: "sd15", "sdxl", "illustrious", or "flux"
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
        true_cfg_scale: Flux-only; >1.0 enables negative_prompt, ignored by
            other architectures

    Returns:
        Dictionary with image_base64 and metadata
    """
    loop = asyncio.get_event_loop()

    def _generate(pipeline_gen):
        extra_kwargs = {"true_cfg_scale": true_cfg_scale} if model_type == "flux" else {}
        image_array, metadata = pipeline_gen.forward(
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
            scheduler_type=scheduler,
            steps=steps,
            width=width,
            height=height,
            cfg=cfg,
            seed=seed,
            **extra_kwargs,
        )
        return image_array, metadata

    # Run generation in thread pool (blocking operation), holding the lease
    async with _leased_pipeline(
        model_type, model_path, "txt2img", loras, custom_vae
    ) as (pipeline_gen, lora_metadata):
        image_array, metadata = await loop.run_in_executor(None, _generate, pipeline_gen)

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
        "seed": metadata["instances"][0]["seed"],
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
    strength: Optional[float] = None,
    true_cfg_scale: float = 1.0,
) -> dict:
    """
    Async wrapper for inpainting.

    Args:
        model_type: "sd15", "sdxl", "illustrious", or "flux"
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
        strength: Flux-only inpaint re-noising strength, ignored by other
            architectures
        true_cfg_scale: Flux-only; >1.0 enables negative_prompt, ignored by
            other architectures

    Returns:
        Dictionary with image_base64 and metadata
    """
    # Decode images
    input_image = base64_to_pil(image_base64)
    mask_image = base64_to_pil(mask_base64)

    loop = asyncio.get_event_loop()

    def _inpaint(pipeline_gen):
        extra_kwargs = {"true_cfg_scale": true_cfg_scale}
        if strength is not None:
            extra_kwargs["strength"] = strength
        image_array, metadata = pipeline_gen.forward_inpaint(
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
            image=input_image,
            mask_image=mask_image,
            scheduler_type=scheduler,
            steps=steps,
            cfg=cfg,
            seed=seed,
            **extra_kwargs,
        )
        return image_array, metadata

    # Run inpainting in thread pool (blocking operation), holding the lease
    async with _leased_pipeline(
        model_type, model_path, "inpaint", loras, custom_vae
    ) as (pipeline_gen, lora_metadata):
        image_array, metadata = await loop.run_in_executor(None, _inpaint, pipeline_gen)

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
        "seed": metadata["instances"][0]["seed"],
    }
