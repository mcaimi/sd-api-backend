#!/usr/bin/env python
"""
Base classes for Stable Diffusion Pipelines

Provides abstract base classes that SD15 and SDXL pipelines inherit from,
eliminating code duplication while allowing model-specific customization.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union
import logging
import random

import numpy as np
import torch
from torch import Generator
from PIL import Image
from diffusers import AutoencoderKL, UNet2DConditionModel, DiffusionPipeline

from libs.globals.vars import RANDOM_BIT_LENGTH, schedulers
from libs.shared.utils import get_gpu
from libs.shared.exceptions import PipelineError
from libs.stablediffusion.funcs import get_random_seed

logger = logging.getLogger(__name__)


def normalize_path(path: Union[str, Path]) -> Path:
    """Convert string or Path to Path object."""
    if isinstance(path, str):
        return Path(path)
    elif isinstance(path, Path):
        return path
    raise TypeError(f"Expected str or Path, got {type(path)}")


def convert_to_pil(image: Union[np.ndarray, Image.Image]) -> Image.Image:
    """Convert numpy array to PIL Image if needed."""
    if isinstance(image, np.ndarray):
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
        return Image.fromarray(image)
    return image


def convert_mask_to_grayscale(mask: Image.Image) -> Image.Image:
    """Ensure mask is in grayscale mode."""
    if mask.mode != "L":
        return mask.convert("L")
    return mask


def load_custom_vae(checkpoint: Union[str, Path]) -> AutoencoderKL:
    """Load a custom Variational Autoencoder from checkpoint."""
    path = normalize_path(checkpoint)
    logger.info("Loading custom VAE: %s", path)
    return AutoencoderKL.from_single_file(
        str(path.absolute()), subfolder="vae", use_safetensors=True
    )


def load_custom_unet(checkpoint: Union[str, Path]) -> UNet2DConditionModel:
    """Load custom UNET weights from checkpoint."""
    path = normalize_path(checkpoint)
    return UNet2DConditionModel.from_single_file(
        str(path.absolute()), subfolder="unet", use_safetensors=True
    )


def gen_noise(width: int, height: int, channels: int = 3) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Generate sample noise image."""
    noise_image = np.random.rand(height, width, channels)
    noise_properties = {
        "width": width,
        "height": height,
        "channels": channels,
        "generation": {
            "status": "idle",
            "output": "noise matrix",
        },
    }
    return noise_image, noise_properties


def format_metadata(
    prompt: str,
    negative_prompt: str = "",
    steps: int = 10,
    width: int = 512,
    height: int = 512,
    cfg: float = 7,
    seed: int = -1,
    scheduler: Optional[str] = None,
) -> Dict[str, Any]:
    """Prepare generation metadata payload."""
    if seed == -1:
        custom_seed = get_random_seed(RANDOM_BIT_LENGTH)
        logger.debug("Generating with random seed: %d", custom_seed)
    else:
        custom_seed = seed
        logger.debug("Generating with constant seed: %d", custom_seed)

    return {
        "instances": [
            {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "num_inference_steps": steps,
                "width": width,
                "height": height,
                "guidance_scale": cfg,
                "seed": custom_seed,
                "scheduler": scheduler,
            }
        ]
    }


class BasePipelineGenerator(ABC):
    """Abstract base class for Stable Diffusion pipeline generators."""

    # Subclasses must define these
    PIPELINE_CLASS = None
    INPAINT_PIPELINE_CLASS = None
    DEFAULT_WIDTH = 512
    DEFAULT_HEIGHT = 512
    MODEL_TYPE = "base"

    def __init__(self, model_checkpoint: str):
        self.model_checkpoint: str = model_checkpoint
        self.pipeline: Optional[DiffusionPipeline] = None
        self.inpaint_pipeline: Optional[DiffusionPipeline] = None
        self.accelerator: Optional[str] = None
        self.dtype: Optional[torch.dtype] = None

    @staticmethod
    @abstractmethod
    def get_resolutions() -> Dict[str, Tuple[int, int]]:
        """Return supported resolutions for this model type."""
        pass

    def _init_device(self) -> None:
        """Initialize GPU/device settings."""
        self.accelerator, self.dtype = get_gpu()

    def _create_generator(self, seed: int) -> Generator:
        """Create a torch Generator with the given seed."""
        if seed == -1:
            actual_seed = get_random_seed(RANDOM_BIT_LENGTH)
        else:
            actual_seed = seed
        return Generator(self.accelerator).manual_seed(actual_seed)

    def _run_inference(
        self,
        pipeline,
        prompt: str,
        negative_prompt: str,
        scheduler_type: str,
        seed: int,
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Run inference on the given pipeline."""
        # Set scheduler
        logger.debug("Using scheduler: %s", scheduler_type)
        pipeline.scheduler = schedulers.get(scheduler_type).from_config(
            pipeline.scheduler.config
        )

        # Create generator
        gen = self._create_generator(seed)

        # Run prediction
        prediction = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            generator=gen,
            **kwargs
        )

        # Format metadata
        metadata = format_metadata(
            prompt=prompt,
            negative_prompt=negative_prompt,
            steps=kwargs.get("num_inference_steps", 10),
            width=kwargs.get("width", self.DEFAULT_WIDTH),
            height=kwargs.get("height", self.DEFAULT_HEIGHT),
            cfg=kwargs.get("guidance_scale", 7),
            seed=seed,
            scheduler=scheduler_type,
        )

        return np.array(prediction.images[0]), metadata

    def forward(
        self,
        positive_prompt: str,
        negative_prompt: str,
        scheduler_type: str,
        steps: int,
        width: int,
        height: int,
        cfg: float,
        seed: int,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Generate an image from text prompts.

        Raises:
            PipelineError: If no pipeline is loaded
        """
        if self.pipeline is None:
            logger.error("Cannot generate image: no pipeline loaded")
            raise PipelineError("No model loaded. Please load a pipeline first.")

        return self._run_inference(
            self.pipeline,
            prompt=positive_prompt,
            negative_prompt=negative_prompt,
            scheduler_type=scheduler_type,
            seed=seed,
            num_inference_steps=steps,
            width=width,
            height=height,
            guidance_scale=cfg,
        )

    def forward_inpaint(
        self,
        positive_prompt: str,
        negative_prompt: str,
        image: Union[np.ndarray, Image.Image],
        mask_image: Union[np.ndarray, Image.Image],
        scheduler_type: str,
        steps: int,
        cfg: float,
        seed: int,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Inpaint an image region.

        Raises:
            PipelineError: If no inpainting pipeline is loaded
        """
        if self.inpaint_pipeline is None:
            logger.error("Cannot inpaint: no inpainting pipeline loaded")
            raise PipelineError("No inpainting model loaded. Please load an inpainting pipeline first.")

        # Ensure proper image formats
        pil_image = convert_to_pil(image)
        pil_mask = convert_mask_to_grayscale(convert_to_pil(mask_image))

        result, metadata = self._run_inference(
            self.inpaint_pipeline,
            prompt=positive_prompt,
            negative_prompt=negative_prompt,
            scheduler_type=scheduler_type,
            seed=seed,
            image=pil_image,
            mask_image=pil_mask,
            num_inference_steps=steps,
            guidance_scale=cfg,
        )

        metadata["inpainting"] = True
        return result, metadata

    def getSchedulerConfig(self) -> Optional[Dict]:
        """Return the current scheduler configuration."""
        if self.inpaint_pipeline is not None:
            return self.inpaint_pipeline.scheduler.config
        elif self.pipeline is not None:
            return self.pipeline.scheduler.config
        return None

    @abstractmethod
    def loadPipeline(self) -> None:
        """Load the main generation pipeline."""
        pass

    @abstractmethod
    def loadInpaintPipeline(self) -> None:
        """Load the inpainting pipeline."""
        pass

    def addLorasToPipeline(self, loras: Optional[Dict] = None) -> None:
        """Add LoRA adapters to the loaded pipeline(s)."""
        if loras is None:
            return

        lora_checkpoints = list(loras.values())

        for entry in lora_checkpoints:
            lora_path = entry.get("lora_path")
            if lora_path is None:
                continue

            weights_file = Path(lora_path)
            strength = entry.get("merge_strength", 0.5)
            adapter_name = f"name_{weights_file.stem}"

            logger.info("Loading LoRA: %s (strength: %.2f)", weights_file, strength)

            for pipeline in [self.pipeline, self.inpaint_pipeline]:
                if pipeline is not None:
                    pipeline.load_lora_weights(weights_file, adapter_name=adapter_name)
                    pipeline.fuse_lora(lora_scale=strength, adapter_name=adapter_name)

        if self.pipeline is None and self.inpaint_pipeline is None:
            raise RuntimeError("No pipeline loaded to add LoRAs to")

    def pipeToConfiguredDevice(self) -> None:
        """Move pipeline(s) to the configured compute device."""
        if self.pipeline is not None:
            self.pipeline.to(self.accelerator)
        if self.inpaint_pipeline is not None:
            self.inpaint_pipeline.to(self.accelerator)

    def loadCustomVAE(self, vae_checkpoint: Union[str, Path]) -> None:
        """Load a custom VAE into the pipeline."""
        custom_vae = load_custom_vae(vae_checkpoint)
        if self.pipeline is not None:
            self.pipeline.vae = custom_vae
        elif self.inpaint_pipeline is not None:
            self.inpaint_pipeline.vae = custom_vae
        else:
            raise RuntimeError("No pipeline loaded to set custom VAE")

