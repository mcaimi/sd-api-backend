#!/usr/bin/env python
"""
Stable Diffusion 1.5 Pipeline Generator

Provides SD1.5-specific implementation of the base pipeline generator.
"""

import logging
from typing import Dict, Tuple, Optional

from diffusers import StableDiffusionPipeline, StableDiffusionInpaintPipeline

from libs.stablediffusion.base import (
    BasePipelineGenerator,
    load_custom_vae,
    load_custom_unet,
    gen_noise,
    format_metadata,
)

logger = logging.getLogger(__name__)


# Re-export for backward compatibility
__all__ = [
    "SD15PipelineGenerator",
    "load_custom_vae",
    "load_custom_unet",
    "gen_noise",
    "format_metadata",
]


class SD15PipelineGenerator(BasePipelineGenerator):
    """SD1.5 Pipeline Generator for text-to-image and inpainting."""

    PIPELINE_CLASS = StableDiffusionPipeline
    INPAINT_PIPELINE_CLASS = StableDiffusionInpaintPipeline
    DEFAULT_WIDTH = 512
    DEFAULT_HEIGHT = 512
    MODEL_TYPE = "sd15"

    @staticmethod
    def get_resolutions() -> Dict[str, Tuple[int, int]]:
        """Return supported SD1.5 resolutions."""
        return {
            "512x512": (512, 512),
            "512x768": (512, 768),
            "768x512": (768, 512),
        }

    # Alias for backward compatibility
    @staticmethod
    def get_sd15_resolutions() -> Dict[str, Tuple[int, int]]:
        """Return supported SD1.5 resolutions (legacy alias)."""
        return SD15PipelineGenerator.get_resolutions()

    def loadPipeline(self) -> None:
        """Load the SD1.5 generation pipeline."""
        self._init_device()
        logger.info("Loading SD1.5 checkpoint: %s", self.model_checkpoint)
        self.pipeline = StableDiffusionPipeline.from_single_file(
            self.model_checkpoint, torch_dtype=self.dtype, use_safetensors=True
        )

    # Alias for backward compatibility
    def loadSDPipeline(self) -> None:
        """Load the SD1.5 pipeline (legacy alias)."""
        self.loadPipeline()

    def loadInpaintPipeline(self) -> None:
        """Load the SD1.5 inpainting pipeline."""
        self._init_device()
        logger.info("Loading SD1.5 inpaint checkpoint: %s", self.model_checkpoint)
        self.inpaint_pipeline = StableDiffusionInpaintPipeline.from_single_file(
            self.model_checkpoint, torch_dtype=self.dtype, use_safetensors=True
        )

    # Alias for backward compatibility
    def loadSDInpaintPipeline(self) -> None:
        """Load the SD1.5 inpainting pipeline (legacy alias)."""
        self.loadInpaintPipeline()

    # Legacy property aliases
    @property
    def sd_pipeline(self) -> Optional[StableDiffusionPipeline]:
        """Legacy alias for pipeline."""
        return self.pipeline

    @sd_pipeline.setter
    def sd_pipeline(self, value: Optional[StableDiffusionPipeline]) -> None:
        self.pipeline = value
