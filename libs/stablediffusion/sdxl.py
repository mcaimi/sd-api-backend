#!/usr/bin/env python
"""
Stable Diffusion XL Pipeline Generator

Provides SDXL-specific implementation of the base pipeline generator.
"""

import logging
from typing import Dict, Tuple, Optional

from diffusers import StableDiffusionXLPipeline, StableDiffusionXLInpaintPipeline

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
    "SDXLPipelineGenerator",
    "load_custom_vae",
    "load_custom_unet",
    "gen_noise",
    "format_metadata",
]


class SDXLPipelineGenerator(BasePipelineGenerator):
    """SDXL Pipeline Generator for text-to-image and inpainting."""

    PIPELINE_CLASS = StableDiffusionXLPipeline
    INPAINT_PIPELINE_CLASS = StableDiffusionXLInpaintPipeline
    DEFAULT_WIDTH = 1024
    DEFAULT_HEIGHT = 1024
    MODEL_TYPE = "sdxl"

    @staticmethod
    def get_resolutions() -> Dict[str, Tuple[int, int]]:
        """Return supported SDXL resolutions."""
        return {
            "1024x1024": (1024, 1024),
            "1152x896": (1152, 896),
            "896x1152": (896, 1152),
            "1216x832": (1216, 832),
            "832x1216": (832, 1216),
            "1344x768": (1344, 768),
            "768x1344": (768, 1344),
            "1536x640": (1536, 640),
            "640x1536": (640, 1536),
        }

    # Alias for backward compatibility
    @staticmethod
    def get_sdxl_resolutions() -> Dict[str, Tuple[int, int]]:
        """Return supported SDXL resolutions (legacy alias)."""
        return SDXLPipelineGenerator.get_resolutions()

    def loadPipeline(self) -> None:
        """Load the SDXL generation pipeline."""
        self._init_device()
        logger.info("Loading SDXL checkpoint: %s", self.model_checkpoint)
        self.pipeline = StableDiffusionXLPipeline.from_single_file(
            self.model_checkpoint, torch_dtype=self.dtype, use_safetensors=True
        )

    # Alias for backward compatibility
    def loadSDXLPipeline(self) -> None:
        """Load the SDXL pipeline (legacy alias)."""
        self.loadPipeline()

    def loadInpaintPipeline(self) -> None:
        """Load the SDXL inpainting pipeline."""
        self._init_device()
        logger.info("Loading SDXL inpaint checkpoint: %s", self.model_checkpoint)
        self.inpaint_pipeline = StableDiffusionXLInpaintPipeline.from_single_file(
            self.model_checkpoint, torch_dtype=self.dtype, use_safetensors=True
        )

    # Alias for backward compatibility
    def loadSDXLInpaintPipeline(self) -> None:
        """Load the SDXL inpainting pipeline (legacy alias)."""
        self.loadInpaintPipeline()

    # Legacy property aliases
    @property
    def sdxl_pipeline(self) -> Optional[StableDiffusionXLPipeline]:
        """Legacy alias for pipeline."""
        return self.pipeline

    @sdxl_pipeline.setter
    def sdxl_pipeline(self, value: Optional[StableDiffusionXLPipeline]) -> None:
        self.pipeline = value
