#!/usr/bin/env python
"""
Illustrious XL Pipeline Generator

Provides Illustrious-specific implementation of the base pipeline generator.
Illustrious XL is architecturally SDXL-based but supports higher native
resolutions (up to 1536x1536).
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
    "IllustriousPipelineGenerator",
    "load_custom_vae",
    "load_custom_unet",
    "gen_noise",
    "format_metadata",
]


class IllustriousPipelineGenerator(BasePipelineGenerator):
    """Illustrious XL Pipeline Generator for text-to-image and inpainting."""

    PIPELINE_CLASS = StableDiffusionXLPipeline
    INPAINT_PIPELINE_CLASS = StableDiffusionXLInpaintPipeline
    DEFAULT_WIDTH = 1024
    DEFAULT_HEIGHT = 1024
    MODEL_TYPE = "illustrious"

    @staticmethod
    def get_resolutions() -> Dict[str, Tuple[int, int]]:
        """Return supported Illustrious XL resolutions."""
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
            "1280x1280": (1280, 1280),
            "1536x1536": (1536, 1536),
        }

    # Alias for backward compatibility
    @staticmethod
    def get_illustrious_resolutions() -> Dict[str, Tuple[int, int]]:
        """Return supported Illustrious XL resolutions (legacy alias)."""
        return IllustriousPipelineGenerator.get_resolutions()

    def loadPipeline(self) -> None:
        """Load the Illustrious XL generation pipeline."""
        self._init_device()
        logger.info("Loading Illustrious XL checkpoint: %s", self.model_checkpoint)
        self.pipeline = StableDiffusionXLPipeline.from_single_file(
            self.model_checkpoint, torch_dtype=self.dtype, use_safetensors=True
        )

    # Alias for backward compatibility
    def loadIllustriousPipeline(self) -> None:
        """Load the Illustrious XL pipeline (legacy alias)."""
        self.loadPipeline()

    def loadInpaintPipeline(self) -> None:
        """Load the Illustrious XL inpainting pipeline."""
        self._init_device()
        logger.info("Loading Illustrious XL inpaint checkpoint: %s", self.model_checkpoint)
        self.inpaint_pipeline = StableDiffusionXLInpaintPipeline.from_single_file(
            self.model_checkpoint, torch_dtype=self.dtype, use_safetensors=True
        )

    # Alias for backward compatibility
    def loadIllustriousInpaintPipeline(self) -> None:
        """Load the Illustrious XL inpainting pipeline (legacy alias)."""
        self.loadInpaintPipeline()

    # Legacy property aliases
    @property
    def illustrious_pipeline(self) -> Optional[StableDiffusionXLPipeline]:
        """Legacy alias for pipeline."""
        return self.pipeline

    @illustrious_pipeline.setter
    def illustrious_pipeline(self, value: Optional[StableDiffusionXLPipeline]) -> None:
        self.pipeline = value
