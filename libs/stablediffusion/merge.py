#!/usr/bin/env python
"""
Unified Model Merging Library for Stable Diffusion

Supports SD1.5 and SDXL models with multiple merge algorithms:
    - Linear interpolation
    - Spherical linear interpolation (SLERP)
    - Additive merging
    - Subtractive merging

Features:
    - Simple 2-model merging
    - Compatibility checks
    - Batch merging of multiple models
    - Multi-step recipe merging
    - Checkpoint caching for performance
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union, Callable, Any
import logging
import time

import torch

from libs.shared.utils import get_gpu
from libs.globals.vars import MergeMethod
from libs.shared.exceptions import MergeError
from libs.stablediffusion.mergeops import (
    MergeConfig,
    load_checkpoint_dict,
    save_checkpoint,
    merge_checkpoints,
)

logger = logging.getLogger(__name__)

# Re-export for convenience
__all__ = [
    "MergeConfig",
    "MergeMethod",
    "MergePipeline",
    "SD15MergePipeline",
    "SDXLMergePipeline",
]


@dataclass
class ModelArchitectureInfo:
    """Information about a model architecture."""
    name: str
    base_resolution: int
    supported_resolutions: List[str] = field(default_factory=list)


# Architecture definitions
SD15_ARCHITECTURE = ModelArchitectureInfo(
    name="sd15",
    base_resolution=512,
    supported_resolutions=["512x512", "512x768", "768x512"],
)

SDXL_ARCHITECTURE = ModelArchitectureInfo(
    name="sdxl",
    base_resolution=1024,
    supported_resolutions=[
        "1024x1024", "1152x896", "896x1152", "1216x832",
        "832x1216", "1344x768", "768x1344", "1536x640", "640x1536",
    ],
)


class MergePipeline:
    """
    Unified merging pipeline for Stable Diffusion models.
    
    Supports both SD1.5 and SDXL architectures with checkpoint caching,
    recipe-based merging, and batch processing.
    """

    def __init__(
        self,
        architecture: ModelArchitectureInfo,
        device: str = "auto",
    ):
        """
        Initialize the merge pipeline.
        
        Args:
            architecture: Model architecture information
            device: Compute device ("auto", "cpu", "cuda", "mps")
        """
        self.architecture = architecture
        self.device = device if device != "auto" else get_gpu()[0]
        self.loaded_checkpoints: Dict[str, Dict[str, Any]] = {}

    def load_and_cache_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        cache_key: Optional[str] = None,
    ) -> str:
        """
        Load and cache a checkpoint for future merging operations.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            cache_key: Optional custom key for the cache
            
        Returns:
            The cache key used to store the checkpoint
        """
        if isinstance(checkpoint_path, str):
            checkpoint_path = Path(checkpoint_path)

        cache_key = cache_key or checkpoint_path.stem

        if cache_key not in self.loaded_checkpoints:
            weights, metadata = load_checkpoint_dict(checkpoint_path, self.device)
            self.loaded_checkpoints[cache_key] = {
                "weights": weights,
                "metadata": metadata,
                "path": checkpoint_path,
            }
            logger.info("Cached checkpoint: %s", cache_key)

        return cache_key

    def create_merge_recipe(
        self,
        base_model: str,
        merge_steps: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Create a merge recipe for complex multi-step merging.
        
        Args:
            base_model: Path to the base model checkpoint
            merge_steps: List of merge step configurations
            
        Returns:
            Recipe dictionary
        """
        return {
            "base_model": base_model,
            "steps": merge_steps,
            "created_at": time.time(),
            "architecture": self.architecture.name,
        }

    def execute_merge_recipe(
        self,
        recipe: Dict[str, Any],
        output_path: Union[str, Path],
    ) -> bool:
        """
        Execute a complex merge recipe with multiple steps.
        
        Args:
            recipe: Recipe dictionary from create_merge_recipe
            output_path: Path to save the final merged model
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load base model
            base_key = self.load_and_cache_checkpoint(recipe["base_model"])
            current_weights = self.loaded_checkpoints[base_key]["weights"].copy()

            recipe_metadata = {
                "recipe": recipe,
                "execution_timestamp": time.time(),
                "steps_executed": [],
                "architecture": self.architecture.name,
            }

            # Execute each merge step
            for i, step in enumerate(recipe["steps"]):
                logger.info("Executing merge step %d/%d", i + 1, len(recipe["steps"]))

                target_key = self.load_and_cache_checkpoint(step["target_model"])
                target_weights = self.loaded_checkpoints[target_key]["weights"]

                # Create merge config
                progress_callback = None
                if step.get("show_progress", False):
                    progress_callback = lambda p: logger.debug("Progress: %.1f%%", p * 100)

                config = MergeConfig(
                    method=MergeMethod(step.get("method", "linear")),
                    alpha=step.get("alpha", 0.5),
                    device=self.device,
                    progress_callback=progress_callback,
                )

                # Perform merge
                base_reference = None
                if config.method in [MergeMethod.ADDITIVE, MergeMethod.SUBTRACT]:
                    base_reference = self.loaded_checkpoints[base_key]["weights"]

                current_weights = merge_checkpoints(
                    current_weights, target_weights, config, base_reference
                )

                # Record step execution
                recipe_metadata["steps_executed"].append({
                    "step_index": i,
                    "target_model": str(step["target_model"]),
                    "method": step.get("method", "linear"),
                    "alpha": step.get("alpha", 0.5),
                    "completed_at": time.time(),
                })

            # Save final result
            return save_checkpoint(current_weights, output_path, recipe_metadata)

        except MergeError:
            # Re-raise MergeErrors as-is
            raise
        except Exception as e:
            logger.error("Failed to execute merge recipe: %s", e)
            raise MergeError(f"Recipe execution failed: {e}") from e

    def merge_for_pipeline_generator(
        self,
        base_model: Union[str, Path],
        target_model: Union[str, Path],
        config: MergeConfig,
        output_path: Union[str, Path],
    ) -> Optional[str]:
        """
        Merge models specifically for use with pipeline generators.
        
        Args:
            base_model: Path to base model
            target_model: Path to target model
            config: Merge configuration
            output_path: Path to save merged model
            
        Returns:
            Path to saved model if successful, None otherwise
        """
        try:
            # Load checkpoints
            base_weights, base_metadata = load_checkpoint_dict(
                base_model, config.device
            )
            target_weights, target_metadata = load_checkpoint_dict(
                target_model, config.device
            )

            # Perform merge
            logger.info(
                "Merging %s with %s using %s",
                base_model, target_model, config.method.value
            )
            merged_weights = merge_checkpoints(base_weights, target_weights, config)

            # Create comprehensive metadata
            merged_metadata = {
                "merge_info": {
                    "base_model": str(base_model),
                    "target_model": str(target_model),
                    "method": config.method.value,
                    "alpha": config.alpha,
                    "timestamp": time.time(),
                },
                "model_info": {
                    "architecture": self.architecture.name,
                    "base_resolution": self.architecture.base_resolution,
                    "supported_resolutions": self.architecture.supported_resolutions,
                },
                "base_metadata": base_metadata,
                "target_metadata": target_metadata,
            }

            # Save merged model
            if save_checkpoint(merged_weights, output_path, merged_metadata):
                logger.info("Merged model saved: %s", output_path)
                return str(output_path)
            else:
                logger.error("Failed to save merged model")
                raise MergeError(f"Failed to save merged model to {output_path}")

        except MergeError:
            # Re-raise MergeErrors as-is
            raise
        except Exception as e:
            logger.error("Merge failed: %s", e)
            raise MergeError(f"Model merge operation failed: {e}") from e

    def clear_cache(self) -> None:
        """Clear cached checkpoints to free memory."""
        self.loaded_checkpoints.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Checkpoint cache cleared")

    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cached checkpoints."""
        return {
            "cached_models": list(self.loaded_checkpoints.keys()),
            "cache_size": len(self.loaded_checkpoints),
            "device": self.device,
            "architecture": self.architecture.name,
        }


# Convenience classes for backward compatibility
class SD15MergePipeline(MergePipeline):
    """SD1.5 specific merge pipeline (convenience class)."""
    
    def __init__(self, device: str = "auto"):
        super().__init__(architecture=SD15_ARCHITECTURE, device=device)


class SDXLMergePipeline(MergePipeline):
    """SDXL specific merge pipeline (convenience class)."""
    
    def __init__(self, device: str = "auto"):
        super().__init__(architecture=SDXL_ARCHITECTURE, device=device)

