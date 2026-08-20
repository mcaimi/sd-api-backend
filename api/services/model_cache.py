#!/usr/bin/env python
"""
Model Cache Manager

Shared cache for loaded SD pipelines to avoid repeated model loading.
"""

import asyncio
import logging
from typing import Dict, Optional, Tuple
from pathlib import Path

from libs.stablediffusion.sd15 import SD15PipelineGenerator
from libs.stablediffusion.sdxl import SDXLPipelineGenerator

logger = logging.getLogger(__name__)


class ModelCacheManager:
    """
    Unified cache manager for SD pipelines.

    Maintains an LRU cache of loaded models to avoid repeated loading overhead.
    Thread-safe for concurrent access.
    """

    def __init__(self, max_models: int = 2):
        """
        Initialize cache manager.

        Args:
            max_models: Maximum number of models to cache
        """
        self.max_models = max_models
        self._cache: Dict[str, Tuple] = {}  # key -> (pipeline, last_used_time)
        self._lock = asyncio.Lock()
        self._access_order: list = []  # For LRU tracking

    def _make_cache_key(self, model_type: str, model_path: str, pipeline_type: str) -> str:
        """Generate cache key from model parameters."""
        return f"{model_type}:{Path(model_path).name}:{pipeline_type}"

    async def get_pipeline(
        self,
        model_type: str,
        model_path: str,
        pipeline_type: str = "txt2img"
    ):
        """
        Get a pipeline from cache or load it.

        Args:
            model_type: "sd15" or "sdxl"
            model_path: Path to model checkpoint
            pipeline_type: "txt2img" or "inpaint"

        Returns:
            Pipeline generator instance
        """
        cache_key = self._make_cache_key(model_type, model_path, pipeline_type)

        async with self._lock:
            # Check cache
            if cache_key in self._cache:
                pipeline, _ = self._cache[cache_key]
                # Update access order
                if cache_key in self._access_order:
                    self._access_order.remove(cache_key)
                self._access_order.append(cache_key)
                logger.info(f"Cache hit for {cache_key}")
                return pipeline

            # Load new pipeline
            logger.info(f"Cache miss for {cache_key}, loading model...")
            pipeline = await self._load_pipeline(model_type, model_path, pipeline_type)

            # Evict oldest if at capacity
            if len(self._cache) >= self.max_models:
                oldest_key = self._access_order.pop(0)
                del self._cache[oldest_key]
                logger.info(f"Evicted {oldest_key} from cache")

            # Add to cache
            self._cache[cache_key] = (pipeline, asyncio.get_event_loop().time())
            self._access_order.append(cache_key)

            return pipeline

    async def _load_pipeline(self, model_type: str, model_path: str, pipeline_type: str):
        """Load a pipeline in a thread pool executor."""
        loop = asyncio.get_event_loop()

        def _load():
            if model_type == "sd15":
                pipeline_gen = SD15PipelineGenerator(model_checkpoint=model_path)
                if pipeline_type == "txt2img":
                    pipeline_gen.loadSDPipeline()
                else:  # inpaint
                    pipeline_gen.loadSDInpaintPipeline()
            elif model_type == "sdxl":
                pipeline_gen = SDXLPipelineGenerator(model_checkpoint=model_path)
                if pipeline_type == "txt2img":
                    pipeline_gen.loadSDXLPipeline()
                else:  # inpaint
                    pipeline_gen.loadSDXLInpaintPipeline()
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            pipeline_gen.pipeToConfiguredDevice()
            return pipeline_gen

        # Run in executor to avoid blocking
        return await loop.run_in_executor(None, _load)

    async def evict(self, model_type: str, model_path: str, pipeline_type: str = "txt2img") -> bool:
        """Remove a specific pipeline from the cache. Returns True if evicted, False if not found."""
        cache_key = self._make_cache_key(model_type, model_path, pipeline_type)
        async with self._lock:
            if cache_key not in self._cache:
                return False
            del self._cache[cache_key]
            self._access_order.remove(cache_key)
            logger.info(f"Evicted {cache_key} from cache")
            return True

    async def clear_cache(self):
        """Clear all cached models."""
        async with self._lock:
            self._cache.clear()
            self._access_order.clear()
        logger.info("Model cache cleared")

    def get_stats(self) -> dict:
        """Get cache statistics."""
        return {
            "cached_models": len(self._cache),
            "max_models": self.max_models,
            "cache_keys": list(self._cache.keys()),
        }


# Singleton instance
_cache_manager: Optional[ModelCacheManager] = None


def get_cache_manager() -> ModelCacheManager:
    """Get the singleton cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = ModelCacheManager(max_models=2)
    return _cache_manager


def set_cache_manager(manager: ModelCacheManager):
    """Set a custom cache manager (for testing)."""
    global _cache_manager
    _cache_manager = manager
