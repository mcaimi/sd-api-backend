#!/usr/bin/env python
"""
Model Cache Manager

Shared cache for loaded SD pipelines to avoid repeated model loading.
"""

import asyncio
import logging
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, Optional, Set
from pathlib import Path

from metastable.pipelines import create_generator

from api.config import get_app_config

logger = logging.getLogger(__name__)

# State of a freshly loaded pipeline: no LoRAs fused, stock VAE.
# States are (((lora_path, strength), ...), custom_vae_path_or_None).
CLEAN_STATE = ((), None)

# Marks a pipeline whose LoRA/VAE application failed or was interrupted;
# it never equals a requested state, so the next lease reloads it.
DIRTY_STATE = object()

# All Flux pipelines share one set of text encoders/VAE (metastable's
# load_flux_shared_components), so they can't run concurrently either.
FLUX_LOCK_GROUP = "flux"


@dataclass
class CachedPipeline:
    """A cached pipeline generator and the LoRA/VAE state applied to it."""

    pipeline: Any
    state: Any = CLEAN_STATE


class ModelCacheManager:
    """
    Unified cache manager for SD pipelines.

    Maintains an LRU cache of loaded models to avoid repeated loading overhead.
    Pipelines are handed out as exclusive leases via acquire(): a pipeline is
    used by one job at a time, and is never evicted while leased.
    """

    def __init__(self, max_models: int = 2):
        """
        Initialize cache manager.

        Args:
            max_models: Maximum number of models to cache
        """
        self.max_models = max_models
        self._cache: "OrderedDict[str, CachedPipeline]" = OrderedDict()  # LRU order, oldest first
        self._in_use: Set[str] = set()
        self._loading = 0  # cache slots reserved by in-flight loads
        self._cond = asyncio.Condition()  # guards the fields above
        self._group_locks: Dict[str, asyncio.Lock] = {}

    def _make_cache_key(self, model_type: str, model_path: str, pipeline_type: str) -> str:
        """Generate cache key from model parameters."""
        return f"{model_type}:{Path(model_path).name}:{pipeline_type}"

    @asynccontextmanager
    async def acquire(
        self,
        model_type: str,
        model_path: str,
        pipeline_type: str = "txt2img",
        state: Any = CLEAN_STATE,
    ) -> AsyncIterator[CachedPipeline]:
        """
        Lease a pipeline from the cache, loading it if needed.

        The lease is exclusive for its whole duration, so the holder can
        apply LoRAs/VAE and run inference without other jobs interfering.
        The yielded entry's state is either CLEAN_STATE or the requested
        state; a pipeline carrying any other state is reloaded first.
        Holders must set entry.state to whatever they leave applied.

        Args:
            model_type: "sd15", "sdxl", "illustrious", or "flux"
            model_path: Path to model checkpoint
            pipeline_type: "txt2img" or "inpaint"
            state: The LoRA/VAE state the holder is going to apply

        Yields:
            CachedPipeline entry
        """
        cache_key = self._make_cache_key(model_type, model_path, pipeline_type)
        group = FLUX_LOCK_GROUP if model_type == "flux" else cache_key
        lock = self._group_locks.setdefault(group, asyncio.Lock())

        async with lock:
            entry = await self._checkout(cache_key, model_type, model_path, pipeline_type, state)
            try:
                yield entry
            finally:
                async with self._cond:
                    self._in_use.discard(cache_key)
                    self._cond.notify_all()

    async def _checkout(
        self, cache_key: str, model_type: str, model_path: str, pipeline_type: str, state: Any
    ) -> CachedPipeline:
        """Return a cached entry for cache_key marked in use, loading it if needed."""
        async with self._cond:
            entry = self._cache.get(cache_key)
            if entry is not None and entry.state not in (CLEAN_STATE, state):
                logger.info(f"Applied LoRA/VAE state of {cache_key} differs from request, reloading")
                del self._cache[cache_key]
                entry = None

            if entry is not None:
                self._cache.move_to_end(cache_key)
                self._in_use.add(cache_key)
                logger.info(f"Cache hit for {cache_key}")
                return entry

            await self._reserve_slot()
            self._loading += 1
            self._in_use.add(cache_key)

        logger.info(f"Cache miss for {cache_key}, loading model...")
        try:
            pipeline = await self._load_pipeline(model_type, model_path, pipeline_type)
        except BaseException:
            async with self._cond:
                self._loading -= 1
                self._in_use.discard(cache_key)
                self._cond.notify_all()
            raise

        entry = CachedPipeline(pipeline=pipeline)
        async with self._cond:
            self._loading -= 1
            self._cache[cache_key] = entry
        return entry

    async def _reserve_slot(self) -> None:
        """
        Wait until a cache slot is free, evicting the least recently used
        idle pipeline if needed. Must be called with self._cond held.
        """
        while len(self._cache) + self._loading >= self.max_models:
            victim = next((key for key in self._cache if key not in self._in_use), None)
            if victim is None:
                # Every cached pipeline is leased: wait for one to be released
                await self._cond.wait()
                continue
            del self._cache[victim]
            logger.info(f"Evicted {victim} from cache")

    async def _load_pipeline(self, model_type: str, model_path: str, pipeline_type: str):
        """Load a pipeline in a thread pool executor."""
        loop = asyncio.get_event_loop()

        def _load():
            extra_kwargs = {}
            if model_type == "flux":
                extra_kwargs["components_dir"] = get_app_config().flux_components_path
            pipeline_gen = create_generator(model_type, model_path, **extra_kwargs)
            if pipeline_type == "txt2img":
                pipeline_gen.load_pipeline()
            else:  # inpaint
                pipeline_gen.load_inpaint_pipeline()

            pipeline_gen.to_device()
            return pipeline_gen

        # Run in executor to avoid blocking
        return await loop.run_in_executor(None, _load)

    async def clear_cache(self):
        """Clear all cached models that aren't currently leased."""
        async with self._cond:
            for key in [key for key in self._cache if key not in self._in_use]:
                del self._cache[key]
        logger.info("Model cache cleared")

    def get_stats(self) -> dict:
        """Get cache statistics."""
        return {
            "cached_models": len(self._cache),
            "max_models": self.max_models,
            "in_use": len(self._in_use),
            "cache_keys": list(self._cache.keys()),
        }


# Singleton instance
_cache_manager: Optional[ModelCacheManager] = None


def get_cache_manager() -> ModelCacheManager:
    """Get the singleton cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = ModelCacheManager(max_models=get_app_config().model_cache_max_models)
    return _cache_manager


def set_cache_manager(manager: ModelCacheManager):
    """Set a custom cache manager (for testing)."""
    global _cache_manager
    _cache_manager = manager
