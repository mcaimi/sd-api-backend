"""Concurrency tests for the model cache leases (no real pipelines are loaded)."""

import asyncio

import pytest

from api.services.model_cache import CLEAN_STATE, DIRTY_STATE, ModelCacheManager


class FakeCache(ModelCacheManager):
    """ModelCacheManager whose loads return a placeholder object."""

    def __init__(self, max_models=2, fail_loads=0):
        super().__init__(max_models=max_models)
        self.loads = []
        self.fail_loads = fail_loads

    async def _load_pipeline(self, model_type, model_path, pipeline_type):
        await asyncio.sleep(0.01)
        if self.fail_loads:
            self.fail_loads -= 1
            raise RuntimeError("load failed")
        self.loads.append((model_type, model_path, pipeline_type))
        return object()


class ConcurrencyProbe:
    """Tracks how many holders are inside a lease at the same time."""

    def __init__(self):
        self.active = 0
        self.peak = 0

    async def hold(self, cache, model_type, model_path, seconds=0.05):
        async with cache.acquire(model_type, model_path):
            self.active += 1
            self.peak = max(self.peak, self.active)
            await asyncio.sleep(seconds)
            self.active -= 1


def test_same_pipeline_is_leased_exclusively():
    async def main():
        cache, probe = FakeCache(), ConcurrencyProbe()
        await asyncio.gather(*(probe.hold(cache, "sdxl", "a.safetensors") for _ in range(3)))
        return cache, probe

    cache, probe = asyncio.run(main())
    assert probe.peak == 1
    assert len(cache.loads) == 1


def test_different_pipelines_run_concurrently():
    async def main():
        cache, probe = FakeCache(max_models=2), ConcurrencyProbe()
        await asyncio.gather(
            probe.hold(cache, "sdxl", "a.safetensors"),
            probe.hold(cache, "sd15", "b.safetensors"),
        )
        return probe

    assert asyncio.run(main()).peak == 2


def test_flux_pipelines_share_one_lock():
    async def main():
        cache, probe = FakeCache(max_models=2), ConcurrencyProbe()
        await asyncio.gather(
            probe.hold(cache, "flux", "a.safetensors"),
            probe.hold(cache, "flux", "b.safetensors"),
        )
        return probe

    assert asyncio.run(main()).peak == 1


def test_leased_pipeline_is_not_evicted():
    async def main():
        cache = FakeCache(max_models=1)
        order = []

        async def first():
            async with cache.acquire("sdxl", "a.safetensors"):
                order.append("a-start")
                await asyncio.sleep(0.05)
                assert "sdxl:a.safetensors:txt2img" in cache.get_stats()["cache_keys"]
                order.append("a-end")

        async def second():
            await asyncio.sleep(0.01)  # let first() take the only slot
            async with cache.acquire("sd15", "b.safetensors"):
                order.append("b-start")

        await asyncio.gather(first(), second())
        return cache, order

    cache, order = asyncio.run(main())
    assert order == ["a-start", "a-end", "b-start"]
    assert cache.get_stats()["cache_keys"] == ["sd15:b.safetensors:txt2img"]


def test_idle_lru_pipeline_is_evicted():
    async def main():
        cache = FakeCache(max_models=2)
        for name in ("a", "b", "a", "c"):
            async with cache.acquire("sdxl", f"{name}.safetensors"):
                pass
        return cache

    cache = asyncio.run(main())
    assert cache.get_stats()["cache_keys"] == ["sdxl:a.safetensors:txt2img", "sdxl:c.safetensors:txt2img"]


@pytest.mark.parametrize("left_state", [(((("x.safetensors", 1.0),), None)), DIRTY_STATE])
def test_pipeline_with_other_state_is_reloaded(left_state):
    async def main():
        cache = FakeCache()
        async with cache.acquire("sdxl", "a.safetensors") as entry:
            entry.state = left_state
        async with cache.acquire("sdxl", "a.safetensors", state=CLEAN_STATE) as entry:
            assert entry.state == CLEAN_STATE
        return cache

    assert len(asyncio.run(main()).loads) == 2


def test_pipeline_with_requested_state_is_reused():
    async def main():
        cache = FakeCache()
        state = ((("x.safetensors", 0.8),), None)
        for _ in range(2):
            async with cache.acquire("sdxl", "a.safetensors", state=state) as entry:
                entry.state = state
        return cache

    assert len(asyncio.run(main()).loads) == 1


def test_failed_load_releases_its_slot():
    async def main():
        cache = FakeCache(max_models=1, fail_loads=1)
        with pytest.raises(RuntimeError):
            async with cache.acquire("sdxl", "a.safetensors"):
                pass
        async with cache.acquire("sdxl", "b.safetensors"):
            pass
        return cache

    stats = asyncio.run(main()).get_stats()
    assert stats["cache_keys"] == ["sdxl:b.safetensors:txt2img"]
    assert stats["in_use"] == 0
