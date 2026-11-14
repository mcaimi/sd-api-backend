"""LoRA/VAE state handling of the generation service's pipeline leases."""

import asyncio

import pytest

from api.services import generation, model_cache
from api.services.model_cache import DIRTY_STATE

from tests.test_model_cache import FakeCache


class FakeGenerator:
    """Records LoRA/VAE application; optionally fails the first add_loras."""

    instances = []

    def __init__(self, fail_first_lora=False):
        self.loras = []
        self.vae = None
        self.fail_first_lora = fail_first_lora
        FakeGenerator.instances.append(self)

    def add_loras(self, specs):
        if self.fail_first_lora:
            raise RuntimeError("adapter clash")
        self.loras.extend(str(spec.path) for spec in specs)

    def load_custom_vae(self, path):
        self.vae = path


class GeneratorCache(FakeCache):
    def __init__(self, fail_first_lora=False, **kwargs):
        super().__init__(**kwargs)
        self.fail_first_lora = fail_first_lora

    async def _load_pipeline(self, model_type, model_path, pipeline_type):
        self.loads.append((model_type, model_path, pipeline_type))
        generator = FakeGenerator(fail_first_lora=self.fail_first_lora)
        self.fail_first_lora = False
        return generator


@pytest.fixture
def cache():
    FakeGenerator.instances = []
    yield
    model_cache.set_cache_manager(None)


LORA = [{"name": "x.safetensors", "path": "/loras/x.safetensors", "strength": 0.7}]


async def lease(loras=None, custom_vae=None):
    async with generation._leased_pipeline("sdxl", "a.safetensors", "txt2img", loras, custom_vae) as (gen, meta):
        return gen, meta


def test_loras_and_vae_are_applied_once_per_state(cache):
    manager = GeneratorCache()
    model_cache.set_cache_manager(manager)

    async def main():
        first, meta = await lease(LORA, "/vae/v.safetensors")
        second, _ = await lease(LORA, "/vae/v.safetensors")
        return first, second, meta

    first, second, meta = asyncio.run(main())
    assert first is second
    assert first.loras == ["/loras/x.safetensors"]
    assert first.vae == "/vae/v.safetensors"
    assert meta["lora_0"]["merge_strength"] == 0.7
    assert len(manager.loads) == 1


def test_loras_do_not_leak_into_later_requests(cache):
    manager = GeneratorCache()
    model_cache.set_cache_manager(manager)

    async def main():
        await lease(LORA)
        clean, _ = await lease()
        return clean

    clean = asyncio.run(main())
    assert clean.loras == []
    assert len(manager.loads) == 2


def test_lora_failure_reloads_and_retries(cache):
    manager = GeneratorCache(fail_first_lora=True)
    model_cache.set_cache_manager(manager)

    gen, _ = asyncio.run(lease(LORA))
    assert len(manager.loads) == 2
    assert gen.loras == ["/loras/x.safetensors"]


def test_inference_failure_keeps_applied_state(cache):
    manager = GeneratorCache()
    model_cache.set_cache_manager(manager)

    async def main():
        with pytest.raises(ValueError):
            async with generation._leased_pipeline("sdxl", "a.safetensors", "txt2img", LORA, None):
                raise ValueError("inference blew up")
        entry = manager._cache["sdxl:a.safetensors:txt2img"]
        assert entry.state is not DIRTY_STATE
        await lease(LORA)

    asyncio.run(main())
    assert len(manager.loads) == 1
