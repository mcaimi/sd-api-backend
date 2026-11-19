"""Merge router: supported model types, header-level compatibility checks, output dtype."""

from pathlib import Path

import pytest
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from safetensors.torch import save_file

from api.routers import merge as merge_router

from tests.conftest import make_config

FLUX_ORIGINAL = ("double_blocks.0.img_attn.qkv.weight", "single_blocks.0.linear1.weight")
FLUX_DIFFUSERS = (
    "transformer_blocks.0.attn.to_q.weight",
    "single_transformer_blocks.0.attn.to_q.weight",
)
ZIMAGE_ORIGINAL = ("layers.0.attention.qkv.weight", "x_embedder.weight")


class FakeJobQueue:
    def __init__(self):
        self.jobs = []

    async def submit_job(self, job_type, params, callback, **kwargs):
        self.jobs.append({"job_type": job_type, "params": params})
        return f"job-{len(self.jobs)}"


@pytest.fixture
def env(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    monkeypatch.setattr(merge_router, "get_app_config", lambda: config)
    queue = FakeJobQueue()
    monkeypatch.setattr(merge_router, "get_job_queue", lambda: queue)
    app = FastAPI()
    app.include_router(merge_router.router, prefix="/api/v1/merge")
    return TestClient(app), config, queue


def checkpoint(config, model_type, name, keys, dtype=torch.bfloat16, prefix="", shape=(2, 2)):
    path = Path(config.get_model_paths(model_type)["checkpoints"]) / name
    save_file({prefix + key: torch.zeros(shape, dtype=dtype) for key in keys}, str(path))
    return path


def merge(client, model_type, **body):
    body = {"model_type": model_type, "base_model": "a.safetensors",
            "target_model": "b.safetensors", "output_name": "merged", **body}
    return client.post("/api/v1/merge/merge", json=body)


@pytest.mark.parametrize("model_type, keys", [("flux", FLUX_ORIGINAL), ("zimage", ZIMAGE_ORIGINAL)])
def test_transformer_merges_are_queued(env, model_type, keys):
    client, config, queue = env
    checkpoint(config, model_type, "a.safetensors", keys)
    # All-in-one target (prefixed transformer) merges with a bare base
    checkpoint(config, model_type, "b.safetensors", keys, prefix="model.diffusion_model.")

    response = merge(client, model_type, output_dtype="bfloat16")
    assert response.status_code == 200, response.json()

    params = queue.jobs[0]["params"]
    assert params["model_type"] == model_type
    assert params["output_dtype"] == "bfloat16"
    assert params["output_path"].endswith("merged.safetensors")


def test_mixed_key_layouts_are_rejected_before_queueing(env):
    client, config, queue = env
    checkpoint(config, "flux", "a.safetensors", FLUX_ORIGINAL)
    checkpoint(config, "flux", "b.safetensors", FLUX_DIFFUSERS)

    response = merge(client, "flux")
    assert response.status_code == 400
    assert "layout" in response.json()["detail"]
    assert queue.jobs == []


def test_fp8_checkpoints_are_rejected(env):
    client, config, queue = env
    checkpoint(config, "flux", "a.safetensors", FLUX_ORIGINAL)
    checkpoint(config, "flux", "b.safetensors", FLUX_ORIGINAL, dtype=torch.float8_e4m3fn)

    response = merge(client, "flux")
    assert response.status_code == 400
    assert queue.jobs == []


def test_wrong_architecture_is_rejected(env):
    client, config, queue = env
    checkpoint(config, "zimage", "a.safetensors", FLUX_ORIGINAL)
    checkpoint(config, "zimage", "b.safetensors", FLUX_ORIGINAL)
    assert merge(client, "zimage").status_code == 400


def test_shape_mismatch_rejects_the_whole_batch(env):
    client, config, queue = env
    checkpoint(config, "sdxl", "a.safetensors", ("w",))
    checkpoint(config, "sdxl", "b.safetensors", ("w",))
    checkpoint(config, "sdxl", "c.safetensors", ("w",), shape=(3, 3))

    body = {"model_type": "sdxl", "base_model": "a.safetensors",
            "target_models": ["b.safetensors", "c.safetensors"]}
    response = client.post("/api/v1/merge/batch", json=body)
    assert response.status_code == 400
    assert queue.jobs == []

    body["target_models"] = ["b.safetensors"]
    assert client.post("/api/v1/merge/batch", json=body).status_code == 200
    assert queue.jobs[0]["params"]["output_dtype"] == "auto"


def test_recipe_carries_output_dtype(env):
    client, config, queue = env
    for name in ("a", "b", "c"):
        checkpoint(config, "zimage", f"{name}.safetensors", ZIMAGE_ORIGINAL)

    body = {"model_type": "zimage", "base_model": "a.safetensors", "output_name": "r",
            "output_dtype": "float16",
            "steps": [{"target_model": "b.safetensors", "method": "linear", "alpha": 0.3},
                      {"target_model": "c.safetensors", "method": "slerp", "alpha": 0.5}]}
    assert client.post("/api/v1/merge/recipe", json=body).status_code == 200

    params = queue.jobs[0]["params"]
    assert params["output_dtype"] == "float16"
    assert [step["method"] for step in params["recipe_steps"]] == ["linear", "slerp"]


def test_missing_models_and_invalid_requests(env):
    client, config, _ = env
    checkpoint(config, "flux", "a.safetensors", FLUX_ORIGINAL)
    response = merge(client, "flux")
    assert response.status_code == 404
    assert response.json()["detail"] == "Target model not found: b.safetensors"

    assert merge(client, "sd3").status_code == 422
    assert merge(client, "flux", output_dtype="fp8").status_code == 422
