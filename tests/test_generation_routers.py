"""Generation routers: job parameters per architecture, Z-Image variants, schedulers."""

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routers import generation as generation_router
from api.routers import info as info_router
from api.routers import model_paths
from api.routers.generation import ARCHITECTURES, create_generation_router
from metastable.schedulers import FLOW_MATCH_SCHEDULERS, SCHEDULERS

from tests.conftest import make_config


class FakeJobQueue:
    def __init__(self):
        self.jobs = []

    async def submit_job(self, job_type, params, callback, **kwargs):
        self.jobs.append({"job_type": job_type, "params": params, "callback": callback.__name__})
        return f"job-{len(self.jobs)}"


@pytest.fixture
def env(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    monkeypatch.setattr(model_paths, "get_app_config", lambda: config)
    queue = FakeJobQueue()
    monkeypatch.setattr(generation_router, "get_job_queue", lambda: queue)

    app = FastAPI()
    for arch in ARCHITECTURES:
        app.include_router(create_generation_router(arch), prefix=f"/api/v1/{arch.model_type}")
    app.include_router(info_router.router, prefix="/api/v1/info")

    for model_type in ("sdxl", "flux", "zimage"):
        checkpoint = Path(config.get_model_paths(model_type)["checkpoints"]) / "model.safetensors"
        checkpoint.write_bytes(b"")
    return TestClient(app), config, queue


def generate(client, model_type, **body):
    body = {"positive_prompt": "a fox", "model_checkpoint": "model.safetensors", **body}
    return client.post(f"/api/v1/{model_type}/generate", json=body)


def test_single_file_architecture_job(env):
    client, _, queue = env
    assert generate(client, "sdxl").status_code == 200

    [job] = queue.jobs
    assert job["job_type"] == "sdxl_generation"
    assert job["callback"] == "generate_image_async"
    assert job["params"]["components_dir"] is None
    assert job["params"]["pipeline_kwargs"] == {}
    assert job["params"]["scheduler"] == "DPM++ 2M"


def test_flux_job_uses_shared_components_and_true_cfg(env):
    client, config, queue = env
    assert generate(client, "flux", true_cfg_scale=2.0).status_code == 200

    params = queue.jobs[0]["params"]
    assert params["components_dir"] == config.components_path("flux")
    assert params["pipeline_kwargs"] == {"true_cfg_scale": 2.0}
    assert params["cfg"] == 3.5


@pytest.mark.parametrize("variant, steps, cfg", [("turbo", 8, 0.0), ("base", 28, 4.0)])
def test_zimage_variant_selects_components_and_defaults(env, variant, steps, cfg):
    client, config, queue = env
    body = {"variant": variant} if variant != "turbo" else {}
    assert generate(client, "zimage", **body).status_code == 200

    params = queue.jobs[0]["params"]
    assert params["components_dir"] == config.components_path("zimage", variant)
    assert (params["steps"], params["cfg"]) == (steps, cfg)
    assert (params["width"], params["height"]) == (1024, 1024)
    assert params["scheduler"] == "FlowMatch Euler"


def test_zimage_explicit_settings_win(env):
    client, _, queue = env
    assert generate(client, "zimage", variant="base", steps=40, cfg_scale=0.0).status_code == 200
    params = queue.jobs[0]["params"]
    assert (params["steps"], params["cfg"]) == (40, 0.0)


def test_zimage_unconfigured_variant_is_rejected(env):
    client, config, queue = env
    del config.parameters.zimage.components.base
    response = generate(client, "zimage", variant="base")
    assert response.status_code == 400
    assert "zimage.components.base" in response.json()["detail"]
    assert queue.jobs == []


def test_zimage_inpaint_strength(env):
    client, _, queue = env
    body = {"positive_prompt": "a fox", "model_checkpoint": "model.safetensors",
            "image_base64": "img", "mask_base64": "mask", "strength": 0.7}
    assert client.post("/api/v1/zimage/inpaint", json=body).status_code == 200

    job = queue.jobs[0]
    assert job["job_type"] == "zimage_inpaint"
    assert job["params"]["pipeline_kwargs"] == {"strength": 0.7}
    assert job["params"]["steps"] == 8


def test_sd_inpaint_gets_no_flux_arguments(env):
    client, _, queue = env
    body = {"positive_prompt": "a fox", "model_checkpoint": "model.safetensors",
            "image_base64": "img", "mask_base64": "mask"}
    assert client.post("/api/v1/sdxl/inpaint", json=body).status_code == 200
    assert queue.jobs[0]["params"]["pipeline_kwargs"] == {}


def test_compare_forwards_settings_to_each_model(env):
    client, config, queue = env
    other = Path(config.get_model_paths("zimage")["checkpoints"]) / "other.safetensors"
    other.write_bytes(b"")
    body = {"positive_prompt": "a fox", "models": ["model.safetensors", "other.safetensors"],
            "variant": "base"}
    response = client.post("/api/v1/zimage/generate/compare", json=body)
    assert response.status_code == 200

    assert [job["params"]["model_path"] for job in queue.jobs] == [
        str(Path(config.get_model_paths("zimage")["checkpoints"]) / name)
        for name in body["models"]
    ]
    assert {job["params"]["components_dir"] for job in queue.jobs} == {config.components_path("zimage", "base")}
    assert {job["params"]["seed"] for job in queue.jobs} == {42}


@pytest.mark.parametrize("model_type, scheduler", [
    ("sdxl", "FlowMatch Euler"), ("flux", "DPM++ 2M"), ("zimage", "FlowMatch Heun"),
])
def test_wrong_scheduler_family_is_rejected(env, model_type, scheduler):
    client, _, queue = env
    assert generate(client, model_type, scheduler=scheduler).status_code == 422
    assert queue.jobs == []


def test_turbo_cfg_zero_is_only_allowed_for_zimage(env):
    client, _, _ = env
    assert generate(client, "zimage", cfg_scale=0.0).status_code == 200
    assert generate(client, "flux", cfg_scale=0.0).status_code == 422


def test_schedulers_by_model_type(env):
    client, _, _ = env
    assert client.get("/api/v1/info/schedulers").json() == list(SCHEDULERS)
    assert client.get("/api/v1/info/schedulers", params={"model_type": "sd15"}).json() == list(SCHEDULERS)
    for model_type in ("flux", "zimage"):
        response = client.get("/api/v1/info/schedulers", params={"model_type": model_type})
        assert response.json() == list(FLOW_MATCH_SCHEDULERS)
    assert client.get("/api/v1/info/schedulers", params={"model_type": "sd3"}).status_code == 400
