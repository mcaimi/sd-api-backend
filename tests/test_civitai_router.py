"""Civitai endpoints, against a mocked Civitai serving recorded responses."""

import json

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routers import civitai as civitai_router
from api.services.civitai_client import CivitaiClient, set_civitai_client
from api.services.model_files import write_sidecar
from tests.conftest import make_config
from tests.helpers import civitai_fixture

MODELS = {
    "618692": "model_flux_bundled",
    "141592": "model_flux_mixed_formats",
    "122359": "model_sdxl_lora",
    "296576": "model_sdxl_vae",
}


class FakeCivitai:
    def __init__(self):
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        path = request.url.path.removeprefix("/api/v1")
        if path == "/enums":
            return httpx.Response(200, json=civitai_fixture("enums"))
        if path == "/models":
            return httpx.Response(200, json=civitai_fixture("search_query_illustrious_checkpoint"))
        if path.startswith("/models/"):
            model_id = path.rsplit("/", 1)[1]
            if model_id in MODELS:
                return httpx.Response(200, json=civitai_fixture(MODELS[model_id]))
            return httpx.Response(404, json={"error": f"No model with id {model_id}"})
        return httpx.Response(404, json={"error": "not mocked"})


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.delenv("CIVITAI_API_KEY", raising=False)
    state = {"config": make_config(tmp_path, civitai={"api_key": "secret"})}
    monkeypatch.setattr(civitai_router, "get_app_config", lambda: state["config"])
    fake = FakeCivitai()
    set_civitai_client(CivitaiClient("https://civitai.com/api/v1", api_key="secret",
                                     transport=httpx.MockTransport(fake)))
    app = FastAPI()
    app.include_router(civitai_router.router, prefix="/api/v1/civitai")
    yield TestClient(app), fake, state, tmp_path
    set_civitai_client(None)


def test_status(setup):
    client, _, _, _ = setup
    data = client.get("/api/v1/civitai/status").json()
    assert data["enabled"] and data["has_api_key"] and not data["nsfw_default"]
    assert "Pony" in data["model_types"]["sdxl"]
    assert data["resource_types"]["loras"] == ["LORA", "LoCon", "DoRA"]


def test_search_translates_filters(setup):
    client, fake, _, _ = setup
    response = client.get("/api/v1/civitai/search", params={
        "model_type": "illustrious", "resource_type": "checkpoints", "query": " anime ", "limit": 3})
    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) == 3 and data["next_cursor"] == "3"
    search = [r for r in fake.requests if r.url.path == "/api/v1/models"][0]
    assert search.url.params.get_list("baseModels") == ["Illustrious", "NoobAI"]
    assert search.url.params.get_list("types") == ["Checkpoint"]
    assert search.url.params["query"] == "anime"
    assert search.url.params["nsfw"] == "false"


@pytest.mark.parametrize("params,detail", [
    ({"model_type": "sd3", "resource_type": "checkpoints"}, "model_type"),
    ({"model_type": "sdxl", "resource_type": "embeddings"}, "resource_type"),
    ({"model_type": "sdxl", "resource_type": "loras", "sort": "Random"}, "sort"),
    ({"model_type": "sdxl", "resource_type": "loras", "period": "Decade"}, "period"),
])
def test_search_validates_parameters(setup, params, detail):
    client, _, _, _ = setup
    response = client.get("/api/v1/civitai/search", params=params)
    assert response.status_code == 400 and detail in response.json()["detail"]


def test_model_detail_marks_local_files(setup):
    client, _, state, _ = setup
    loras = state["config"].get_model_paths("sdxl")["loras"]
    from pathlib import Path
    subdir = Path(loras) / "civitai"
    subdir.mkdir()
    local = subdir / "add-detail-xl.safetensors"
    local.write_bytes(b"x")
    write_sidecar(local, {"sha256": "0D9BD1B873A7863E128B4672E3E245838858F71469A3CEC58123C16C06F83BD7"})
    (Path(loras) / "tweak-detail-xl.safetensors").write_bytes(b"y")

    data = client.get("/api/v1/civitai/models/122359", params={"model_type": "sdxl"}).json()
    files = {f["name"]: f for v in data["versions"] for f in v["files"]}
    assert files["add-detail-xl.safetensors"]["local_match"] == "sha256"
    assert files["add-detail-xl.safetensors"]["local_path"] == str(local)
    assert files["tweak-detail-xl.safetensors"]["local_match"] == "name"


def test_model_detail_flux_files(setup):
    client, _, _, _ = setup
    data = client.get("/api/v1/civitai/models/141592", params={"model_type": "flux"}).json()
    files = [f for v in data["versions"] for f in v["files"]]
    for f in files:
        loadable = f["format"] == "SafeTensor" and f["fp"] != "nf4"
        assert f["compatible"] == loadable, f["name"]
    assert {f["format"] for f in files} == {"SafeTensor", "GGUF"}


def test_unknown_model_is_404(setup):
    client, _, _, _ = setup
    response = client.get("/api/v1/civitai/models/1", params={"model_type": "flux"})
    assert response.status_code == 404 and "No model with id 1" in response.json()["detail"]


def test_disabled_feature_is_404(setup, tmp_path):
    client, _, state, _ = setup
    state["config"] = make_config(tmp_path, civitai={"enabled": False})
    response = client.get("/api/v1/civitai/search", params={"model_type": "sdxl", "resource_type": "loras"})
    assert response.status_code == 404
    assert client.get("/api/v1/civitai/status").json()["enabled"] is False


def test_download_endpoint(setup, monkeypatch):
    client, _, _, _ = setup
    calls = []

    async def fake_start(**kwargs):
        calls.append(kwargs)
        return {"job_id": "j1", "filename": "a.safetensors", "path": "/m/a.safetensors", "warnings": []}

    monkeypatch.setattr(civitai_router, "start_download", fake_start)
    body = {"version_id": 10, "file_id": 100, "model_type": "sdxl", "resource_type": "loras"}
    response = client.post("/api/v1/civitai/downloads", json=body)
    assert response.status_code == 202 and response.json()["job_id"] == "j1"
    assert calls[0]["accept_mismatch"] is False and calls[0]["file_id"] == 100


def test_download_endpoint_errors(setup, monkeypatch):
    client, _, _, _ = setup
    from api.services.downloads import DownloadError

    async def fake_start(**kwargs):
        raise DownloadError(409, {"message": "Base model doesn't match", "warnings": ["Made for Pony"]})

    monkeypatch.setattr(civitai_router, "start_download", fake_start)
    body = {"version_id": 10, "file_id": 100, "model_type": "sdxl", "resource_type": "loras"}
    response = client.post("/api/v1/civitai/downloads", json=body)
    assert response.status_code == 409 and response.json()["detail"]["warnings"] == ["Made for Pony"]
    bad = client.post("/api/v1/civitai/downloads", json=dict(body, model_type="sd3"))
    assert bad.status_code == 400
