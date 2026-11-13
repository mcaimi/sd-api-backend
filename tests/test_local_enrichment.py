"""Models listing enrichment, previews, identify and remote Flux layout checks."""

import json
import struct
from pathlib import Path

import httpx
import pytest
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from safetensors.torch import save_file

from api.routers import civitai as civitai_router
from api.routers import models as models_router
from api.services import model_files
from api.services.civitai_client import CivitaiClient, set_civitai_client
from api.services.model_files import read_sidecar, write_sidecar
from tests.conftest import make_config
from tests.helpers import civitai_fixture, flux_header_keys

LORA_SHA = "0d9bd1b873a7863e128b4672e3e245838858f71469a3cec58123c16c06f83bd7"


def flux_version():
    model = civitai_fixture("model_flux_bundled")
    version = dict(model["modelVersions"][0])
    version["model"] = {"name": model["name"], "type": model["type"], "nsfw": model["nsfw"]}
    version["modelId"] = model["id"]
    return version


def header_bytes(keys):
    raw = json.dumps({k: {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]} for k in keys}).encode()
    return struct.pack("<Q", len(raw)) + raw


class FakeCivitai:
    def __init__(self):
        self.requests = []
        self.flux = flux_version()
        self.aio_header = header_bytes(flux_header_keys("all_in_one"))

    def __call__(self, request):
        self.requests.append(request)
        host, path = request.url.host, request.url.path
        if path == f"/api/v1/model-versions/by-hash/{LORA_SHA}":
            return httpx.Response(200, json=civitai_fixture("version_by_hash_sdxl_lora"))
        if path.startswith("/api/v1/model-versions/by-hash/"):
            return httpx.Response(404, json={"error": "Model not found"})
        if path == f"/api/v1/model-versions/{self.flux['id']}":
            return httpx.Response(200, json=self.flux)
        if host == "civitai.com" and path.startswith("/api/download/models/"):
            return httpx.Response(307, headers={"Location": "https://cdn.r2.example/f"})
        if host == "cdn.r2.example":
            start, end = map(int, request.headers["Range"].removeprefix("bytes=").split("-"))
            return httpx.Response(206, content=self.aio_header[start:end + 1])
        if host == "image.civitai.com":
            return httpx.Response(200, content=b"\xff\xd8preview")
        return httpx.Response(404, json={"error": "not mocked"})


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.delenv("CIVITAI_API_KEY", raising=False)
    config = make_config(tmp_path, civitai={"api_key": "secret"})
    for module in (civitai_router, models_router):
        monkeypatch.setattr(module, "get_app_config", lambda: config)
    fake = FakeCivitai()
    set_civitai_client(CivitaiClient("https://civitai.com/api/v1", api_key="secret",
                                     transport=httpx.MockTransport(fake)))
    civitai_router._layout_cache.clear()
    app = FastAPI()
    app.include_router(models_router.router, prefix="/api/v1/models")
    app.include_router(civitai_router.router, prefix="/api/v1/civitai")
    yield TestClient(app), config, fake
    set_civitai_client(None)


def test_listing_includes_sidecar_preview_and_flux_layout(env):
    client, config, _ = env
    loras = Path(config.get_model_paths("sdxl")["loras"])
    lora = loras / "detail.safetensors"
    lora.write_bytes(b"x")
    write_sidecar(lora, {"source": "civitai", "model_id": 1, "model_name": "Detail", "trained_words": ["dt"],
                         "base_model": "SDXL 1.0", "sha256": LORA_SHA, "unexpected": "ignored"})
    lora.with_suffix(".preview.jpg").write_bytes(b"jpg")
    hashed_only = loras / "local.safetensors"
    hashed_only.write_bytes(b"y")
    write_sidecar(hashed_only, {"source": "local", "sha256": "ab"})

    models = {m["name"]: m for m in client.get("/api/v1/models/sdxl/loras").json()["models"]}
    assert models["detail.safetensors"]["civitai"]["trained_words"] == ["dt"]
    assert models["detail.safetensors"]["has_preview"] is True
    assert models["detail.safetensors"]["flux_layout"] is None
    assert models["local.safetensors"]["civitai"] is None and models["local.safetensors"]["has_preview"] is False

    flux = Path(config.get_model_paths("flux")["checkpoints"]) / "t.safetensors"
    save_file({"double_blocks.0.img_attn.norm.key_norm.scale": torch.zeros(1)}, str(flux))
    entry = client.get("/api/v1/models/flux/checkpoints").json()["models"][0]
    assert entry["flux_layout"]["transformer"] is True and entry["flux_layout"]["t5xxl"] is False


def test_preview_endpoint(env):
    client, config, _ = env
    loras = Path(config.get_model_paths("sd15")["loras"])
    (loras / "a.safetensors").write_bytes(b"x")
    (loras / "a.preview.jpg").write_bytes(b"jpg")
    (loras / "b.safetensors").write_bytes(b"x")
    response = client.get("/api/v1/models/sd15/loras/a.safetensors/preview")
    assert response.status_code == 200 and response.content == b"jpg"
    assert client.get("/api/v1/models/sd15/loras/b.safetensors/preview").status_code == 404
    assert client.get("/api/v1/models/sd15/loras/missing.safetensors/preview").status_code == 404


def test_identify_match_writes_sidecar_and_preview(env):
    client, config, _ = env
    lora = Path(config.get_model_paths("sdxl")["loras"]) / "renamed.safetensors"
    lora.write_bytes(b"x")
    write_sidecar(lora, {"source": "local", "sha256": LORA_SHA})  # hash cached by an earlier attempt

    response = client.post("/api/v1/civitai/identify",
                           json={"model_type": "sdxl", "resource_type": "loras", "model_name": "renamed.safetensors"})
    data = response.json()
    assert response.status_code == 200 and data["matched"] is True
    sidecar = read_sidecar(lora)
    assert sidecar["source"] == "civitai" and sidecar["model_id"] == 122359
    assert sidecar["file_name"] == "add-detail-xl.safetensors" and sidecar["sha256"] == LORA_SHA
    assert "identified_at" in sidecar
    assert lora.with_suffix(".preview.jpg").read_bytes() == b"\xff\xd8preview"


def test_identify_no_match_caches_hash(env, monkeypatch):
    client, config, _ = env
    lora = Path(config.get_model_paths("sdxl")["loras"]) / "unknown.safetensors"
    lora.write_bytes(b"not on civitai")
    calls = []
    real = model_files.sha256_file
    monkeypatch.setattr(civitai_router, "sha256_file", lambda path: calls.append(path) or real(path))
    body = {"model_type": "sdxl", "resource_type": "loras", "model_name": "unknown.safetensors"}

    assert client.post("/api/v1/civitai/identify", json=body).json() == {"matched": False, "sidecar": None}
    assert client.post("/api/v1/civitai/identify", json=body).json()["matched"] is False
    assert len(calls) == 1
    assert read_sidecar(lora)["source"] == "local"
    assert client.get("/api/v1/models/sdxl/loras").json()["models"][0]["civitai"] is None


def test_identify_unknown_model_is_404(env):
    client, _, _ = env
    response = client.post("/api/v1/civitai/identify",
                           json={"model_type": "sdxl", "resource_type": "loras", "model_name": "nope.safetensors"})
    assert response.status_code == 404


def test_remote_flux_layout(env):
    client, config, fake = env
    version = fake.flux
    file = version["files"][0]
    components = Path(config.flux_components_path)
    for name in ("tokenizer", "tokenizer_2"):
        (components / name).mkdir()
        (components / name / "x.json").write_text("{}")

    params = {"version_id": version["id"], "file_id": file["id"]}
    data = client.get("/api/v1/civitai/files/layout", params=params).json()
    assert data["layout"] == {"transformer": True, "clip_l": True, "t5xxl": True, "vae": True,
                              "format": "bfl", "variant": "dev"}
    assert data["missing_components"] == ["scheduler"]

    cdn_calls = len([r for r in fake.requests if r.url.host == "cdn.r2.example"])
    client.get("/api/v1/civitai/files/layout", params=params)
    assert len([r for r in fake.requests if r.url.host == "cdn.r2.example"]) == cdn_calls


def test_remote_layout_rejects_non_flux_and_unknown_files(env):
    client, _, fake = env
    fake.flux["baseModel"] = "SDXL 1.0"
    response = client.get("/api/v1/civitai/files/layout",
                          params={"version_id": fake.flux["id"], "file_id": fake.flux["files"][0]["id"]})
    assert response.status_code == 400
    response = client.get("/api/v1/civitai/files/layout", params={"version_id": fake.flux["id"], "file_id": 1})
    assert response.status_code == 404
