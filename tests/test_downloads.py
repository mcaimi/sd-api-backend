"""Civitai download service, against a mocked Civitai and CDN."""

import asyncio
import hashlib
import json
from pathlib import Path

import httpx
import pytest
import torch
from safetensors.torch import save

from api.services import downloads
from api.services.civitai_client import CivitaiClient, set_civitai_client
from api.services.downloads import DownloadError
from api.services.job_queue import JobQueue, set_job_queue
from api.services.model_files import write_sidecar
from tests.conftest import make_config

LORA_BYTES = save({"lora_unet_down.weight": torch.zeros(64, 64)})
FLUX_AIO_BYTES = save({
    "model.diffusion_model.double_blocks.0.img_attn.norm.key_norm.scale": torch.zeros(8),
    "model.diffusion_model.guidance_in.in_layer.weight": torch.zeros(8),
    "text_encoders.clip_l.transformer.text_model.final_layer_norm.weight": torch.zeros(8),
    "vae.decoder.conv_in.bias": torch.zeros(8),
})


def make_version(payload: bytes, *, version_id=10, model_type="LORA", base_model="SDXL 1.0",
                 name="my lora (v2).safetensors", fmt="SafeTensor", sha256=None):
    return {
        "id": version_id, "modelId": 5, "name": "v2",
        "model": {"name": "Test Model", "type": model_type, "nsfw": False},
        "baseModel": base_model, "baseModelType": "Standard", "trainedWords": ["trigger"],
        "images": [{"url": "https://image.civitai.com/abc/original=true/1.jpeg", "type": "image", "nsfwLevel": 1}],
        "files": [{
            "id": 100, "name": name, "type": "Model", "sizeKB": len(payload) / 1024,
            "metadata": {"format": fmt, "fp": "fp16", "size": "full"},
            "hashes": {"SHA256": (sha256 or hashlib.sha256(payload).hexdigest()).upper()},
            "downloadUrl": f"https://civitai.com/api/download/models/{version_id}",
        }],
    }


class FakeServers:
    """Civitai API + download redirect + CDN + image host."""

    def __init__(self, version, payload, cdn_status=200, cdn_type="application/octet-stream", slow=False):
        self.version = version
        self.payload = payload
        self.cdn_status = cdn_status
        self.cdn_type = cdn_type
        self.slow = slow
        self.requests = []

    async def _slow_body(self):
        for i in range(0, len(self.payload), 1024):
            yield self.payload[i:i + 1024]
            await asyncio.sleep(0.05)

    def __call__(self, request):
        self.requests.append(request)
        host, path = request.url.host, request.url.path
        if host == "civitai.com" and path == f"/api/v1/model-versions/{self.version['id']}":
            return httpx.Response(200, json=self.version)
        if host == "civitai.com" and path.startswith("/api/download/models/"):
            return httpx.Response(307, headers={"Location": "https://cdn.r2.example/file?sig=1"})
        if host == "cdn.r2.example":
            if self.cdn_status != 200:
                return httpx.Response(self.cdn_status, json={"error": "Unauthorized", "message": "log in first"})
            body = self._slow_body() if self.slow else self.payload
            return httpx.Response(200, headers={"content-type": self.cdn_type}, content=body)
        if host == "image.civitai.com":
            assert "/width=450/" in path
            return httpx.Response(200, content=b"\xff\xd8jpeg")
        return httpx.Response(404, json={"error": "not mocked"})


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.delenv("CIVITAI_API_KEY", raising=False)
    config = make_config(tmp_path, civitai={"api_key": "secret", "min_free_space_gb": 0})
    monkeypatch.setattr(downloads, "get_app_config", lambda: config)
    yield config
    set_civitai_client(None)
    set_job_queue(None)


def run_scenario(servers, coro_fn):
    """Run coro_fn(queue, client) with a started job queue and the fake servers."""
    async def main():
        client = CivitaiClient("https://civitai.com/api/v1", api_key="secret",
                               transport=httpx.MockTransport(servers))
        set_civitai_client(client)
        queue = JobQueue(num_workers=1, io_workers=2, result_ttl=60)
        set_job_queue(queue)
        await queue.start_workers()
        try:
            return await coro_fn(queue, client)
        finally:
            await queue.stop_workers()
            await client.aclose()
    return asyncio.run(main())


async def wait_done(queue, job_id, timeout=5):
    for _ in range(int(timeout / 0.02)):
        status = await queue.get_job_status(job_id)
        if status["status"] in ("completed", "failed", "cancelled") and status["completed_at"]:
            return status
        await asyncio.sleep(0.02)
    raise AssertionError(f"job still {status['status']}")


def download(model_type="sdxl", resource_type="loras", **kwargs):
    async def scenario(queue, client):
        started = await downloads.start_download(10, 100, model_type, resource_type, client=client, **kwargs)
        return started, await wait_done(queue, started["job_id"])
    return scenario


def test_happy_path(env):
    servers = FakeServers(make_version(LORA_BYTES), LORA_BYTES)
    started, status = run_scenario(servers, download())
    dest = Path(env.get_model_paths("sdxl")["loras"]) / "civitai" / "my_lora_v2.safetensors"
    assert started["path"] == str(dest) and started["filename"] == dest.name
    assert status["status"] == "completed", status["error"]
    assert dest.read_bytes() == LORA_BYTES
    assert not Path(str(dest) + ".part").exists()
    assert status["result"]["sha256"] == hashlib.sha256(LORA_BYTES).hexdigest()
    assert status["detail"]["phase"] == "verifying"
    sidecar = json.loads(dest.with_suffix(".civitai.json").read_text())
    assert sidecar["sha256"] == hashlib.sha256(LORA_BYTES).hexdigest()
    assert sidecar["trained_words"] == ["trigger"] and sidecar["version_id"] == 10
    assert sidecar["civitai_type"] == "LORA" and "downloaded_at" in sidecar
    assert dest.with_suffix(".preview.jpg").read_bytes() == b"\xff\xd8jpeg"


def test_api_key_not_forwarded_to_cdn(env):
    servers = FakeServers(make_version(LORA_BYTES), LORA_BYTES)
    run_scenario(servers, download())
    assert all(r.headers.get("Authorization") == "Bearer secret" for r in servers.requests if r.url.host == "civitai.com")
    assert all("Authorization" not in r.headers for r in servers.requests if r.url.host != "civitai.com")


def test_flux_checkpoint_layout_is_recorded(env):
    version = make_version(FLUX_AIO_BYTES, model_type="Checkpoint", base_model="Flux.1 D", name="aio.safetensors")
    _, status = run_scenario(FakeServers(version, FLUX_AIO_BYTES), download("flux", "checkpoints"))
    layout = status["result"]["flux_layout"]
    assert (layout["transformer"], layout["clip_l"], layout["t5xxl"], layout["vae"]) == (True, True, False, True)
    dest = Path(status["result"]["path"])
    assert json.loads(dest.with_suffix(".civitai.json").read_text())["flux_layout"] == layout


def test_hash_mismatch_discards_file(env):
    version = make_version(LORA_BYTES, sha256="0" * 64)
    _, status = run_scenario(FakeServers(version, LORA_BYTES), download())
    assert status["status"] == "failed" and "Hash mismatch" in status["error"]
    assert list(Path(env.get_model_paths("sdxl")["loras"]).rglob("*.safetensors*")) == []


def test_auth_failure_on_download(env):
    servers = FakeServers(make_version(LORA_BYTES), LORA_BYTES, cdn_status=401)
    _, status = run_scenario(servers, download())
    assert status["status"] == "failed" and "log in first" in status["error"]


def test_web_page_instead_of_file(env):
    servers = FakeServers(make_version(LORA_BYTES), b"<html>login</html>", cdn_type="text/html; charset=utf-8")
    _, status = run_scenario(servers, download())
    assert status["status"] == "failed" and "web page" in status["error"]


def test_cancel_mid_stream_removes_partial_file(env):
    payload = LORA_BYTES + b"\0" * 20_000
    servers = FakeServers(make_version(payload), payload, slow=True)

    async def scenario(queue, client):
        started = await downloads.start_download(10, 100, "sdxl", "loras", client=client)
        part = Path(started["path"] + ".part")
        for _ in range(100):
            if part.exists() and (await queue.get_job_status(started["job_id"]))["detail"]:
                break
            await asyncio.sleep(0.02)
        assert part.exists()
        assert await queue.cancel_job(started["job_id"])
        return started, await wait_done(queue, started["job_id"])

    started, status = run_scenario(servers, scenario)
    assert status["status"] == "cancelled"
    assert not Path(started["path"]).exists() and not Path(started["path"] + ".part").exists()


def test_duplicate_in_flight_download_is_rejected(env):
    payload = LORA_BYTES + b"\0" * 20_000
    servers = FakeServers(make_version(payload), payload, slow=True)

    async def scenario(queue, client):
        first = await downloads.start_download(10, 100, "sdxl", "loras", client=client)
        with pytest.raises(DownloadError) as exc:
            await downloads.start_download(10, 100, "sdxl", "loras", client=client)
        await queue.cancel_job(first["job_id"])
        return first, exc.value

    first, error = run_scenario(servers, scenario)
    assert error.status_code == 409 and error.detail["job_id"] == first["job_id"]


def test_existing_identical_file_anywhere_in_tree_is_rejected(env):
    loras = Path(env.get_model_paths("sdxl")["loras"])
    (loras / "manual").mkdir()
    existing = loras / "manual" / "renamed.safetensors"
    existing.write_bytes(LORA_BYTES)
    write_sidecar(existing, {"sha256": hashlib.sha256(LORA_BYTES).hexdigest()})

    async def scenario(queue, client):
        with pytest.raises(DownloadError) as exc:
            await downloads.start_download(10, 100, "sdxl", "loras", client=client)
        return exc.value

    error = run_scenario(FakeServers(make_version(LORA_BYTES), LORA_BYTES), scenario)
    assert error.status_code == 409 and error.detail["path"] == str(existing)


def test_name_clash_with_different_file_gets_version_suffix(env):
    loras = Path(env.get_model_paths("sdxl")["loras"])
    (loras / "my_lora_v2.safetensors").write_bytes(b"something else")
    started, status = run_scenario(FakeServers(make_version(LORA_BYTES), LORA_BYTES), download())
    assert started["filename"] == "my_lora_v2-v10.safetensors"
    assert status["status"] == "completed"


def test_not_enough_disk_space(env, monkeypatch):
    monkeypatch.setattr(downloads.shutil, "disk_usage", lambda path: type("U", (), {"free": 100})())

    async def scenario(queue, client):
        with pytest.raises(DownloadError) as exc:
            await downloads.start_download(10, 100, "sdxl", "loras", client=client)
        return exc.value

    error = run_scenario(FakeServers(make_version(LORA_BYTES), LORA_BYTES), scenario)
    assert error.status_code == 507 and "Not enough disk space" in error.detail


def test_base_model_mismatch_needs_confirmation(env):
    servers = FakeServers(make_version(LORA_BYTES), LORA_BYTES)

    async def scenario(queue, client):
        with pytest.raises(DownloadError) as exc:
            await downloads.start_download(10, 100, "illustrious", "loras", client=client)
        started = await downloads.start_download(10, 100, "illustrious", "loras", accept_mismatch=True, client=client)
        return exc.value, started, await wait_done(queue, started["job_id"])

    error, started, status = run_scenario(servers, scenario)
    assert error.status_code == 409 and error.detail["warnings"] == ["Made for SDXL 1.0 (sdxl), not illustrious"]
    assert status["status"] == "completed"
    assert "/illustrious/" in started["path"]


@pytest.mark.parametrize("kwargs,args,status_code,text", [
    ({"fmt": "GGUF"}, (10, 100, "sdxl", "loras"), 400, "GGUF format not supported"),
    ({}, (10, 100, "sdxl", "checkpoints"), 400, "belongs in loras"),
    ({}, (10, 999, "sdxl", "loras"), 404, "not part of Civitai version"),
])
def test_invalid_requests(env, kwargs, args, status_code, text):
    servers = FakeServers(make_version(LORA_BYTES, **kwargs), LORA_BYTES)

    async def scenario(queue, client):
        with pytest.raises(DownloadError) as exc:
            await downloads.start_download(*args, client=client)
        return exc.value

    error = run_scenario(servers, scenario)
    assert error.status_code == status_code and text in str(error.detail)


def test_cleanup_removes_only_partial_safetensors(env):
    loras = Path(env.get_model_paths("sdxl")["loras"])
    (loras / "civitai").mkdir()
    partial = loras / "civitai" / "x.safetensors.part"
    partial.write_bytes(b"x")
    other = loras / "notes.txt.part"
    other.write_bytes(b"x")
    model = loras / "y.safetensors"
    model.write_bytes(b"x")
    assert downloads.cleanup_partial_downloads() == [partial]
    assert not partial.exists() and other.exists() and model.exists()


def test_preview_url_is_resized_and_restricted():
    assert downloads.preview_download_url("https://image.civitai.com/a/original=true/1.jpeg") == \
        "https://image.civitai.com/a/width=450/1.jpeg"
    assert downloads.preview_download_url("https://evil.example/original=true/1.jpeg") is None
    assert downloads.preview_download_url("http://image.civitai.com/a/1.jpeg") is None
