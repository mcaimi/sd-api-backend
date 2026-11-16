"""Resolving checkpoint/LoRA/VAE names to files in (sub)folders of the model directories."""

import json
from pathlib import Path

import pytest
from fastapi import HTTPException

from api.models.generation import GenerationRequest
from api.routers import model_paths
from api.services.model_files import resolve_checkpoint_from_registry


def touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def test_resolve_file_at_root(tmp_path):
    model = touch(tmp_path / "model.safetensors")
    assert resolve_checkpoint_from_registry("model.safetensors", str(tmp_path)) == model


def test_resolve_file_in_subfolder(tmp_path):
    model = touch(tmp_path / "civitai" / "DreamShaper8_pruned.safetensors")
    assert resolve_checkpoint_from_registry("DreamShaper8_pruned.safetensors", str(tmp_path)) == model


def test_resolve_relative_subpath(tmp_path):
    model = touch(tmp_path / "civitai" / "deep" / "model.safetensors")
    assert resolve_checkpoint_from_registry("civitai/deep/model.safetensors", str(tmp_path)) == model


def test_resolve_missing_file(tmp_path):
    assert resolve_checkpoint_from_registry("missing.safetensors", str(tmp_path)) is None


def test_resolve_never_escapes_directory(tmp_path):
    root = tmp_path / "checkpoints"
    root.mkdir()
    touch(tmp_path / "outside.safetensors")
    assert resolve_checkpoint_from_registry("../outside.safetensors", str(root)) is None
    assert resolve_checkpoint_from_registry(str(tmp_path / "outside.safetensors"), str(root)) is None


@pytest.fixture
def config(app_config, monkeypatch):
    monkeypatch.setattr(model_paths, "get_app_config", lambda: app_config)
    return app_config


def request(**kwargs) -> GenerationRequest:
    return GenerationRequest(positive_prompt="a cat", model_checkpoint="model.safetensors", **kwargs)


def test_resolve_generation_paths(config):
    model = touch(Path(config.checkpoints_sd15_path) / "civitai" / "model.safetensors")
    lora = touch(Path(config.loras_sd15_path) / "styles" / "lora.safetensors")
    vae = touch(Path(config.vae_sd15_path) / "vaes" / "vae.safetensors")

    model_path, loras, custom_vae = model_paths.resolve_generation_paths(
        "sd15",
        request(loras=[{"name": "lora.safetensors", "strength": 0.8}], custom_vae="vae.safetensors"),
    )

    assert model_path == model
    assert loras == [{"name": "lora.safetensors", "path": str(lora), "strength": 0.8}]
    assert custom_vae == str(vae)
    # Both end up in the generation metadata persisted as JSON
    json.dumps({"loras": loras, "vae": custom_vae})


def test_resolve_generation_paths_without_extras(config):
    touch(Path(config.checkpoints_sdxl_path) / "model.safetensors")
    _, loras, custom_vae = model_paths.resolve_generation_paths("sdxl", request())
    assert loras is None
    assert custom_vae is None


@pytest.mark.parametrize(
    "extras, detail",
    [
        ({}, "Model checkpoint not found"),
        ({"loras": [{"name": "missing.safetensors", "strength": 1.0}]}, "LoRA not found"),
        ({"custom_vae": "missing.safetensors"}, "VAE not found"),
    ],
)
def test_resolve_generation_paths_not_found(config, extras, detail):
    if extras:
        touch(Path(config.checkpoints_flux_path) / "model.safetensors")
    with pytest.raises(HTTPException) as exc:
        model_paths.resolve_generation_paths("flux", request(**extras))
    assert exc.value.status_code == 404
    assert exc.value.detail.startswith(detail)
