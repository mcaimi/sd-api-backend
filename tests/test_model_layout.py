"""Flux checkpoint layout detection, against key lists from real Civitai headers."""

import pytest
import torch
from safetensors.torch import save_file

from api.services import model_layout
from tests.helpers import flux_header_keys


@pytest.mark.parametrize("fixture,expected", [
    ("transformer_bfl_bare", {"transformer": True, "clip_l": False, "t5xxl": False, "vae": False,
                              "format": "bfl", "variant": "dev"}),
    ("transformer_comfy_prefixed", {"transformer": True, "clip_l": False, "t5xxl": False, "vae": False,
                                    "format": "bfl", "variant": "dev"}),
    ("transformer_schnell_prefixed", {"transformer": True, "clip_l": False, "t5xxl": False, "vae": False,
                                      "format": "bfl", "variant": "schnell"}),
    ("all_in_one", {"transformer": True, "clip_l": True, "t5xxl": True, "vae": True,
                    "format": "bfl", "variant": "dev"}),
])
def test_real_headers(fixture, expected):
    layout = model_layout.classify_keys(flux_header_keys(fixture))
    assert layout == expected
    assert model_layout.is_bundled(layout) == (fixture == "all_in_one")


def test_diffusers_layout_and_non_flux_file():
    diffusers = model_layout.classify_keys(["transformer_blocks.0.attn.to_q.weight", "time_text_embed.guidance_embedder.x"])
    assert diffusers["format"] == "diffusers" and diffusers["variant"] == "dev"
    sdxl = model_layout.classify_keys(["model.diffusion_model.input_blocks.0.0.weight", "first_stage_model.decoder.x"])
    assert sdxl["transformer"] is False and sdxl["format"] is None


def test_classify_local_file(tmp_path):
    path = tmp_path / "aio.safetensors"
    save_file({
        "model.diffusion_model.double_blocks.0.img_attn.norm.key_norm.scale": torch.zeros(1),
        "text_encoders.t5xxl.transformer.shared.weight": torch.zeros(1),
        "vae.decoder.conv_in.bias": torch.zeros(1),
    }, str(path))
    layout = model_layout.classify(path)
    assert (layout["transformer"], layout["clip_l"], layout["t5xxl"], layout["vae"]) == (True, False, True, True)
    assert layout["variant"] == "schnell"


def test_missing_components(tmp_path):
    for name in ("tokenizer", "tokenizer_2", "scheduler", "text_encoder"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "config.json").write_text("{}")
    (tmp_path / "vae").mkdir()  # empty folder counts as missing
    transformer_only = model_layout.classify_keys(flux_header_keys("transformer_comfy_prefixed"))
    all_in_one = model_layout.classify_keys(flux_header_keys("all_in_one"))
    assert model_layout.missing_components(transformer_only, str(tmp_path)) == ["text_encoder_2", "vae"]
    assert model_layout.missing_components(all_in_one, str(tmp_path)) == []
    assert model_layout.missing_components(all_in_one, str(tmp_path / "nope")) == ["tokenizer", "tokenizer_2", "scheduler"]
