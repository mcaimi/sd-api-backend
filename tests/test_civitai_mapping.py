"""Civitai type mapping, file compatibility rules and normalisation, against recorded API responses."""

import copy

import pytest

from api.services import civitai_mapping as mapping
from tests.helpers import civitai_fixture


def files_by_name(detail, version_index=0):
    return {f["name"]: f for f in detail["versions"][version_index]["files"]}


def test_every_mapped_base_model_exists_on_civitai():
    known = set(civitai_fixture("enums")["BaseModel"])
    for model_type, names in mapping.MODEL_BASE_MODELS.items():
        assert set(names) <= known, model_type


def test_unknown_base_models_are_dropped():
    assert mapping.base_models("illustrious", known={"Illustrious"}) == ["Illustrious"]


def test_excluded_base_models_are_not_mapped():
    for name in ("Pony V7", "Flux.1 Kontext", "Flux.2 D", "SD 2.1", "SD 3.5"):
        assert mapping.model_type_for_base_model(name) is None


def test_pony_and_noobai_mapping():
    assert mapping.model_type_for_base_model("Pony") == "sdxl"
    assert mapping.model_type_for_base_model("NoobAI") == "illustrious"


def test_flux_formats():
    files = files_by_name(mapping.normalize_detail(civitai_fixture("model_flux_mixed_formats"), "flux", False))
    assert files["pixelwave_flux1_schnell_04_fp8.safetensors"]["compatible"]
    assert files["pixelwave_flux1_schnell_04_bf16.safetensors"]["compatible"]
    assert files["pixelwave_flux1_schnell_04_Q4_K_M.gguf"]["reason"] == "GGUF format not supported"
    assert files["pixelwave_flux1_schnell_04_nf4.safetensors"]["reason"] == "Pre-quantized NF4 not supported"


def test_inpainting_checkpoint_is_rejected():
    files = files_by_name(mapping.normalize_detail(civitai_fixture("model_flux_inpainting"), "flux", False))
    assert files["flux1-fill-devFP8.safetensors"]["reason"] == "Inpainting checkpoints not supported"


def test_sft_extension_is_accepted():
    detail = mapping.normalize_detail(civitai_fixture("model_flux_bundled"), "flux", False)
    files = {f["name"]: f for v in detail["versions"] for f in v["files"]}
    assert files["flux1-schnell.sft"]["compatible"]


def test_vae_model_files_go_to_vaes():
    files = files_by_name(mapping.normalize_detail(civitai_fixture("model_sdxl_vae"), "sdxl", False))
    vae = files["sdxl_vae.safetensors"]
    assert vae["compatible"] and vae["target_resource_type"] == "vaes"


def test_lora_files_go_to_loras_with_hash_and_size():
    files = files_by_name(mapping.normalize_detail(civitai_fixture("model_sdxl_lora"), "sdxl", False))
    lora = files["add-detail-xl.safetensors"]
    assert lora["target_resource_type"] == "loras"
    assert lora["sha256"] == "0d9bd1b873a7863e128b4672e3e245838858f71469a3cec58123c16c06f83bd7"
    assert lora["size_bytes"] > 200_000_000


def test_base_model_mismatch_is_a_warning():
    files = files_by_name(mapping.normalize_detail(civitai_fixture("model_sdxl_lora"), "illustrious", False))
    lora = files["add-detail-xl.safetensors"]
    assert lora["compatible"]
    assert lora["warnings"] == ["Made for SDXL 1.0 (sdxl), not illustrious"]


def _first_file_and_version(fixture="model_sdxl_lora"):
    model = civitai_fixture(fixture)
    version = copy.deepcopy(model["modelVersions"][0])
    return model, version, version["files"][0]


@pytest.mark.parametrize("change,reason", [
    (lambda v, f: f["metadata"].update(format="PickleTensor"), "PickleTensor format not supported"),
    (lambda v, f: f.update(type="Training Data"), "Training Data files not supported"),
    (lambda v, f: f.update(hashes={}), "No SHA256 hash to verify the download"),
    (lambda v, f: v.update(usageControl="Generation"), "Generation-only on Civitai"),
    (lambda v, f: v.update(baseModel="SD 3.5"), "Base model SD 3.5 not supported"),
    (lambda v, f: f.update(name="model-bnb-4bit.safetensors"), "Pre-quantized NF4 not supported"),
])
def test_rejection_rules(change, reason):
    model, version, file = _first_file_and_version()
    change(version, file)
    result = mapping.file_compatibility(file, version, model["type"], "sdxl")
    assert not result.ok and result.reason == reason


def test_unsupported_model_type_is_rejected():
    model, version, file = _first_file_and_version()
    result = mapping.file_compatibility(file, version, "TextualInversion", "sdxl")
    assert not result.ok and result.reason == "TextualInversion models not supported"


def test_nf4_name_rule_does_not_match_inside_words():
    model, version, file = _first_file_and_version()
    file["name"] = "gnf4rce_bnbx.safetensors"
    assert mapping.file_compatibility(file, version, model["type"], "sdxl").ok


def test_vae_file_inside_checkpoint_version_goes_to_vaes():
    model, version, file = _first_file_and_version()
    file["type"] = "VAE"
    result = mapping.file_compatibility(file, version, "Checkpoint", "sdxl")
    assert result.ok and result.target_resource_type == "vaes"


def test_paid_access_is_a_warning():
    model, version, file = _first_file_and_version()
    version["paidAccess"] = {"permanent": True, "endsAt": None}
    result = mapping.file_compatibility(file, version, model["type"], "sdxl")
    assert result.ok and result.warnings == ["May require purchase or early access on Civitai"]


def test_search_summaries_keep_only_matching_models():
    data = civitai_fixture("search_query_illustrious_checkpoint")
    summaries = [mapping.normalize_summary(m, "illustrious", False) for m in data["items"]]
    assert all(summaries)
    assert {"Illustrious", "NoobAI"} >= {b for s in summaries for b in s["base_models"]}
    assert all(mapping.normalize_summary(m, "flux", False) is None for m in data["items"])


def test_detail_puts_matching_versions_first():
    model = civitai_fixture("model_sdxl_lora")
    model["modelVersions"][0]["baseModel"] = "SD 1.5"
    detail = mapping.normalize_detail(model, "sdxl", False)
    assert detail["versions"][0]["base_model"] == "SDXL 1.0"
    assert detail["versions"][-1]["base_model"] == "SD 1.5"


def test_preview_prefers_sfw_images():
    images = [
        {"url": "x", "type": "image", "nsfwLevel": 8},
        {"url": "v", "type": "video", "nsfwLevel": 1},
        {"url": "s", "type": "image", "nsfwLevel": 1},
    ]
    assert mapping.pick_preview(images, nsfw_allowed=False)["url"] == "s"
    assert mapping.pick_preview(images, nsfw_allowed=True)["url"] == "x"
    assert mapping.pick_preview(images[:2], nsfw_allowed=False)["url"] == "v"
    only_nsfw = mapping.pick_preview(images[:1], nsfw_allowed=False)
    assert only_nsfw["url"] == "x" and only_nsfw["nsfw_level"] == 8
    assert mapping.pick_preview([], nsfw_allowed=False) is None
