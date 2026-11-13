#!/usr/bin/env python
"""
Civitai Mapping

Maps this backend's model/resource types to Civitai's filters, decides which
Civitai files can be downloaded, and normalises Civitai API responses into
the shapes returned by the /civitai endpoints. Pure functions over Civitai's
JSON, checked against recorded API responses in tests/fixtures/civitai.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Set

logger = logging.getLogger(__name__)

# App resource_type -> Civitai model types
RESOURCE_CIVITAI_TYPES = {
    "checkpoints": ("Checkpoint",),
    "loras": ("LORA", "LoCon", "DoRA"),
    "vaes": ("VAE",),
}

# App model_type -> Civitai base models (checked against GET /api/v1/enums).
# Pony V7 is AuraFlow-based and Flux.1 Kontext/Flux.2/Flux.3 use different
# pipelines, so they're left out on purpose.
MODEL_BASE_MODELS = {
    "sd15": ("SD 1.4", "SD 1.5", "SD 1.5 LCM", "SD 1.5 Hyper"),
    "sdxl": (
        "SDXL 0.9", "SDXL 1.0", "SDXL 1.0 LCM", "SDXL Lightning", "SDXL Hyper",
        "SDXL Turbo", "SDXL Distilled", "Pony",
    ),
    "illustrious": ("Illustrious", "NoobAI"),
    "flux": ("Flux.1 D", "Flux.1 S", "Flux.1 Krea"),
}

SORT_OPTIONS = ("Highest Rated", "Most Downloaded", "Newest")
PERIOD_OPTIONS = ("AllTime", "Year", "Month", "Week", "Day")

# Checkpoint variants that don't load as a txt2img checkpoint
UNSUPPORTED_CHECKPOINT_VARIANTS = ("Inpainting", "Refiner", "Pix2Pix")

MODEL_FILE_TYPES = ("Model", "Pruned Model")

# Civitai image nsfwLevel is a bitmask: 1 = PG, 2 = PG-13, 4 = R, 8 = X, 16 = XXX
SFW_NSFW_LEVEL = 1

_NF4_NAME = re.compile(r"(^|[_.\-\s])(nf4|bnb)([_.\-\s]|$)", re.IGNORECASE)


def civitai_types(resource_type: str) -> List[str]:
    return list(RESOURCE_CIVITAI_TYPES[resource_type])


def base_models(model_type: str, known: Optional[Set[str]] = None) -> List[str]:
    """
    Civitai base models for model_type. When known (Civitai's current
    BaseModel enum) is given, names Civitai no longer recognises are dropped
    with a warning: Civitai silently returns no results for unknown names.
    """
    names = list(MODEL_BASE_MODELS[model_type])
    if known is None:
        return names
    missing = [name for name in names if name not in known]
    if missing:
        logger.warning(f"Civitai no longer lists base models {missing} for {model_type}; update MODEL_BASE_MODELS")
    return [name for name in names if name in known]


def model_type_for_base_model(base_model: Optional[str]) -> Optional[str]:
    for model_type, names in MODEL_BASE_MODELS.items():
        if base_model in names:
            return model_type
    return None


def resource_type_for_civitai_type(civitai_type: Optional[str]) -> Optional[str]:
    for resource_type, types in RESOURCE_CIVITAI_TYPES.items():
        if civitai_type in types:
            return resource_type
    return None


@dataclass
class Compatibility:
    """Whether a Civitai file can be downloaded into this backend."""

    ok: bool
    target_resource_type: Optional[str] = None
    reason: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


def file_compatibility(
    file: dict, version: dict, model_civitai_type: Optional[str], model_type: str
) -> Compatibility:
    """
    Check one Civitai file against the download rules.

    Args:
        file: Civitai file object
        version: Civitai model version object the file belongs to
        model_civitai_type: Civitai type of the model (e.g. "Checkpoint")
        model_type: App model type the user is browsing

    Returns:
        Compatibility with the resource directory the file would go to
    """
    metadata = file.get("metadata") or {}
    name = file.get("name") or ""
    file_type = file.get("type")

    # A VAE file attached to a checkpoint version goes to the VAE folder
    if file_type == "VAE":
        target = "vaes"
    elif file_type in MODEL_FILE_TYPES:
        target = resource_type_for_civitai_type(model_civitai_type)
        if target is None:
            return Compatibility(False, reason=f"{model_civitai_type} models not supported")
    else:
        return Compatibility(False, reason=f"{file_type} files not supported")

    def reject(reason: str) -> Compatibility:
        return Compatibility(False, target_resource_type=target, reason=reason)

    file_format = metadata.get("format")
    if file_format != "SafeTensor":
        return reject(f"{file_format or 'Unknown'} format not supported")
    if metadata.get("fp") == "nf4" or _NF4_NAME.search(name.rsplit(".", 1)[0]):
        return reject("Pre-quantized NF4 not supported")
    variant = version.get("baseModelType")
    if target == "checkpoints" and variant in UNSUPPORTED_CHECKPOINT_VARIANTS:
        return reject(f"{variant} checkpoints not supported")
    usage = version.get("usageControl")
    if usage and usage != "Download":
        return reject("Generation-only on Civitai")
    if not (file.get("hashes") or {}).get("SHA256"):
        return reject("No SHA256 hash to verify the download")

    base_model = version.get("baseModel")
    version_model_type = model_type_for_base_model(base_model)
    if version_model_type is None:
        return reject(f"Base model {base_model} not supported")

    warnings = []
    if version_model_type != model_type:
        warnings.append(f"Made for {base_model} ({version_model_type}), not {model_type}")
    if version.get("paidAccess") or version.get("availability") not in (None, "Public"):
        warnings.append("May require purchase or early access on Civitai")
    return Compatibility(True, target_resource_type=target, warnings=warnings)


def pick_preview(images: Iterable[dict], nsfw_allowed: bool) -> Optional[dict]:
    """
    Pick a preview: the first image (not video) allowed by the NSFW setting,
    else the first video, else the first image flagged so the UI blurs it.
    """
    images = [image for image in images or [] if image.get("url")]

    def preview(image: dict) -> dict:
        return {
            "url": image["url"],
            "type": "video" if image.get("type") == "video" else "image",
            "nsfw_level": image.get("nsfwLevel") or 0,
            "width": image.get("width"),
            "height": image.get("height"),
        }

    def allowed(image: dict) -> bool:
        return nsfw_allowed or (image.get("nsfwLevel") or 0) <= SFW_NSFW_LEVEL

    for kind in ("image", "video"):
        for image in images:
            if (image.get("type") or "image") == kind and allowed(image):
                return preview(image)
    return preview(images[0]) if images else None


def normalize_file(file: dict, version: dict, model_civitai_type: Optional[str], model_type: str) -> dict:
    metadata = file.get("metadata") or {}
    compatibility = file_compatibility(file, version, model_civitai_type, model_type)
    return {
        "id": file.get("id"),
        "name": file.get("name"),
        "size_bytes": int(round((file.get("sizeKB") or 0) * 1024)),
        "type": file.get("type"),
        "format": metadata.get("format"),
        "fp": metadata.get("fp"),
        "size_variant": metadata.get("size"),
        "sha256": ((file.get("hashes") or {}).get("SHA256") or "").lower() or None,
        "primary": bool(file.get("primary")),
        "target_resource_type": compatibility.target_resource_type,
        "compatible": compatibility.ok,
        "reason": compatibility.reason,
        "warnings": compatibility.warnings,
        "local_path": None,
        "local_match": None,
    }


def normalize_version(
    version: dict, model_civitai_type: Optional[str], model_type: str, nsfw_allowed: bool
) -> dict:
    return {
        "id": version.get("id"),
        "name": version.get("name"),
        "base_model": version.get("baseModel"),
        "base_model_type": version.get("baseModelType"),
        "model_type": model_type_for_base_model(version.get("baseModel")),
        "trained_words": version.get("trainedWords") or [],
        "published_at": version.get("publishedAt"),
        "early_access": bool(version.get("paidAccess")) or version.get("availability") not in (None, "Public"),
        "preview": pick_preview(version.get("images"), nsfw_allowed),
        "files": [
            normalize_file(file, version, model_civitai_type, model_type)
            for file in version.get("files") or []
        ],
    }


def _summary_fields(model: dict, versions: List[dict], nsfw_allowed: bool) -> dict:
    stats = model.get("stats") or {}
    preview = None
    for version in versions:
        preview = pick_preview(version.get("images"), nsfw_allowed)
        if preview:
            break
    return {
        "id": model.get("id"),
        "name": model.get("name"),
        "type": model.get("type"),
        "creator": (model.get("creator") or {}).get("username"),
        "nsfw": bool(model.get("nsfw")),
        "tags": model.get("tags") or [],
        "stats": {
            "downloads": stats.get("downloadCount", 0),
            "rating": stats.get("rating"),
            "rating_count": stats.get("ratingCount", 0),
            "thumbs_up": stats.get("thumbsUpCount", 0),
            "favorites": stats.get("favoriteCount", 0),
        },
        "preview": preview,
        "base_models": sorted({v.get("baseModel") for v in versions if v.get("baseModel")}),
    }


def _matching_versions(model: dict, model_type: str) -> List[dict]:
    return [
        version for version in model.get("modelVersions") or []
        if model_type_for_base_model(version.get("baseModel")) == model_type
    ]


def normalize_summary(model: dict, model_type: str, nsfw_allowed: bool) -> Optional[dict]:
    """Search result entry, or None if no version is for model_type."""
    versions = _matching_versions(model, model_type)
    if not versions:
        return None
    return _summary_fields(model, versions, nsfw_allowed)


def normalize_detail(model: dict, model_type: str, nsfw_allowed: bool) -> dict:
    """
    Model details. Versions for model_type come first; other versions are
    kept (their files carry a mismatch warning or an unsupported reason).
    """
    all_versions = model.get("modelVersions") or []
    matching = _matching_versions(model, model_type)
    ordered = matching + [v for v in all_versions if v not in matching]
    detail = _summary_fields(model, matching or all_versions, nsfw_allowed)
    detail["description"] = model.get("description") or ""
    detail["versions"] = [
        normalize_version(version, model.get("type"), model_type, nsfw_allowed) for version in ordered
    ]
    return detail
