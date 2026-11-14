#!/usr/bin/env python
"""
Flux Checkpoint Layout

Tells transformer-only Flux checkpoints apart from bundled ("all-in-one")
ones by their tensor names, read from the safetensors header only. Works on
local files and on headers peeked from Civitai before downloading.

Prefixes verified against the headers of real Civitai Flux files (key
lists in tests/fixtures/flux_headers).
"""

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from safetensors import safe_open

logger = logging.getLogger(__name__)

COMPONENT_PREFIXES = {
    "transformer": (
        "model.diffusion_model.double_blocks.",  # ComfyUI-style single file (most common)
        "double_blocks.",  # official BFL release
        "transformer_blocks.",  # diffusers layout
    ),
    "clip_l": ("text_encoders.clip_l.transformer.",),
    "t5xxl": ("text_encoders.t5xxl.transformer.",),
    "vae": ("vae.", "first_stage_model."),
}
# Only Flux.1-dev (and its finetunes) has a guidance embedder
DEV_PREFIXES = (
    "model.diffusion_model.guidance_in.",
    "guidance_in.",
    "time_text_embed.guidance_embedder.",
)

# Shared components directory entries a Flux pipeline needs, by layout key;
# None means always needed from the shared directory.
SHARED_COMPONENTS = {
    "tokenizer": None,
    "tokenizer_2": None,
    "scheduler": None,
    "text_encoder": "clip_l",
    "text_encoder_2": "t5xxl",
    "vae": "vae",
}


def classify_keys(keys: Iterable[str]) -> dict:
    """
    Classify a Flux checkpoint from its tensor names.

    Returns:
        {"transformer", "clip_l", "t5xxl", "vae": bool,
         "format": "bfl" | "diffusers" | None, "variant": "dev" | "schnell" | None}
    """
    keys = [key for key in keys if key != "__metadata__"]

    def has(prefixes: Tuple[str, ...]) -> bool:
        return any(key.startswith(prefixes) for key in keys)

    layout = {name: has(prefixes) for name, prefixes in COMPONENT_PREFIXES.items()}
    if layout["transformer"]:
        layout["format"] = "diffusers" if has(("transformer_blocks.",)) else "bfl"
        layout["variant"] = "dev" if has(DEV_PREFIXES) else "schnell"
    else:
        layout["format"] = None
        layout["variant"] = None
    return layout


def is_bundled(layout: dict) -> bool:
    """True if the checkpoint carries any component besides the transformer."""
    return any(layout.get(name) for name in ("clip_l", "t5xxl", "vae"))


_cache: Dict[Tuple[str, float], dict] = {}


def classify(path: Path) -> dict:
    """Classify a local Flux checkpoint (cached by path and mtime)."""
    path = Path(path)
    cache_key = (str(path.resolve()), path.stat().st_mtime)
    if cache_key not in _cache:
        with safe_open(str(path), "pt") as f:
            _cache[cache_key] = classify_keys(f.keys())
    return _cache[cache_key]


def missing_components(layout: dict, components_dir: str) -> List[str]:
    """
    Shared Flux components the checkpoint needs from components_dir (because
    it doesn't carry them) but that are missing there.
    """
    root = Path(components_dir)
    missing = []
    for name, provided_by in SHARED_COMPONENTS.items():
        if provided_by and layout.get(provided_by):
            continue
        folder = root / name
        if not folder.is_dir() or not any(folder.iterdir()):
            missing.append(name)
    return missing
