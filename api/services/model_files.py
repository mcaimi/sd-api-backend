#!/usr/bin/env python
"""
Model Files

Helpers for model files on disk: safe file names for downloads, the
Civitai sidecar/preview files stored next to a model, and finding a model
already present in a resource directory.
"""

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

from metastable.models import enumerate_models

logger = logging.getLogger(__name__)

MODEL_SUFFIX = ".safetensors"
SIDECAR_SUFFIX = ".civitai.json"
PREVIEW_SUFFIX = ".preview.jpg"
PARTIAL_SUFFIX = ".part"
MAX_STEM_LENGTH = 150

_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_filename(name: str) -> str:
    """
    Turn a remote file name into a safe local .safetensors file name:
    no directories, only [A-Za-z0-9._-], bounded length. ".sft" files
    (safetensors under another extension) get the ".safetensors" suffix so
    enumerate_models() lists them.
    """
    base = Path(name.replace("\\", "/")).name
    stem = base
    for suffix in (MODEL_SUFFIX, ".sft"):
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    stem = _UNSAFE_CHARS.sub("_", stem).strip("._")[:MAX_STEM_LENGTH] or "model"
    return stem + MODEL_SUFFIX


def is_within(path: Path, root: Path) -> bool:
    """True if path resolves to a location inside root."""
    return path.resolve().is_relative_to(root.resolve())


def sidecar_path(model_path: Path) -> Path:
    return model_path.with_suffix(SIDECAR_SUFFIX)


def preview_path(model_path: Path) -> Path:
    return model_path.with_suffix(PREVIEW_SUFFIX)


def partial_path(model_path: Path) -> Path:
    return model_path.with_name(model_path.name + PARTIAL_SUFFIX)


def read_sidecar(model_path: Path) -> Optional[dict]:
    """The Civitai sidecar of a model, or None if missing or unreadable."""
    path = sidecar_path(model_path)
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as e:
        logger.warning(f"Ignoring unreadable sidecar {path}: {e}")
        return None


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Lowercase hex SHA256 of a file, read in chunks (blocking: run in a thread)."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def write_sidecar(model_path: Path, data: dict) -> None:
    with open(sidecar_path(model_path), "w") as f:
        json.dump(data, f, indent=2)


class LocalModelIndex:
    """
    Models present anywhere under one resource directory, indexed by file
    name and by the SHA256 recorded in their Civitai sidecars.
    """

    def __init__(self, models_dir: str):
        self.models_dir = models_dir
        self.by_name: Dict[str, Path] = enumerate_models(models_dir)
        self.by_sha256: Dict[str, Path] = {}
        for path in self.by_name.values():
            sidecar = read_sidecar(path)
            if sidecar and sidecar.get("sha256"):
                self.by_sha256[str(sidecar["sha256"]).lower()] = path

    def lookup(self, filename: str, sha256: Optional[str] = None) -> Tuple[Optional[Path], Optional[str]]:
        """
        Returns:
            (path, match) where match is "sha256" when a sidecar records the
            same hash, "name" when only the file name matches, or (None, None)
        """
        if sha256 and sha256.lower() in self.by_sha256:
            return self.by_sha256[sha256.lower()], "sha256"
        if filename in self.by_name:
            return self.by_name[filename], "name"
        return None, None
