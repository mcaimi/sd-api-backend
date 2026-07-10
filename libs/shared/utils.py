#!/usr/bin/env python
"""
Shared Utility Functions

Common utilities used across the application for file operations,
GPU detection, safetensors handling, etc.
"""

import json
import logging
import random
import string
import struct
from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple, Union

import torch
import torch.cuda as cuda
import torch.backends.mps as apple_mps

from libs.globals.vars import DEFAULT_MODELS_PATH, SFT_HEADER_LEN

logger = logging.getLogger(__name__)


def check_or_create_path(target_path: Union[str, Path]) -> None:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        target_path: Path to the directory
    """
    path = Path(target_path) if isinstance(target_path, str) else target_path
    if not path.is_dir():
        path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=6)
def enumerate_models(path: Union[str, Path] = DEFAULT_MODELS_PATH) -> Dict[str, Path]:
    """
    Enumerate all safetensors model files in a directory (cached).

    This function is cached to avoid repeated filesystem scans.
    Cache size of 6 supports sd15/sdxl × checkpoints/loras/vae.

    Args:
        path: Directory to search for models

    Returns:
        Dictionary mapping filename to full Path
    """
    # Convert to string for cache key compatibility
    model_path = Path(path) if isinstance(path, str) else path
    model_files = model_path.glob("**/*.safetensors")

    return {filepath.name: filepath for filepath in model_files}


def _read_safetensors_header_impl(filepath_str: str) -> Dict:
    """
    Internal implementation of safetensors header reading.

    Args:
        filepath_str: Absolute path string to the safetensors file

    Returns:
        Parsed metadata dictionary

    Raises:
        FileNotFoundError: If the file doesn't exist
        Exception: If file cannot be read or parsed
    """
    filepath = Path(filepath_str)

    if not filepath.exists():
        raise FileNotFoundError(f"Safetensors file not found: {filepath}")

    with open(filepath, "rb") as f:
        # Read header length (8 bytes, little-endian unsigned long long)
        header_bytes = f.read(SFT_HEADER_LEN)
        metadata_len = struct.unpack("<Q", header_bytes)[0]

        # Read and parse metadata
        metadata = f.read(metadata_len)
        return json.loads(metadata)


@lru_cache(maxsize=100)
def read_safetensors_header(filename: Union[str, Path]) -> Dict:
    """
    Read and parse the metadata header from a safetensors file (cached).

    This function is cached to avoid repeated file I/O for the same model.
    Cache size of 100 supports multiple model selections across pages.

    Args:
        filename: Path to the safetensors file

    Returns:
        Parsed metadata dictionary

    Raises:
        TypeError: If filename is not str or Path
        FileNotFoundError: If the file doesn't exist
        Exception: If file cannot be read or parsed
    """
    # Convert to absolute path string for cache key stability
    if isinstance(filename, str):
        filepath = Path(filename).absolute()
    elif isinstance(filename, Path):
        filepath = filename.absolute()
    else:
        raise TypeError(
            f"read_safetensors_header(): filename must be str or Path, got {type(filename).__name__}"
        )

    # Use string path for cache key (Path objects aren't hashable consistently)
    return _read_safetensors_header_impl(str(filepath))


@lru_cache(maxsize=1)
def get_gpu() -> Tuple[str, torch.dtype]:
    """
    Detect and return the best available compute device (cached).

    This function is cached to avoid repeated GPU detection calls.
    Results are static for the session lifetime.

    Returns:
        Tuple of (device_name, recommended_dtype)
        Device will be one of: "mps", "cuda", or "cpu"
    """
    accelerator = "cpu"
    dtype = torch.float16

    if apple_mps.is_available():
        logger.info("Apple Metal Performance Shaders Available!")
        accelerator = "mps"
    elif cuda.is_available():
        device_name = cuda.get_device_name()
        device_capabilities = cuda.get_device_capability()
        device_available_mem, device_total_mem = [
            x / 1024**3 for x in cuda.mem_get_info()
        ]
        logger.info(
            "GPU available: %s - %s - %.1f/%.1f GB VRAM",
            device_name,
            device_capabilities,
            device_available_mem,
            device_total_mem,
        )
        accelerator = "cuda"
    else:
        logger.warning("No GPU found. Using CPU (this will be slow).")
        dtype = torch.float32

    return accelerator, dtype


def random_string(length: int = 6) -> str:
    """
    Generate a random lowercase string.

    Args:
        length: Length of the string to generate

    Returns:
        Random string of specified length
    """
    return "".join(random.choice(string.ascii_lowercase) for _ in range(length))
