#!/usr/bin/env python
"""
Image Encoding Utilities

Base64 encoding/decoding for image transport over REST API.
"""

import base64
import numpy as np
from io import BytesIO
from PIL import Image


def numpy_to_base64(image_array: np.ndarray) -> str:
    """
    Convert numpy array to base64-encoded PNG string.

    Args:
        image_array: Numpy array with shape (H, W, C)

    Returns:
        Base64-encoded PNG string
    """
    # Convert to PIL Image
    if image_array.dtype != np.uint8:
        # Normalize to 0-255 range if needed
        if image_array.max() <= 1.0:
            image_array = (image_array * 255).astype(np.uint8)
        else:
            image_array = image_array.astype(np.uint8)

    image = Image.fromarray(image_array)

    # Encode to PNG in memory
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    # Base64 encode
    image_bytes = buffer.getvalue()
    base64_str = base64.b64encode(image_bytes).decode("utf-8")

    return base64_str


def base64_to_pil(base64_str: str) -> Image.Image:
    """
    Convert base64-encoded image string to PIL Image.

    Args:
        base64_str: Base64-encoded image string

    Returns:
        PIL Image object
    """
    # Decode base64
    image_bytes = base64.b64decode(base64_str)

    # Load as PIL Image
    image = Image.open(BytesIO(image_bytes))

    return image


def base64_to_numpy(base64_str: str) -> np.ndarray:
    """
    Convert base64-encoded image string to numpy array.

    Args:
        base64_str: Base64-encoded image string

    Returns:
        Numpy array with shape (H, W, C)
    """
    # Convert to PIL first
    image = base64_to_pil(base64_str)

    # Convert to numpy array
    image_array = np.array(image)

    return image_array


def pil_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """
    Convert PIL Image to base64-encoded string.

    Args:
        image: PIL Image object
        format: Image format (PNG, JPEG, etc.)

    Returns:
        Base64-encoded image string
    """
    buffer = BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)

    image_bytes = buffer.getvalue()
    base64_str = base64.b64encode(image_bytes).decode("utf-8")

    return base64_str
