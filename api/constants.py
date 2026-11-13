#!/usr/bin/env python
"""
Shared Constants

Supported model architectures and resource types, plus request validators
for path parameters that name them.
"""

from fastapi import HTTPException

MODEL_TYPES = ("sd15", "sdxl", "illustrious", "flux")
RESOURCE_TYPES = ("checkpoints", "loras", "vaes")


def _quoted(values) -> str:
    return ", ".join(f"'{v}'" for v in values)


def validate_model_type(model_type: str) -> str:
    """Raise HTTP 400 unless model_type is a supported architecture."""
    if model_type not in MODEL_TYPES:
        raise HTTPException(
            status_code=400, detail=f"Invalid model_type. Must be one of {_quoted(MODEL_TYPES)}"
        )
    return model_type


def validate_resource_type(resource_type: str) -> str:
    """Raise HTTP 400 unless resource_type is a supported resource directory."""
    if resource_type not in RESOURCE_TYPES:
        raise HTTPException(
            status_code=400, detail=f"Invalid resource_type. Must be one of {_quoted(RESOURCE_TYPES)}"
        )
    return resource_type
