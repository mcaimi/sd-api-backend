#!/usr/bin/env python
"""
Global Constants and Configuration

Contains application-wide constants, scheduler definitions,
and enumeration types.
"""

from enum import Enum
from typing import Dict, Type

from diffusers.schedulers import (
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    KDPM2DiscreteScheduler,
    HeunDiscreteScheduler,
    LMSDiscreteScheduler,
    FlowMatchEulerDiscreteScheduler,
    FlowMatchHeunDiscreteScheduler,
)


# Default paths
DEFAULT_MODELS_PATH = "models/stable-diffusion"
DEFAULT_LORA_PATH = "models/lora"

# Safetensors header length (8 bytes for metadata length field)
SFT_HEADER_LEN = 8

# Random seed generation bit length
RANDOM_BIT_LENGTH = 32


# Available noise schedulers
schedulers: Dict[str, Type] = {
    "DPM++ 2M": DPMSolverMultistepScheduler,
    "DPM++ SDE": DPMSolverSinglestepScheduler,
    "DPM2": KDPM2DiscreteScheduler,
    "Euler a": EulerAncestralDiscreteScheduler,
    "Euler": EulerDiscreteScheduler,
    "Heun": HeunDiscreteScheduler,
    "LMS": LMSDiscreteScheduler,
}


# Flow-matching schedulers for Flux (incompatible with the discrete-noise
# schedulers above)
flux_schedulers: Dict[str, Type] = {
    "FlowMatch Euler": FlowMatchEulerDiscreteScheduler,
    "FlowMatch Heun": FlowMatchHeunDiscreteScheduler,
}


class MergeMethod(Enum):
    """
    Enumeration of available model merge methods.

    LINEAR: Classic weighted average interpolation
    SLERP: Spherical linear interpolation (better preserves model characteristics)
    ADDITIVE: Add difference from target model to base
    SUBTRACT: Remove features of target model from base
    """
    LINEAR = "linear"
    SLERP = "slerp"
    ADDITIVE = "additive"
    SUBTRACT = "subtract"

    @classmethod
    def from_string(cls, value: str) -> "MergeMethod":
        """Create MergeMethod from string value."""
        for method in cls:
            if method.value == value.lower():
                return method
        raise ValueError(f"Unknown merge method: {value}")
