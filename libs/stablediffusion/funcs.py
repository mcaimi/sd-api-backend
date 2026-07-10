#!/usr/bin/env python
"""
Tensor Manipulation Functions for Model Merging

Provides:
    - Spherical Tensor Interpolation (SLERP)
    - Linear Tensor Interpolation
    - Unified Merge Tool with multiple methods
    - Random seed generation
"""

import logging
import random

import torch
import torch.nn.functional as F

from libs.globals.vars import MergeMethod, RANDOM_BIT_LENGTH
from libs.shared.exceptions import MergeError

logger = logging.getLogger(__name__)


def get_random_seed(seed_len: int = RANDOM_BIT_LENGTH) -> int:
    """
    Generate a random seed value.
    
    Args:
        seed_len: Number of random bits to generate
        
    Returns:
        Random integer seed
    """
    return random.getrandbits(seed_len)


def slerp(t0: torch.Tensor, t1: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Spherical linear interpolation between two tensors.
    
    SLERP interpolates along the surface of a hypersphere, which can
    preserve the magnitude and structure of neural network weights better
    than linear interpolation for some merge operations.
    
    Args:
        t0: First tensor (at alpha=0)
        t1: Second tensor (at alpha=1)
        alpha: Interpolation factor (0.0 to 1.0)
        
    Returns:
        Interpolated tensor with same shape as inputs
        
    Raises:
        AssertionError: If tensors have different shapes
    """
    # Save original shape and validate
    original_shape = t0.shape
    assert original_shape == t1.shape, "Tensors must have the same shape"
    
    # Flatten tensors for calculation
    t0_flat = t0.flatten()
    t1_flat = t1.flatten()
    
    # Normalize vectors (L2 norm)
    t0_norm = F.normalize(t0_flat, dim=0)
    t1_norm = F.normalize(t1_flat, dim=0)
    
    # Calculate angle between vectors
    dot_product = torch.dot(t0_norm, t1_norm)
    dot_product = torch.clamp(dot_product, -1.0, 1.0)
    omega = torch.acos(dot_product)
    
    # Handle parallel vectors (fall back to linear interpolation)
    if omega.abs() < 1e-6:
        return t0 * (1 - alpha) + t1 * alpha
    
    # Spherical interpolation
    sin_omega = torch.sin(omega)
    result = (
        (torch.sin((1 - alpha) * omega) / sin_omega) * t0_flat +
        (torch.sin(alpha * omega) / sin_omega) * t1_flat
    )
    
    return result.reshape(original_shape)


def linear(t0: torch.Tensor, t1: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Linear interpolation between two tensors.
    
    Args:
        t0: First tensor (weighted by alpha)
        t1: Second tensor (weighted by 1-alpha)
        alpha: Interpolation factor (0.0 to 1.0)
        
    Returns:
        Interpolated tensor
        
    Raises:
        AssertionError: If tensors have different shapes
    """
    assert t0.shape == t1.shape, "Tensors must have the same shape"
    return (t0 * alpha) + (t1 * (1 - alpha))


def merge_tensors(
    tensor_a: torch.Tensor,
    tensor_b: torch.Tensor,
    method: MergeMethod,
    alpha: float,
    **kwargs,
) -> torch.Tensor:
    """
    Merge two tensors using the specified method.
    
    Args:
        tensor_a: Base tensor
        tensor_b: Target tensor
        method: Merge method to use
        alpha: Merge strength (interpretation depends on method)
        **kwargs: Additional arguments:
            - base_tensor: Reference tensor for ADDITIVE/SUBTRACT methods
            
    Returns:
        Merged tensor with original dtype
        
    Raises:
        MergeError: If an unsupported merge method is specified
    """
    # Preserve original dtype and convert to float32 for calculation
    original_dtype = tensor_a.dtype
    tensor_a = tensor_a.to(torch.float32)
    tensor_b = tensor_b.to(torch.float32)
    
    if method == MergeMethod.LINEAR:
        result = linear(tensor_a, tensor_b, alpha)
        
    elif method == MergeMethod.SLERP:
        result = slerp(tensor_a, tensor_b, alpha)
        
    elif method == MergeMethod.ADDITIVE:
        # Add the difference from base model
        base_tensor = kwargs.get("base_tensor", tensor_a).to(torch.float32)
        diff = tensor_b - base_tensor
        result = tensor_a + diff * alpha
        
    elif method == MergeMethod.SUBTRACT:
        # Subtract features from base model
        base_tensor = kwargs.get("base_tensor", tensor_a).to(torch.float32)
        diff = tensor_b - base_tensor
        result = tensor_a - diff * alpha
        
    else:
        raise MergeError(f"Unsupported merge method: {method}")
    
    return result.to(original_dtype)
