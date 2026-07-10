#!/usr/bin/env python
"""
Custom Exception Classes

Provides domain-specific exceptions for the Stable Diffusion API.
"""


class CheckpointLoadError(Exception):
    """
    Exception raised when a checkpoint file cannot be loaded.

    This can occur due to:
    - File not found
    - Invalid file format
    - Corrupted checkpoint data
    - Insufficient memory
    """
    pass


class MergeError(Exception):
    """
    Exception raised when a merge operation fails.

    This can occur due to:
    - Incompatible model architectures
    - Shape mismatches between tensors
    - Unsupported merge method
    - Memory allocation failures
    """
    pass


class ConfigurationError(Exception):
    """
    Exception raised when configuration loading or validation fails.

    This can occur due to:
    - Missing configuration file
    - Invalid YAML syntax
    - Missing required configuration keys
    """
    pass


class PipelineError(Exception):
    """
    Exception raised when a pipeline operation fails.

    This can occur due to:
    - Model not loaded
    - Invalid inference parameters
    - Device/memory issues
    """
    pass
