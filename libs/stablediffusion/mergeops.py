#!/usr/bin/env python
"""
Merge Operations for Stable Diffusion Checkpoints

Provides low-level functions for:
- Loading checkpoints
- Validating checkpoint compatibility
- Merging checkpoint dictionaries
- Saving merged results
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from safetensors.torch import load_file, save_file

from libs.globals.vars import MergeMethod
from libs.stablediffusion.funcs import merge_tensors
from libs.shared.exceptions import CheckpointLoadError, MergeError

logger = logging.getLogger(__name__)


@dataclass
class MergeConfig:
    """Configuration for merge operations."""
    method: MergeMethod = MergeMethod.LINEAR
    alpha: float = 0.5
    device: str = "cpu"
    preserve_metadata: bool = True
    chunk_size: Optional[int] = None  # For memory-efficient processing
    progress_callback: Optional[Callable[[float], None]] = None


def load_checkpoint_dict(
    checkpoint_file: Union[str, Path],
    device: str = "cpu",
) -> Tuple[Dict[str, torch.Tensor], Dict]:
    """
    Load a checkpoint file and return weights and metadata.
    
    Args:
        checkpoint_file: Path to the safetensors checkpoint
        device: Device to load tensors to
        
    Returns:
        Tuple of (weights_dict, metadata_dict)
        
    Raises:
        CheckpointLoadError: If the checkpoint cannot be loaded
    """
    checkpoint_path = Path(checkpoint_file) if isinstance(checkpoint_file, str) else checkpoint_file
    
    if not checkpoint_path.exists():
        raise CheckpointLoadError(f"Checkpoint file not found: {checkpoint_path}")
    
    if not checkpoint_path.suffix == ".safetensors":
        raise CheckpointLoadError(
            f"Only safetensors format supported, got: {checkpoint_path.suffix}"
        )
    
    try:
        logger.info("Loading checkpoint: %s", checkpoint_path)
        weights = load_file(checkpoint_path, device=device)

        # Try to load associated metadata file
        metadata = {}
        metadata_file = checkpoint_path.with_suffix(".json")
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                metadata = json.load(f)

        return weights, metadata

    except Exception as e:
        logger.error("Failed to load checkpoint %s: %s", checkpoint_path, e)
        raise CheckpointLoadError(
            f"Failed to load checkpoint {checkpoint_path}: {e}"
        )


def validate_checkpoints(
    base_ckpt: Dict[str, torch.Tensor],
    *other_ckpts: Dict[str, torch.Tensor],
) -> Tuple[bool, List[str]]:
    """
    Validate that checkpoints are compatible for merging.

    Args:
        base_ckpt: Base checkpoint dictionary
        *other_ckpts: Additional checkpoints to validate against base

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    base_keys = set(base_ckpt.keys())

    for i, ckpt in enumerate(other_ckpts):
        ckpt_keys = set(ckpt.keys())

        # Check for missing/extra keys
        missing_in_base = ckpt_keys - base_keys
        missing_in_ckpt = base_keys - ckpt_keys

        if missing_in_base:
            error_msg = f"Checkpoint {i + 1} has extra keys: {list(missing_in_base)[:5]}..."
            errors.append(error_msg)
            logger.warning(error_msg)
        if missing_in_ckpt:
            error_msg = f"Checkpoint {i + 1} missing keys: {list(missing_in_ckpt)[:5]}..."
            errors.append(error_msg)
            logger.warning(error_msg)

        # Check tensor shapes for matching keys
        for key in base_keys.intersection(ckpt_keys):
            if base_ckpt[key].shape != ckpt[key].shape:
                error_msg = (
                    f"Shape mismatch for key '{key}': "
                    f"{base_ckpt[key].shape} vs {ckpt[key].shape}"
                )
                errors.append(error_msg)
                logger.error(error_msg)

    is_valid = len(errors) == 0
    if is_valid:
        logger.debug("Checkpoint validation passed")
    else:
        logger.error("Checkpoint validation failed with %d errors", len(errors))

    return is_valid, errors


def merge_checkpoints(
    base_ckpt: Dict[str, torch.Tensor],
    target_ckpt: Dict[str, torch.Tensor],
    config: MergeConfig,
    base_reference: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Merge two checkpoints using the specified configuration.
    
    Args:
        base_ckpt: Base checkpoint dictionary
        target_ckpt: Target checkpoint dictionary
        config: Merge configuration
        base_reference: Optional reference for additive/subtractive merges
        
    Returns:
        Merged checkpoint dictionary
        
    Raises:
        MergeError: If checkpoints are incompatible or merge fails
    """
    # Ensure alpha is a float
    if isinstance(config.alpha, str):
        config.alpha = float(config.alpha)
    
    # Validate checkpoints
    is_valid, errors = validate_checkpoints(base_ckpt, target_ckpt)
    if not is_valid:
        raise MergeError(f"Checkpoint validation failed: {errors}")
    
    merged_dict = {}
    total_keys = len(base_ckpt)
    
    for i, key in enumerate(base_ckpt.keys()):
        # Report progress
        if config.progress_callback:
            config.progress_callback(i / total_keys)
        
        tensor_a = base_ckpt[key]
        tensor_b = target_ckpt[key]
        
        # Prepare additional kwargs for some merge methods
        merge_kwargs = {}
        if config.method in [MergeMethod.ADDITIVE, MergeMethod.SUBTRACT]:
            if base_reference:
                merge_kwargs["base_tensor"] = base_reference[key]
        
        # Merge this tensor
        merged_dict[key] = merge_tensors(
            tensor_a, tensor_b, config.method, config.alpha, **merge_kwargs
        )
    
    # Final progress update
    if config.progress_callback:
        config.progress_callback(1.0)
    
    return merged_dict


def save_checkpoint(
    weights: Dict[str, torch.Tensor],
    filename: Union[str, Path],
    metadata: Optional[Dict] = None,
    backup_existing: bool = True,
) -> bool:
    """
    Save a checkpoint dictionary to disk.
    
    Args:
        weights: Checkpoint weights dictionary
        filename: Output path
        metadata: Optional metadata to save alongside
        backup_existing: Whether to backup existing file
        
    Returns:
        True if successful, False otherwise
    """
    filepath = Path(filename) if isinstance(filename, str) else filename
    
    try:
        # Backup existing file if requested
        if backup_existing and filepath.exists():
            backup_path = filepath.with_suffix(
                f".backup_{int(time.time())}.safetensors"
            )
            filepath.rename(backup_path)
            logger.info("Created backup: %s", backup_path)

        # Save weights
        logger.info("Saving checkpoint: %s", filepath)
        save_file(weights, filepath)

        # Save metadata if provided
        if metadata:
            metadata_file = filepath.with_suffix(".json")
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)
            logger.debug("Saved metadata: %s", metadata_file)

        return True

    except Exception as e:
        logger.error("Failed to save checkpoint: %s", e)
        return False


def batch_merge_checkpoints(
    base_checkpoint: Union[str, Path],
    checkpoint_list: List[Union[str, Path]],
    output_dir: Union[str, Path],
    config: MergeConfig,
) -> List[str]:
    """
    Merge a base checkpoint with multiple other checkpoints.
    
    Args:
        base_checkpoint: Path to base checkpoint
        checkpoint_list: List of checkpoint paths to merge with base
        output_dir: Directory for output files
        config: Merge configuration
        
    Returns:
        List of successfully created output file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load base checkpoint
    base_weights, base_metadata = load_checkpoint_dict(base_checkpoint, config.device)
    
    results = []
    
    for i, target_checkpoint in enumerate(checkpoint_list):
        try:
            logger.info("Merging %d/%d: %s", i + 1, len(checkpoint_list), target_checkpoint)

            # Load target checkpoint
            target_weights, target_metadata = load_checkpoint_dict(
                target_checkpoint, config.device
            )

            # Perform merge
            merged_weights = merge_checkpoints(base_weights, target_weights, config)

            # Create output filename
            target_path = Path(target_checkpoint)
            output_filename = output_path / f"merged_{target_path.stem}.safetensors"

            # Create metadata
            merged_metadata = {
                "base_model": str(base_checkpoint),
                "target_model": str(target_checkpoint),
                "merge_method": config.method.value,
                "merge_alpha": config.alpha,
                "timestamp": time.time(),
                "base_metadata": base_metadata,
                "target_metadata": target_metadata,
            }

            # Save result
            if save_checkpoint(merged_weights, output_filename, merged_metadata):
                results.append(str(output_filename))
                logger.info("Successfully saved: %s", output_filename)
            else:
                logger.error("Failed to save: %s", output_filename)

        except Exception as e:
            logger.error("Error merging %s: %s", target_checkpoint, e)
    
    return results


def merge_checkpoints_chunked(
    base_ckpt: Dict[str, torch.Tensor],
    target_ckpt: Dict[str, torch.Tensor],
    config: MergeConfig,
    chunk_size: int = 100,
) -> Dict[str, torch.Tensor]:
    """
    Memory-efficient merge processing in chunks.
    
    Useful for large models on systems with limited memory.
    
    Args:
        base_ckpt: Base checkpoint dictionary
        target_ckpt: Target checkpoint dictionary
        config: Merge configuration
        chunk_size: Number of tensors to process per chunk
        
    Returns:
        Merged checkpoint dictionary
    """
    keys = list(base_ckpt.keys())
    merged_dict = {}
    
    total_chunks = (len(keys) + chunk_size - 1) // chunk_size
    
    for chunk_idx in range(0, len(keys), chunk_size):
        chunk_num = chunk_idx // chunk_size + 1
        chunk_keys = keys[chunk_idx:chunk_idx + chunk_size]
        logger.debug("Processing chunk %d/%d", chunk_num, total_chunks)

        # Create chunk dictionaries
        base_chunk = {k: base_ckpt[k] for k in chunk_keys}
        target_chunk = {k: target_ckpt[k] for k in chunk_keys}
        
        # Create config without progress callback for chunks
        chunk_config = MergeConfig(
            method=config.method,
            alpha=config.alpha,
            device=config.device,
        )
        
        # Merge chunk
        merged_chunk = merge_checkpoints(base_chunk, target_chunk, chunk_config)
        merged_dict.update(merged_chunk)
        
        # Free memory
        del base_chunk, target_chunk, merged_chunk
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return merged_dict
