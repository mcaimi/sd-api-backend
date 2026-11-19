#!/usr/bin/env python
"""
Pydantic Models for Model Merging

Request and response models for model merging endpoints.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Literal

from metastable.merge import ARCHITECTURES, MergeMethod

# Model types metastable can merge (sd15, sdxl, illustrious, flux, zimage)
MERGE_MODEL_TYPES = tuple(ARCHITECTURES)

OutputDtype = Literal["auto", "bfloat16", "float16", "float32"]

_MODEL_TYPE_FIELD = f"Model type: {', '.join(repr(t) for t in MERGE_MODEL_TYPES)}"
_OUTPUT_DTYPE_FIELD = (
    "dtype of the merged tensors; 'auto' keeps each tensor's dtype from the base model"
)


class MergeTypeValidated(BaseModel):
    """Validates `model_type` against the architectures metastable can merge."""

    @field_validator('model_type', check_fields=False)
    @classmethod
    def validate_model_type(cls, v):
        if v not in MERGE_MODEL_TYPES:
            raise ValueError(f"model_type must be one of {', '.join(MERGE_MODEL_TYPES)}, got: {v}")
        return v


class MergeMethodValidated(BaseModel):
    """Validates `method` against metastable's merge methods."""

    @field_validator('method', check_fields=False)
    @classmethod
    def validate_method(cls, v):
        try:
            MergeMethod.from_string(v)
        except ValueError:
            raise ValueError(f"Invalid merge method: {v}")
        return v


class MergeRequest(MergeTypeValidated, MergeMethodValidated):
    """Request for model merging."""
    model_type: str = Field(..., description=_MODEL_TYPE_FIELD)
    base_model: str = Field(..., description="Base model checkpoint filename")
    target_model: str = Field(..., description="Target model checkpoint filename")
    method: str = Field("linear", description="Merge method (linear, slerp, additive, subtract)")
    alpha: float = Field(0.5, ge=0.0, le=1.0, description="Weight of the target model (0.0 = base, 1.0 = target)")
    output_name: str = Field(..., description="Output checkpoint filename")
    output_dtype: OutputDtype = Field("auto", description=_OUTPUT_DTYPE_FIELD)
    preserve_metadata: bool = Field(True, description="Preserve model metadata")


class BatchMergeRequest(MergeTypeValidated, MergeMethodValidated):
    """Request for batch merging."""
    model_type: str = Field(..., description=_MODEL_TYPE_FIELD)
    base_model: str
    target_models: List[str] = Field(..., min_length=1, description="List of target models")
    method: str = Field("linear")
    alpha: float = Field(0.5, ge=0.0, le=1.0)
    output_subdir: str = Field("batch_merged", description="Output subdirectory")
    output_dtype: OutputDtype = Field("auto", description=_OUTPUT_DTYPE_FIELD)
    preserve_metadata: bool = True


class MergeRecipeStep(MergeMethodValidated):
    """Single step in a merge recipe."""
    target_model: str
    method: str
    alpha: float = Field(ge=0.0, le=1.0)


class MergeRecipeRequest(MergeTypeValidated):
    """Request for executing a merge recipe."""
    model_type: str = Field(..., description=_MODEL_TYPE_FIELD)
    base_model: str
    steps: List[MergeRecipeStep] = Field(..., min_length=1, description="Merge recipe steps")
    output_name: str
    output_dtype: OutputDtype = Field("auto", description=_OUTPUT_DTYPE_FIELD)


class MergeResponse(BaseModel):
    """Response from merge operation."""
    job_id: str
    status: str
    message: str
