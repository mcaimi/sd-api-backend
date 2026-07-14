#!/usr/bin/env python
"""
Pydantic Models for Model Merging

Request and response models for model merging endpoints.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List

from libs.globals.vars import MergeMethod


class MergeRequest(BaseModel):
    """Request for model merging."""
    base_model: str = Field(..., description="Base model checkpoint filename")
    target_model: str = Field(..., description="Target model checkpoint filename")
    method: str = Field("linear", description="Merge method (linear, slerp, additive, subtract)")
    alpha: float = Field(0.5, ge=0.0, le=1.0, description="Merge strength/alpha")
    output_name: str = Field(..., description="Output checkpoint filename")
    preserve_metadata: bool = Field(True, description="Preserve model metadata")

    @field_validator('method')
    def validate_method(cls, v):
        try:
            MergeMethod.from_string(v)
        except ValueError:
            raise ValueError(f"Invalid merge method: {v}")
        return v


class BatchMergeRequest(BaseModel):
    """Request for batch merging."""
    base_model: str
    target_models: List[str] = Field(..., min_length=1, description="List of target models")
    method: str = Field("linear")
    alpha: float = Field(0.5, ge=0.0, le=1.0)
    output_subdir: str = Field("batch_merged", description="Output subdirectory")
    preserve_metadata: bool = True

    @field_validator('method')
    def validate_method(cls, v):
        try:
            MergeMethod.from_string(v)
        except ValueError:
            raise ValueError(f"Invalid merge method: {v}")
        return v


class MergeRecipeStep(BaseModel):
    """Single step in a merge recipe."""
    target_model: str
    method: str
    alpha: float = Field(ge=0.0, le=1.0)

    @field_validator('method')
    def validate_method(cls, v):
        try:
            MergeMethod.from_string(v)
        except ValueError:
            raise ValueError(f"Invalid merge method: {v}")
        return v


class MergeRecipeRequest(BaseModel):
    """Request for executing a merge recipe."""
    base_model: str
    steps: List[MergeRecipeStep] = Field(..., min_length=1, description="Merge recipe steps")
    output_name: str


class MergeResponse(BaseModel):
    """Response from merge operation."""
    job_id: str
    status: str
    message: str
