#!/usr/bin/env python
"""
Pydantic Models for Image Generation

Request and response models for text-to-image and inpainting endpoints.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List

from libs.globals.vars import schedulers, flux_schedulers


class LoraConfig(BaseModel):
    """LoRA adapter configuration."""
    name: str = Field(..., description="LoRA filename")
    strength: float = Field(0.5, ge=0.0, le=1.0, description="LoRA merge strength")


class GenerationRequest(BaseModel):
    """Request for SD1.5/SDXL image generation."""
    positive_prompt: str = Field(..., min_length=1, max_length=2000, description="Positive prompt")
    negative_prompt: str = Field(default="", max_length=2000, description="Negative prompt")
    model_checkpoint: str = Field(..., description="Model checkpoint filename")
    width: int = Field(512, ge=256, le=2048, description="Image width")
    height: int = Field(512, ge=256, le=2048, description="Image height")
    steps: int = Field(30, ge=1, le=150, description="Inference steps")
    cfg_scale: float = Field(7.0, ge=1.0, le=30.0, description="Classifier-free guidance scale")
    seed: int = Field(-1, ge=-1, description="Random seed (-1 for random)")
    scheduler: str = Field("DPM++ 2M", description="Noise scheduler")
    loras: Optional[List[LoraConfig]] = Field(default=None, description="LoRA adapters")
    custom_vae: Optional[str] = Field(default=None, description="Custom VAE checkpoint")

    @field_validator('scheduler')
    def validate_scheduler(cls, v):
        if v not in schedulers:
            raise ValueError(f"Invalid scheduler: {v}. Must be one of: {list(schedulers.keys())}")
        return v


class InpaintRequest(BaseModel):
    """Request for SD1.5/SDXL inpainting."""
    positive_prompt: str = Field(..., min_length=1, max_length=2000)
    negative_prompt: str = Field(default="", max_length=2000)
    model_checkpoint: str
    image_base64: str = Field(..., description="Base64-encoded input image")
    mask_base64: str = Field(..., description="Base64-encoded mask image")
    steps: int = Field(20, ge=1, le=150)
    cfg_scale: float = Field(7.0, ge=1.0, le=30.0)
    seed: int = Field(-1, ge=-1)
    scheduler: str = Field("DPM++ 2M")
    loras: Optional[List[LoraConfig]] = None
    custom_vae: Optional[str] = None

    @field_validator('scheduler')
    def validate_scheduler(cls, v):
        if v not in schedulers:
            raise ValueError(f"Invalid scheduler: {v}")
        return v


class BatchGenerationRequest(BaseModel):
    """Request for batch generation."""
    requests: List[GenerationRequest] = Field(..., max_length=10, description="List of generation requests")


class CompareModelsRequest(BaseModel):
    """Request for model comparison."""
    positive_prompt: str
    negative_prompt: str = ""
    models: List[str] = Field(..., min_length=2, max_length=6, description="Model checkpoints to compare")
    width: int = Field(512, ge=256, le=2048)
    height: int = Field(512, ge=256, le=2048)
    steps: int = Field(30, ge=1, le=150)
    cfg_scale: float = Field(7.0, ge=1.0, le=30.0)
    seed: int = Field(42, ge=-1, description="Fixed seed for fair comparison")
    scheduler: str = Field("DPM++ 2M")

    @field_validator('scheduler')
    def validate_scheduler(cls, v):
        if v not in schedulers:
            raise ValueError(f"Invalid scheduler: {v}")
        return v


class FluxGenerationRequest(GenerationRequest):
    """Request for Flux image generation.

    Flux uses flow-matching schedulers (incompatible with the SD/SDXL
    scheduler set) and a distilled guidance scale, so scheduler validation
    and the cfg_scale default are overridden; everything else is shared.
    """
    scheduler: str = Field("FlowMatch Euler", description="Noise scheduler")
    cfg_scale: float = Field(3.5, ge=1.0, le=30.0, description="Flux guidance scale (distilled)")
    true_cfg_scale: float = Field(
        1.0, ge=1.0, le=10.0,
        description="True CFG scale; >1.0 enables negative_prompt",
    )

    @field_validator('scheduler')
    def validate_scheduler(cls, v):
        if v not in flux_schedulers:
            raise ValueError(f"Invalid scheduler: {v}. Must be one of: {list(flux_schedulers.keys())}")
        return v


class FluxInpaintRequest(InpaintRequest):
    """Request for Flux inpainting."""
    scheduler: str = Field("FlowMatch Euler")
    cfg_scale: float = Field(7.0, ge=1.0, le=30.0)
    strength: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Inpaint re-noising strength; None uses the pipeline default (0.6)",
    )
    true_cfg_scale: float = Field(1.0, ge=1.0, le=10.0)

    @field_validator('scheduler')
    def validate_scheduler(cls, v):
        if v not in flux_schedulers:
            raise ValueError(f"Invalid scheduler: {v}. Must be one of: {list(flux_schedulers.keys())}")
        return v


class FluxBatchGenerationRequest(BaseModel):
    """Request for batch Flux generation."""
    requests: List[FluxGenerationRequest] = Field(..., max_length=10, description="List of generation requests")


class FluxCompareModelsRequest(CompareModelsRequest):
    """Request for Flux model comparison."""
    scheduler: str = Field("FlowMatch Euler")

    @field_validator('scheduler')
    def validate_scheduler(cls, v):
        if v not in flux_schedulers:
            raise ValueError(f"Invalid scheduler: {v}. Must be one of: {list(flux_schedulers.keys())}")
        return v


class JobResponse(BaseModel):
    """Response containing job ID."""
    job_id: str
    status: str
    message: str


class JobStatusResponse(BaseModel):
    """Response containing job status and result."""
    job_id: str
    job_type: str
    status: str
    progress: float
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    result: Optional[dict] = None
    error: Optional[str] = None
