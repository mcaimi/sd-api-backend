#!/usr/bin/env python
"""
Pydantic Models for System Endpoints

Request and response models for system information and health check.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    service: str


class SystemInfo(BaseModel):
    """System information response."""

    device: str
    dtype: str
    vendor: Optional[str] = None
    gpu_name: Optional[str] = None
    cache_stats: Dict[str, Any]
    queue_stats: Dict[str, Any]
    gpu_stats: Dict[str, Any]
    host_stats: Dict[str, Any]


class ModelCivitaiInfo(BaseModel):
    """Civitai details of a local model, from its .civitai.json sidecar."""

    model_id: Optional[int] = None
    version_id: Optional[int] = None
    model_name: Optional[str] = None
    version_name: Optional[str] = None
    base_model: Optional[str] = None
    trained_words: List[str] = []
    nsfw: bool = False
    url: Optional[str] = None


class ModelEntry(BaseModel):
    """A local model file."""

    name: str
    path: str
    civitai: Optional[ModelCivitaiInfo] = None
    has_preview: bool = False
    flux_layout: Optional[Dict[str, Any]] = Field(
        None, description="Flux checkpoints only: components bundled in the file"
    )


class ModelListResponse(BaseModel):
    """Response containing list of available models."""

    model_type: str
    resource_type: str
    models: List[ModelEntry]


class ModelMetadataResponse(BaseModel):
    """Response containing model metadata."""

    name: str
    path: str
    metadata: Dict[str, Any]
