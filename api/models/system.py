#!/usr/bin/env python
"""
Pydantic Models for System Endpoints

Request and response models for system information and health check.
"""

from pydantic import BaseModel
from typing import Dict, List, Any


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    service: str


class SystemInfo(BaseModel):
    """System information response."""
    device: str
    dtype: str
    cache_stats: Dict[str, Any]
    queue_stats: Dict[str, Any]


class ModelListResponse(BaseModel):
    """Response containing list of available models."""
    model_type: str
    resource_type: str
    models: List[Dict[str, str]]


class ModelMetadataResponse(BaseModel):
    """Response containing model metadata."""
    name: str
    path: str
    metadata: Dict[str, Any]
