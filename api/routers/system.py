#!/usr/bin/env python
"""
System Router

Health check and system information endpoints.
"""

from fastapi import APIRouter

from api.models.system import HealthResponse, SystemInfo
from api.services.model_cache import get_cache_manager
from api.services.job_queue import get_job_queue
from libs.shared.utils import get_gpu

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        service="sd-api-backend"
    )


@router.get("/info", response_model=SystemInfo)
async def system_info():
    """Get system information including GPU, cache, and queue stats."""
    device, dtype = get_gpu()
    cache_manager = get_cache_manager()
    job_queue = get_job_queue()

    return SystemInfo(
        device=device,
        dtype=str(dtype),
        cache_stats=cache_manager.get_stats(),
        queue_stats=job_queue.get_stats()
    )
