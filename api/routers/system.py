#!/usr/bin/env python
"""
System Router

Health check and system information endpoints.
"""

import asyncio

from fastapi import APIRouter

from api.models.system import HealthResponse, SystemInfo
from api.services.model_cache import get_cache_manager
from api.services.job_queue import get_job_queue
from libs.shared.utils import get_gpu
from libs.shared.telemetry import get_gpu_memory_stats, get_gpu_compute_stats, get_host_stats

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
    """Get system information including GPU, cache, queue, and host stats."""
    device, dtype = get_gpu()
    cache_manager = get_cache_manager()
    job_queue = get_job_queue()

    loop = asyncio.get_event_loop()
    gpu_stats, compute_stats, host_stats = await asyncio.gather(
        loop.run_in_executor(None, get_gpu_memory_stats, device),
        loop.run_in_executor(None, get_gpu_compute_stats, device),
        loop.run_in_executor(None, get_host_stats),
    )
    gpu_stats["compute"] = compute_stats

    return SystemInfo(
        device=device,
        dtype=str(dtype),
        cache_stats=cache_manager.get_stats(),
        queue_stats=job_queue.get_stats(),
        gpu_stats=gpu_stats,
        host_stats=host_stats,
    )
