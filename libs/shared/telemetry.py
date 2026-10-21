#!/usr/bin/env python
"""
Telemetry Helpers

Backend-agnostic (mps/cuda/cpu) GPU memory/compute stats and host resource
stats, for the system telemetry endpoint. All functions are synchronous and
perform blocking calls (subprocess, psutil sampling) - callers should run
them via loop.run_in_executor().
"""

import logging
import os
import shutil
import subprocess

import psutil
import torch

logger = logging.getLogger(__name__)


def get_gpu_memory_stats(device: str) -> dict:
    """Get GPU memory usage for the active compute device."""
    if device == "cuda":
        free, total = torch.cuda.mem_get_info()
        return {
            "available": True,
            "used_bytes": total - free,
            "total_bytes": total,
            "allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
            "total_is_approximate": False,
        }

    if device == "mps":
        allocated = torch.mps.current_allocated_memory()
        driver_allocated = torch.mps.driver_allocated_memory()
        total = psutil.virtual_memory().total
        return {
            "available": True,
            "used_bytes": driver_allocated,
            "total_bytes": total,
            "allocated_bytes": allocated,
            "reserved_bytes": driver_allocated,
            "total_is_approximate": True,
        }

    return {"available": False, "reason": "no GPU detected (running on CPU)"}


def get_gpu_compute_stats(device: str) -> dict:
    """Get GPU compute utilization percentage, when the platform exposes one."""
    if device != "cuda":
        return {
            "available": False,
            "reason": f"GPU compute utilization is not exposed by PyTorch on '{device}'",
        }

    if not shutil.which("nvidia-smi"):
        return {"available": False, "reason": "nvidia-smi not found on PATH"}

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        )
        utilization = float(result.stdout.strip().splitlines()[0])
        return {"available": True, "utilization_percent": utilization}
    except Exception as e:
        logger.warning("Failed to query nvidia-smi: %s", e)
        return {"available": False, "reason": f"nvidia-smi query failed: {e}"}


def get_host_stats() -> dict:
    """Get host (node) CPU/RAM usage."""
    cpu_percent = psutil.cpu_percent(interval=0.1)
    mem = psutil.virtual_memory()

    try:
        load_avg = os.getloadavg()
    except (AttributeError, OSError):
        load_avg = None

    return {
        "cpu_percent": cpu_percent,
        "cpu_count": psutil.cpu_count(),
        "memory_used_bytes": mem.used,
        "memory_total_bytes": mem.total,
        "memory_percent": mem.percent,
        "load_average": load_avg,
    }
