#!/usr/bin/env python
"""
Jobs Router

Job management endpoints for querying and cancelling jobs.
"""

from typing import List
from fastapi import APIRouter, HTTPException

from api.models.generation import JobStatusResponse
from api.services.job_queue import get_job_queue

router = APIRouter()


@router.get("/", response_model=List[JobStatusResponse])
async def list_jobs():
    """List all jobs with their current status."""
    job_queue = get_job_queue()
    jobs = await job_queue.list_jobs()
    return jobs


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get job status and result."""
    job_queue = get_job_queue()
    job_info = await job_queue.get_job_status(job_id)

    if not job_info:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobStatusResponse(**job_info)


@router.delete("/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a pending or running job."""
    job_queue = get_job_queue()
    cancelled = await job_queue.cancel_job(job_id)

    if not cancelled:
        raise HTTPException(
            status_code=400,
            detail="Job cannot be cancelled (already completed or failed)",
        )

    return {"message": "Job cancelled successfully", "job_id": job_id}
