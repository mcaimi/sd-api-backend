#!/usr/bin/env python
"""
Async Job Queue Implementation

In-memory async job queue with background workers for long-running operations.
"""

import asyncio
import uuid
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Optional, Callable, Any, List

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """Job execution status."""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Job:
    """Represents a job in the queue."""

    def __init__(self, job_id: str, job_type: str, params: dict, callback: Callable):
        self.job_id = job_id
        self.job_type = job_type  # "generation", "merge", etc.
        self.params = params
        self.callback = callback
        self.status = JobStatus.QUEUED
        self.created_at = datetime.now()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.result: Optional[Any] = None
        self.error: Optional[str] = None
        self.progress: float = 0.0


class JobQueue:
    """In-memory async job queue with background workers."""

    def __init__(self, num_workers: int = 1, result_ttl: int = 3600):
        """
        Initialize job queue.

        Args:
            num_workers: Number of background worker tasks
            result_ttl: Time to live for completed jobs (seconds)
        """
        self.num_workers = num_workers
        self.result_ttl = result_ttl
        self.queue: asyncio.Queue = asyncio.Queue()
        self.jobs: Dict[str, Job] = {}
        self.workers: List[asyncio.Task] = []
        self._lock = asyncio.Lock()
        self._running = False

    async def submit_job(self, job_type: str, params: dict, callback: Callable) -> str:
        """
        Submit a job and return job ID.

        Args:
            job_type: Type of job (for logging/tracking)
            params: Parameters to pass to callback
            callback: Async function to execute

        Returns:
            Job ID string
        """
        job_id = str(uuid.uuid4())
        job = Job(job_id, job_type, params, callback)

        async with self._lock:
            self.jobs[job_id] = job

        await self.queue.put(job)
        logger.info(f"Job {job_id} ({job_type}) submitted")
        return job_id

    async def get_job_status(self, job_id: str) -> Optional[dict]:
        """
        Get job status and result.

        Args:
            job_id: Job ID to query

        Returns:
            Dictionary with job status or None if not found
        """
        async with self._lock:
            job = self.jobs.get(job_id)
            if not job:
                return None

            return {
                "job_id": job.job_id,
                "job_type": job.job_type,
                "status": job.status.value,
                "progress": job.progress,
                "created_at": job.created_at.isoformat(),
                "started_at": job.started_at.isoformat() if job.started_at else None,
                "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                "result": job.result,
                "error": job.error,
            }

    async def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a pending or running job.

        Args:
            job_id: Job ID to cancel

        Returns:
            True if cancelled, False if already completed/failed
        """
        async with self._lock:
            job = self.jobs.get(job_id)
            if not job or job.status in [JobStatus.COMPLETED, JobStatus.FAILED]:
                return False
            job.status = JobStatus.CANCELLED
            logger.info(f"Job {job_id} cancelled")
            return True

    async def _worker(self, worker_id: int):
        """Background worker that processes jobs."""
        logger.info(f"Worker {worker_id} started")

        while self._running:
            try:
                # Get job with timeout to allow checking _running flag
                job = await asyncio.wait_for(self.queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            try:
                async with self._lock:
                    if job.status == JobStatus.CANCELLED:
                        self.queue.task_done()
                        continue
                    job.status = JobStatus.RUNNING
                    job.started_at = datetime.now()

                logger.info(f"Worker {worker_id} executing job {job.job_id}")

                # Execute job callback
                result = await job.callback(**job.params)

                async with self._lock:
                    job.status = JobStatus.COMPLETED
                    job.completed_at = datetime.now()
                    job.result = result
                    job.progress = 1.0

                logger.info(f"Job {job.job_id} completed successfully")

            except Exception as e:
                logger.error(f"Job {job.job_id} failed: {e}", exc_info=True)
                async with self._lock:
                    job.status = JobStatus.FAILED
                    job.completed_at = datetime.now()
                    job.error = str(e)

            finally:
                self.queue.task_done()

    async def start_workers(self):
        """Start background worker tasks."""
        if self._running:
            logger.warning("Workers already running")
            return

        self._running = True
        for i in range(self.num_workers):
            worker = asyncio.create_task(self._worker(i))
            self.workers.append(worker)
        logger.info(f"Started {self.num_workers} workers")

    async def stop_workers(self):
        """Stop all workers gracefully."""
        if not self._running:
            return

        self._running = False

        # Wait for current jobs to complete
        await self.queue.join()

        # Cancel all workers
        for worker in self.workers:
            worker.cancel()

        # Wait for cancellation to complete
        await asyncio.gather(*self.workers, return_exceptions=True)
        self.workers.clear()
        logger.info("All workers stopped")

    async def cleanup_old_jobs(self):
        """Remove completed jobs older than TTL."""
        cutoff = datetime.now() - timedelta(seconds=self.result_ttl)
        async with self._lock:
            to_remove = [
                job_id for job_id, job in self.jobs.items()
                if job.completed_at and job.completed_at < cutoff
            ]
            for job_id in to_remove:
                del self.jobs[job_id]

            if to_remove:
                logger.info(f"Cleaned up {len(to_remove)} old jobs")

    async def list_jobs(self) -> List[dict]:
        """
        List all jobs with their current status.

        Returns:
            List of job status dictionaries
        """
        async with self._lock:
            return [
                {
                    "job_id": job.job_id,
                    "job_type": job.job_type,
                    "status": job.status.value,
                    "progress": job.progress,
                    "created_at": job.created_at.isoformat(),
                    "started_at": job.started_at.isoformat() if job.started_at else None,
                    "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                    "error": job.error,
                }
                for job in self.jobs.values()
            ]

    def get_stats(self) -> dict:
        """Get queue statistics."""
        return {
            "queue_size": self.queue.qsize(),
            "total_jobs": len(self.jobs),
            "num_workers": self.num_workers,
            "workers_running": self._running,
        }


# Singleton instance
_job_queue: Optional[JobQueue] = None


def get_job_queue() -> JobQueue:
    """Get the singleton job queue instance."""
    global _job_queue
    if _job_queue is None:
        _job_queue = JobQueue(num_workers=1, result_ttl=3600)
    return _job_queue


def set_job_queue(queue: JobQueue):
    """Set a custom job queue instance (for testing)."""
    global _job_queue
    _job_queue = queue
