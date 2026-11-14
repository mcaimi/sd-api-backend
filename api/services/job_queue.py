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

from api.config import get_app_config

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """Job execution status."""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# Lanes: "gpu" runs generation/merge jobs, "io" runs downloads, so a long
# download never occupies a GPU worker.
GPU_LANE = "gpu"
IO_LANE = "io"

# Placed on a lane queue to stop one worker
_STOP = object()


class Job:
    """Represents a job in the queue."""

    def __init__(
        self,
        job_id: str,
        job_type: str,
        params: dict,
        callback: Callable,
        lane: str = GPU_LANE,
        cancellable: bool = False,
        with_reporter: bool = False,
    ):
        self.job_id = job_id
        self.job_type = job_type  # "generation", "merge", etc.
        self.params = params
        self.callback = callback
        self.lane = lane
        # Cancellable jobs are interrupted when cancelled while running;
        # others (GPU work in executor threads can't be interrupted) only
        # have their result discarded.
        self.cancellable = cancellable
        self.with_reporter = with_reporter
        self.status = JobStatus.QUEUED
        self.created_at = datetime.now()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.result: Optional[Any] = None
        self.error: Optional[str] = None
        self.progress: float = 0.0
        self.detail: Optional[dict] = None  # free-form progress info, e.g. bytes downloaded
        self.task: Optional[asyncio.Task] = None


class JobQueue:
    """In-memory async job queue with background workers, one pool per lane."""

    def __init__(self, num_workers: int = 1, result_ttl: int = 3600, io_workers: int = 1):
        """
        Initialize job queue.

        Args:
            num_workers: Number of "gpu" lane workers (generation and merge)
            result_ttl: Time to live for completed jobs (seconds)
            io_workers: Number of "io" lane workers (downloads)
        """
        self.num_workers = num_workers
        self.io_workers = io_workers
        self.result_ttl = result_ttl
        self._lane_workers = {GPU_LANE: num_workers, IO_LANE: io_workers}
        self._queues: Dict[str, asyncio.Queue] = {lane: asyncio.Queue() for lane in self._lane_workers}
        self.jobs: Dict[str, Job] = {}
        self.workers: List[asyncio.Task] = []
        self._cleanup_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        self._running = False
        self._stopping = False

    async def submit_job(
        self,
        job_type: str,
        params: dict,
        callback: Callable,
        lane: str = GPU_LANE,
        cancellable: bool = False,
        with_reporter: bool = False,
    ) -> str:
        """
        Submit a job and return job ID.

        Args:
            job_type: Type of job (for logging/tracking)
            params: Parameters to pass to callback
            callback: Async function to execute
            lane: Worker pool to run on: "gpu" or "io"
            cancellable: Interrupt the callback when the job is cancelled while running
            with_reporter: Pass the callback a report(progress, detail=None)
                function to publish progress (0.0-1.0) and a detail dict

        Returns:
            Job ID string
        """
        if lane not in self._queues:
            raise ValueError(f"Unknown job lane: {lane}")

        job_id = str(uuid.uuid4())
        job = Job(job_id, job_type, params, callback, lane, cancellable, with_reporter)

        async with self._lock:
            self.jobs[job_id] = job
            if self._stopping:
                # No worker will pick it up anymore
                job.status = JobStatus.CANCELLED
                job.completed_at = datetime.now()
                job.error = "Server is shutting down"
                return job_id

        await self._queues[lane].put(job)
        logger.info(f"Job {job_id} ({job_type}) submitted to lane {lane}")
        return job_id

    def _serialize(self, job: Job, include_result: bool) -> dict:
        data = {
            "job_id": job.job_id,
            "job_type": job.job_type,
            "lane": job.lane,
            "cancellable": job.cancellable,
            "status": job.status.value,
            "progress": job.progress,
            "detail": job.detail,
            "created_at": job.created_at.isoformat(),
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "error": job.error,
        }
        if include_result:
            data["result"] = job.result
        return data

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
            return self._serialize(job, include_result=True)

    async def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a pending or running job.

        Queued jobs never run. Running cancellable jobs are interrupted;
        running non-cancellable jobs finish, but their result is discarded.

        Args:
            job_id: Job ID to cancel

        Returns:
            True if cancelled, False if not found or already finished
        """
        async with self._lock:
            job = self.jobs.get(job_id)
            if not job or job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                return False
            if job.status == JobStatus.QUEUED:
                # Never reaches a worker, so stamp it here for TTL cleanup
                job.completed_at = datetime.now()
            elif job.cancellable and job.task is not None:
                job.task.cancel()
            job.status = JobStatus.CANCELLED
            logger.info(f"Job {job_id} cancelled")
            return True

    def _reporter(self, job: Job) -> Callable:
        def report(progress: float, detail: Optional[dict] = None) -> None:
            job.progress = min(max(float(progress), 0.0), 1.0)
            if detail is not None:
                job.detail = detail
        return report

    async def _run_job(self, job: Job, worker_name: str):
        """Run one job's callback and record its outcome."""
        async with self._lock:
            if job.status == JobStatus.CANCELLED:
                return
            job.status = JobStatus.RUNNING
            job.started_at = datetime.now()
            kwargs = dict(job.params)
            if job.with_reporter:
                kwargs["report"] = self._reporter(job)
            job.task = asyncio.create_task(job.callback(**kwargs))

        logger.info(f"Worker {worker_name} executing job {job.job_id}")
        try:
            result = await job.task
        except asyncio.CancelledError:
            async with self._lock:
                job.status = JobStatus.CANCELLED
                job.completed_at = datetime.now()
                job.error = job.error or "Cancelled"
            if asyncio.current_task().cancelling():
                raise  # the worker itself is being cancelled
            logger.info(f"Job {job.job_id} cancelled while running")
        except Exception as e:
            logger.error(f"Job {job.job_id} failed: {e}", exc_info=True)
            async with self._lock:
                if job.status != JobStatus.CANCELLED:
                    job.status = JobStatus.FAILED
                    job.error = str(e)
                job.completed_at = datetime.now()
        else:
            async with self._lock:
                job.completed_at = datetime.now()
                if job.status == JobStatus.CANCELLED:
                    # Cancelled while running but not interruptible: drop the result
                    logger.info(f"Job {job.job_id} finished after being cancelled, result discarded")
                else:
                    job.status = JobStatus.COMPLETED
                    job.result = result
                    job.progress = 1.0
                    logger.info(f"Job {job.job_id} completed successfully")
        finally:
            job.task = None

    async def _worker(self, lane: str, worker_id: int):
        """Background worker that processes jobs from one lane until stopped."""
        worker_name = f"{lane}-{worker_id}"
        queue = self._queues[lane]
        logger.info(f"Worker {worker_name} started")

        while True:
            job = await queue.get()
            try:
                if job is _STOP:
                    break
                await self._run_job(job, worker_name)
            finally:
                queue.task_done()

        logger.info(f"Worker {worker_name} stopped")

    async def start_workers(self):
        """Start background worker tasks."""
        if self._running:
            logger.warning("Workers already running")
            return

        self._running = True
        self._stopping = False
        for lane, count in self._lane_workers.items():
            for i in range(count):
                self.workers.append(asyncio.create_task(self._worker(lane, i)))
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info(f"Started workers: {self._lane_workers}")

    async def stop_workers(self):
        """
        Stop all workers.

        Queued jobs are cancelled, running cancellable jobs (downloads) are
        interrupted, and running GPU jobs are allowed to finish.
        """
        if not self._running:
            return

        self._running = False

        if self._cleanup_task:
            self._cleanup_task.cancel()
            await asyncio.gather(self._cleanup_task, return_exceptions=True)
            self._cleanup_task = None

        async with self._lock:
            self._stopping = True
            now = datetime.now()
            for queue in self._queues.values():
                while not queue.empty():
                    job = queue.get_nowait()
                    queue.task_done()
                    if job is not _STOP and job.status == JobStatus.QUEUED:
                        job.status = JobStatus.CANCELLED
                        job.completed_at = now
                        job.error = "Server is shutting down"
            for job in self.jobs.values():
                if job.status == JobStatus.RUNNING and job.cancellable and job.task is not None:
                    job.task.cancel()

        # Each worker finishes its current job, then takes one stop marker
        for lane, count in self._lane_workers.items():
            for _ in range(count):
                self._queues[lane].put_nowait(_STOP)

        await asyncio.gather(*self.workers, return_exceptions=True)
        self.workers.clear()
        logger.info("All workers stopped")

    async def _cleanup_loop(self):
        """Periodically drop finished jobs older than result_ttl."""
        interval = min(60, self.result_ttl)
        while self._running:
            await asyncio.sleep(interval)
            try:
                await self.cleanup_old_jobs()
            except Exception as e:
                logger.error(f"Job cleanup failed: {e}", exc_info=True)

    async def cleanup_old_jobs(self):
        """Remove completed jobs older than TTL."""
        cutoff = datetime.now() - timedelta(seconds=self.result_ttl)
        async with self._lock:
            to_remove = [
                job_id for job_id, job in self.jobs.items()
                if job.completed_at and job.completed_at < cutoff
                and job.status != JobStatus.RUNNING
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
            return [self._serialize(job, include_result=False) for job in self.jobs.values()]

    async def find_active_job(self, job_type: str, **params) -> Optional[str]:
        """Return the ID of a queued/running job of job_type whose params include params."""
        async with self._lock:
            for job in self.jobs.values():
                if (
                    job.job_type == job_type
                    and job.status in (JobStatus.QUEUED, JobStatus.RUNNING)
                    and all(job.params.get(key) == value for key, value in params.items())
                ):
                    return job.job_id
        return None

    def get_stats(self) -> dict:
        """Get queue statistics."""
        lanes = {
            lane: {
                "queue_size": self._queues[lane].qsize(),
                "num_workers": count,
                "running_jobs": sum(
                    1 for job in self.jobs.values()
                    if job.lane == lane and job.status == JobStatus.RUNNING
                ),
            }
            for lane, count in self._lane_workers.items()
        }
        return {
            "queue_size": sum(lane["queue_size"] for lane in lanes.values()),
            "total_jobs": len(self.jobs),
            "num_workers": self.num_workers,
            "workers_running": self._running,
            "lanes": lanes,
        }


# Singleton instance
_job_queue: Optional[JobQueue] = None


def get_job_queue() -> JobQueue:
    """Get the singleton job queue instance."""
    global _job_queue
    if _job_queue is None:
        config = get_app_config()
        num_workers = config.job_queue_num_workers
        if num_workers > 1:
            logger.warning(
                f"api.job_queue.num_workers={num_workers}: GPU jobs on different checkpoints run "
                "concurrently and peak VRAM/RAM grows with each running job "
                "(Flux jobs still run one at a time)"
            )
        _job_queue = JobQueue(
            num_workers=num_workers,
            result_ttl=config.job_queue_result_ttl,
            io_workers=config.job_queue_io_workers,
        )
    return _job_queue


def set_job_queue(queue: JobQueue):
    """Set a custom job queue instance (for testing)."""
    global _job_queue
    _job_queue = queue
