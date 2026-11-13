"""Job queue configuration and TTL cleanup."""

import asyncio
from datetime import datetime, timedelta

import pytest

from api.config import AppConfig
from api.parameters import Parameters
from api.services.job_queue import GPU_LANE, IO_LANE, JobQueue, JobStatus
from metastable import ConfigurationError


async def wait_for_status(queue, job_ids, status, timeout=2.0):
    """Poll until every job in job_ids has the given status."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while True:
        statuses = [(await queue.get_job_status(job_id))["status"] for job_id in job_ids]
        if all(s == status for s in statuses):
            return
        if loop.time() > deadline:
            raise AssertionError(f"expected {status}, got {statuses}")
        await asyncio.sleep(0.01)


def config(api_section):
    return AppConfig(config_file="test", parameters=Parameters({"api": api_section}))


def test_queue_and_cache_settings_are_read_from_config():
    cfg = config({"job_queue": {"num_workers": 3, "result_ttl": 120}, "model_cache": {"max_models": 4}})
    assert cfg.job_queue_num_workers == 3
    assert cfg.job_queue_result_ttl == 120
    assert cfg.model_cache_max_models == 4


def test_queue_and_cache_settings_default_when_missing():
    cfg = config({})
    assert cfg.job_queue_num_workers == 1
    assert cfg.job_queue_result_ttl == 3600
    assert cfg.model_cache_max_models == 2


@pytest.mark.parametrize("value", [0, -1, "2", True])
def test_invalid_worker_count_is_rejected(value):
    with pytest.raises(ConfigurationError):
        config({"job_queue": {"num_workers": value}}).job_queue_num_workers


def test_old_finished_and_cancelled_jobs_are_cleaned_up():
    async def main():
        queue = JobQueue(num_workers=1, result_ttl=60)

        async def noop():
            return None

        finished = await queue.submit_job("test", {}, noop)
        cancelled = await queue.submit_job("test", {}, noop)
        pending = await queue.submit_job("test", {}, noop)

        queue.jobs[finished].status = JobStatus.COMPLETED
        queue.jobs[finished].completed_at = datetime.now() - timedelta(seconds=120)
        assert await queue.cancel_job(cancelled)
        queue.jobs[cancelled].completed_at -= timedelta(seconds=120)

        await queue.cleanup_old_jobs()
        return set(queue.jobs), pending

    remaining, pending = asyncio.run(main())
    assert remaining == {pending}


def test_workers_run_jobs_concurrently_when_configured():
    async def main():
        queue = JobQueue(num_workers=2, result_ttl=60)
        await queue.start_workers()
        running = []
        peak = 0

        async def job():
            nonlocal peak
            running.append(1)
            peak = max(peak, len(running))
            await asyncio.sleep(0.05)
            running.pop()

        ids = [await queue.submit_job("test", {}, job) for _ in range(2)]
        await wait_for_status(queue, ids, "completed")
        await queue.stop_workers()
        return peak

    assert asyncio.run(main()) == 2


def test_io_job_runs_while_gpu_lane_is_busy():
    async def main():
        queue = JobQueue(num_workers=1, io_workers=1, result_ttl=60)
        await queue.start_workers()
        gpu_release = asyncio.Event()

        async def gpu_job():
            await gpu_release.wait()

        async def io_job():
            return "downloaded"

        gpu_id = await queue.submit_job("generation", {}, gpu_job)
        io_id = await queue.submit_job("download", {}, io_job, lane=IO_LANE)
        await wait_for_status(queue, [io_id], "completed")
        gpu_status = (await queue.get_job_status(gpu_id))["status"]
        gpu_release.set()
        await queue.stop_workers()
        return gpu_status, await queue.get_job_status(io_id)

    gpu_status, io_job = asyncio.run(main())
    assert gpu_status == "running"
    assert io_job["result"] == "downloaded"
    assert io_job["lane"] == IO_LANE


def test_unknown_lane_is_rejected():
    async def main():
        async def noop():
            return None
        with pytest.raises(ValueError):
            await JobQueue().submit_job("test", {}, noop, lane="tpu")

    asyncio.run(main())


def test_reporter_publishes_progress_and_detail():
    async def main():
        queue = JobQueue(io_workers=1, result_ttl=60)
        await queue.start_workers()
        reported = asyncio.Event()
        finish = asyncio.Event()

        async def job(report):
            report(0.25, {"bytes_done": 25, "bytes_total": 100})
            reported.set()
            await finish.wait()

        job_id = await queue.submit_job("download", {}, job, lane=IO_LANE, with_reporter=True)
        await reported.wait()
        mid = (await queue.list_jobs())[0]
        finish.set()
        await wait_for_status(queue, [job_id], "completed")
        await queue.stop_workers()
        return mid, await queue.get_job_status(job_id)

    mid, final = asyncio.run(main())
    assert mid["progress"] == 0.25
    assert mid["detail"] == {"bytes_done": 25, "bytes_total": 100}
    assert final["progress"] == 1.0


def test_running_cancellable_job_is_interrupted():
    async def main():
        queue = JobQueue(io_workers=1, result_ttl=60)
        await queue.start_workers()
        started = asyncio.Event()
        cleaned_up = asyncio.Event()

        async def job():
            started.set()
            try:
                await asyncio.sleep(60)
            finally:
                cleaned_up.set()

        job_id = await queue.submit_job("download", {}, job, lane=IO_LANE, cancellable=True)
        await started.wait()
        assert await queue.cancel_job(job_id)
        await asyncio.wait_for(cleaned_up.wait(), 1)
        # cancel_job() flips the status at once; the worker stamps completed_at when the task unwinds
        for _ in range(100):
            status = await queue.get_job_status(job_id)
            if status["completed_at"]:
                break
            await asyncio.sleep(0.01)
        # the worker survives and keeps taking jobs
        async def after():
            return "ok"
        next_id = await queue.submit_job("download", {}, after, lane=IO_LANE)
        await wait_for_status(queue, [next_id], "completed")
        await queue.stop_workers()
        return status

    status = asyncio.run(main())
    assert status["status"] == "cancelled"
    assert status["completed_at"] is not None


def test_cancelled_gpu_job_keeps_cancelled_status():
    async def main():
        queue = JobQueue(num_workers=1, result_ttl=60)
        await queue.start_workers()
        started = asyncio.Event()
        finish = asyncio.Event()

        async def job():
            started.set()
            await finish.wait()
            return {"image_base64": "..."}

        job_id = await queue.submit_job("generation", {}, job)
        await started.wait()
        assert await queue.cancel_job(job_id)
        finish.set()
        await asyncio.sleep(0.05)
        await queue.stop_workers()
        return await queue.get_job_status(job_id)

    status = asyncio.run(main())
    assert status["status"] == "cancelled"
    assert status["result"] is None
    assert status["completed_at"] is not None


def test_cancelling_a_finished_job_fails():
    async def main():
        queue = JobQueue(result_ttl=60)

        async def noop():
            return None

        job_id = await queue.submit_job("test", {}, noop)
        assert await queue.cancel_job(job_id)
        return await queue.cancel_job(job_id), await queue.cancel_job("missing")

    assert asyncio.run(main()) == (False, False)


def test_shutdown_cancels_queued_jobs_and_interrupts_downloads():
    async def main():
        queue = JobQueue(num_workers=1, io_workers=1, result_ttl=60)
        await queue.start_workers()
        gpu_started = asyncio.Event()
        io_started = asyncio.Event()

        async def gpu_job():
            gpu_started.set()
            await asyncio.sleep(0.1)
            return "rendered"

        async def io_job():
            io_started.set()
            await asyncio.sleep(60)

        async def noop():
            return None

        running_gpu = await queue.submit_job("generation", {}, gpu_job)
        running_io = await queue.submit_job("download", {}, io_job, lane=IO_LANE, cancellable=True)
        await gpu_started.wait()
        await io_started.wait()
        queued_gpu = await queue.submit_job("generation", {}, noop)
        queued_io = await queue.submit_job("download", {}, noop, lane=IO_LANE)

        await asyncio.wait_for(queue.stop_workers(), 2)
        late = await queue.submit_job("generation", {}, noop)
        ids = {"running_gpu": running_gpu, "running_io": running_io, "queued_gpu": queued_gpu,
               "queued_io": queued_io, "late": late}
        return {name: (await queue.get_job_status(job_id))["status"] for name, job_id in ids.items()}

    statuses = asyncio.run(main())
    assert statuses == {
        "running_gpu": "completed",
        "running_io": "cancelled",
        "queued_gpu": "cancelled",
        "queued_io": "cancelled",
        "late": "cancelled",
    }


def test_find_active_job_matches_type_and_params():
    async def main():
        queue = JobQueue(result_ttl=60)

        async def noop(**kwargs):
            return None

        job_id = await queue.submit_job("download", {"file_id": 7, "dest": "x"}, noop, lane=IO_LANE)
        found = await queue.find_active_job("download", file_id=7)
        missing = await queue.find_active_job("download", file_id=8)
        await queue.cancel_job(job_id)
        after_cancel = await queue.find_active_job("download", file_id=7)
        return found == job_id, missing, after_cancel

    assert asyncio.run(main()) == (True, None, None)


def test_stats_report_each_lane():
    stats = JobQueue(num_workers=2, io_workers=3).get_stats()
    assert stats["num_workers"] == 2
    assert stats["lanes"][GPU_LANE]["num_workers"] == 2
    assert stats["lanes"][IO_LANE]["num_workers"] == 3
    assert stats["queue_size"] == 0
