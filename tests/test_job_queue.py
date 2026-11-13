"""Job queue configuration and TTL cleanup."""

import asyncio
from datetime import datetime, timedelta

import pytest

from api.config import AppConfig
from api.parameters import Parameters
from api.services.job_queue import JobQueue, JobStatus
from metastable import ConfigurationError


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

        for _ in range(2):
            await queue.submit_job("test", {}, job)
        # stop_workers() doesn't drain queued jobs (workers exit first), so wait here
        await queue.queue.join()
        await queue.stop_workers()
        return peak

    assert asyncio.run(main()) == 2
