document.addEventListener("alpine:init", () => {
  const POLL_MS = 3000;
  // Faster while a download runs, so its bytes/speed/ETA stay current
  const DOWNLOAD_POLL_MS = 1000;

  Alpine.data("jobs", () => tabComponent("jobs", {
    jobs: [],
    loading: true,
    error: null,
    pollHandle: null,
    // Full results, fetched when a job is expanded: the job list leaves them out (they can hold images)
    results: {},
    expanded: {},
    pollMs: POLL_MS,

    statusVariant(status) {
      switch (status) {
        case "completed": return "success";
        case "failed": return "danger";
        case "cancelled": return "neutral";
        case "running": return "brand";
        default: return "warning";
      }
    },

    canCancel(job) {
      return ["queued", "running"].includes(job.status);
    },

    resultJson(result) {
      const { image_base64, ...rest } = result;
      return JSON.stringify(rest, null, 2);
    },

    expand(job) {
      this.expanded[job.job_id] = true;
      this.loadResult(job);
    },

    async loadResult(job) {
      if (job.status !== "completed" || this.results[job.job_id]) return;
      try {
        const full = await Api.getJob(job.job_id);
        if (full.result) this.results[job.job_id] = full.result;
      } catch (e) {
        notify(`Could not load the job result: ${e.message}`);
      }
    },

    formatTimestamp,

    downloadText(job) {
      const d = job.detail;
      if (!d) return "";
      if (d.phase === "verifying") return "Verifying…";
      const parts = [`${formatBytes(d.bytes_done)} / ${formatBytes(d.bytes_total)}`];
      if (d.speed_bps) parts.push(`${formatBytes(d.speed_bps)}/s`);
      if (d.eta_s != null) parts.push(`ETA ${formatDuration(d.eta_s)}`);
      return parts.join(" · ");
    },

    schedulePolling() {
      const downloading = this.jobs.some((job) => job.lane === "io" && ["queued", "running"].includes(job.status));
      const pollMs = downloading ? DOWNLOAD_POLL_MS : POLL_MS;
      if (this.pollHandle && pollMs === this.pollMs) return;
      clearInterval(this.pollHandle);
      this.pollMs = pollMs;
      this.pollHandle = setInterval(() => this.refresh(), pollMs);
    },

    async cancel(jobId) {
      try {
        await Api.cancelJob(jobId);
      } catch (e) {
        notify(`Could not cancel job: ${e.message}`);
      }
      await this.refresh();
    },

    async refresh() {
      try {
        const jobs = await Api.listJobs();
        this.jobs = jobs.sort((a, b) => b.created_at.localeCompare(a.created_at));
        const ids = new Set(jobs.map((job) => job.job_id));
        for (const id of Object.keys(this.results)) if (!ids.has(id)) delete this.results[id];
        for (const job of jobs) if (this.expanded[job.job_id]) this.loadResult(job);
        this.error = null;
        if (this.pollHandle) this.schedulePolling();
      } catch (e) {
        this.error = e.message;
      } finally {
        this.loading = false;
      }
    },

    activate() {
      this.refresh();
      this.schedulePolling();
    },

    deactivate() {
      if (this.pollHandle) {
        clearInterval(this.pollHandle);
        this.pollHandle = null;
      }
    },
  }));
});
