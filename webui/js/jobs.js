document.addEventListener("alpine:init", () => {
  const POLL_MS = 3000;

  Alpine.data("jobs", () => tabComponent("jobs", {
    jobs: [],
    loading: true,
    error: null,
    pollHandle: null,

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

    resultJson(job) {
      const { image_base64, ...rest } = job.result;
      return JSON.stringify(rest, null, 2);
    },

    formatTimestamp,

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
        this.error = null;
      } catch (e) {
        this.error = e.message;
      } finally {
        this.loading = false;
      }
    },

    activate() {
      this.refresh();
      if (!this.pollHandle) this.pollHandle = setInterval(() => this.refresh(), POLL_MS);
    },

    deactivate() {
      if (this.pollHandle) {
        clearInterval(this.pollHandle);
        this.pollHandle = null;
      }
    },
  }));
});
