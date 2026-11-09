document.addEventListener("alpine:init", () => {
  const POLL_MS = 5000;

  Alpine.data("telemetry", () => tabComponent("telemetry", {
    info: null,
    error: null,
    pollHandle: null,

    formatGB(bytes) {
      return (bytes / 1024 ** 3).toFixed(1);
    },

    clamp(percent) {
      return Math.max(0, Math.min(100, percent));
    },

    // Rings turn amber, then red, as a resource fills up.
    ringStyle(percent) {
      const role = percent >= 90 ? "danger" : percent >= 70 ? "warning" : "brand";
      return `--indicator-color: var(--wa-color-${role}-fill-loud)`;
    },

    get cachePercent() {
      const cache = this.info.cache_stats;
      return cache.max_models ? (cache.cached_models / cache.max_models) * 100 : 0;
    },

    get vramPercent() {
      const gpu = this.info.gpu_stats;
      return (gpu.used_bytes / gpu.total_bytes) * 100;
    },

    get loadAverage() {
      const load = this.info.host_stats.load_average;
      return load ? load.map((n) => n.toFixed(2)).join(" · ") : "n/a";
    },

    async refresh() {
      try {
        this.info = await Api.getSystemInfo();
        this.error = null;
      } catch (e) {
        this.error = e.message;
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
