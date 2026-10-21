const Telemetry = (() => {
  const POLL_MS = 5000;
  let pollHandle = null;

  function formatGB(bytes) {
    return (bytes / 1024 ** 3).toFixed(1);
  }

  function meter(percent, label) {
    const pct = Math.max(0, Math.min(100, percent));
    return `
      <div class="app-meter-label">${label}</div>
      <div class="app-meter"><div class="app-meter__fill" style="width:${pct}%"></div></div>
    `;
  }

  function alert(message) {
    return `<div class="pf-v5-c-alert pf-m-info pf-m-inline"><div class="pf-v5-c-alert__title">${escapeHtml(message)}</div></div>`;
  }

  function renderCacheCard(cache) {
    const pct = cache.max_models ? (cache.cached_models / cache.max_models) * 100 : 0;
    const keys = cache.cache_keys.length
      ? `<ul>${cache.cache_keys.map((k) => `<li>${escapeHtml(k)}</li>`).join("")}</ul>`
      : "<p>Cache is empty.</p>";
    return `
      <div class="pf-v5-c-card">
        <div class="pf-v5-c-card__title">Model Cache</div>
        <div class="pf-v5-c-card__body">
          ${meter(pct, `${cache.cached_models} / ${cache.max_models} models loaded`)}
          ${keys}
        </div>
      </div>
    `;
  }

  function renderGpuMemoryCard(device, gpu) {
    let body;
    if (gpu.available) {
      const pct = (gpu.used_bytes / gpu.total_bytes) * 100;
      const approxNote = gpu.total_is_approximate
        ? " (approximate - unified memory, no fixed VRAM ceiling)"
        : "";
      body = `
        ${meter(pct, `${formatGB(gpu.used_bytes)} GB / ${formatGB(gpu.total_bytes)} GB used${approxNote}`)}
        <p>PyTorch allocated: ${formatGB(gpu.allocated_bytes)} GB</p>
      `;
    } else {
      body = alert(gpu.reason || "GPU memory stats unavailable");
    }
    return `
      <div class="pf-v5-c-card">
        <div class="pf-v5-c-card__title">GPU VRAM (${escapeHtml(device)})</div>
        <div class="pf-v5-c-card__body">${body}</div>
      </div>
    `;
  }

  function renderGpuComputeCard(compute) {
    const body = compute.available
      ? meter(compute.utilization_percent, `${compute.utilization_percent.toFixed(0)}% utilization`)
      : alert(compute.reason || "GPU compute utilization unavailable");
    return `
      <div class="pf-v5-c-card">
        <div class="pf-v5-c-card__title">GPU Compute</div>
        <div class="pf-v5-c-card__body">${body}</div>
      </div>
    `;
  }

  function renderHostCard(host) {
    const loadAvg = host.load_average
      ? host.load_average.map((n) => n.toFixed(2)).join(", ")
      : "n/a";
    return `
      <div class="pf-v5-c-card">
        <div class="pf-v5-c-card__title">Node (Host)</div>
        <div class="pf-v5-c-card__body">
          ${meter(host.cpu_percent, `CPU: ${host.cpu_percent.toFixed(0)}% across ${host.cpu_count} cores`)}
          ${meter(host.memory_percent, `RAM: ${formatGB(host.memory_used_bytes)} GB / ${formatGB(host.memory_total_bytes)} GB`)}
          <p>Load average: ${loadAvg}</p>
        </div>
      </div>
    `;
  }

  function render(info) {
    const container = document.getElementById("tab-telemetry");
    container.innerHTML = `
      <h2 class="pf-v5-c-title pf-m-xl">Telemetry</h2>
      <div class="telemetry-grid">
        ${renderCacheCard(info.cache_stats)}
        ${renderGpuMemoryCard(info.device, info.gpu_stats)}
        ${renderGpuComputeCard(info.gpu_stats.compute)}
        ${renderHostCard(info.host_stats)}
      </div>
    `;
  }

  async function refresh() {
    try {
      const info = await Api.getSystemInfo();
      render(info);
    } catch (e) {
      document.getElementById("tab-telemetry").innerHTML =
        `<div class="pf-v5-c-alert pf-m-danger"><div class="pf-v5-c-alert__title">Failed to load telemetry: ${escapeHtml(e.message)}</div></div>`;
    }
  }

  function activate() {
    refresh();
    if (!pollHandle) pollHandle = setInterval(refresh, POLL_MS);
  }

  function deactivate() {
    if (pollHandle) {
      clearInterval(pollHandle);
      pollHandle = null;
    }
  }

  return { activate, deactivate };
})();
