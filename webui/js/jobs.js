const Jobs = (() => {
  const POLL_MS = 3000;
  let pollHandle = null;

  function statusClass(status) {
    switch (status) {
      case "completed": return "pf-m-green";
      case "failed": return "pf-m-red";
      case "cancelled": return "pf-m-grey";
      case "running": return "pf-m-blue";
      default: return "pf-m-gold";
    }
  }

  function renderResult(job) {
    if (!job.result) {
      return job.error ? `<pre>${escapeHtml(job.error)}</pre>` : "<p>No result.</p>";
    }
    const { image_base64, ...rest } = job.result;
    let html = "";
    if (image_base64) {
      html += `<div class="job-preview"><img src="data:image/png;base64,${image_base64}" alt="job result"></div>`;
    }
    html += `<pre>${escapeHtml(JSON.stringify(rest, null, 2))}</pre>`;
    return html;
  }

  async function cancel(jobId) {
    try {
      await Api.cancelJob(jobId);
    } catch (e) {
      alert(`Could not cancel job: ${e.message}`);
    }
    await refresh();
  }

  function render(jobs) {
    const container = document.getElementById("tab-jobs");
    const sorted = [...jobs].sort((a, b) => b.created_at.localeCompare(a.created_at));

    const rows = sorted.map((job) => `
      <tbody>
        <tr class="job-row" data-job-id="${job.job_id}">
          <td>${job.job_id.slice(0, 8)}</td>
          <td>${escapeHtml(job.job_type)}</td>
          <td><span class="pf-v5-c-label ${statusClass(job.status)}"><span class="pf-v5-c-label__content">${job.status}</span></span></td>
          <td>${Math.round(job.progress * 100)}%</td>
          <td>${job.created_at}</td>
          <td>
            ${["queued", "running"].includes(job.status)
              ? `<button class="pf-v5-c-button pf-m-danger pf-m-small" data-cancel="${job.job_id}">Cancel</button>`
              : ""}
          </td>
        </tr>
        <tr class="job-details-row" data-job-id="${job.job_id}" hidden>
          <td colspan="6" class="job-details"></td>
        </tr>
      </tbody>
    `).join("");

    container.innerHTML = `
      <h2 class="pf-v5-c-title pf-m-xl">Jobs</h2>
      <table class="pf-v5-c-table" role="grid">
        <thead>
          <tr>
            <th>Job ID</th><th>Type</th><th>Status</th><th>Progress</th><th>Created</th><th></th>
          </tr>
        </thead>
        ${rows || "<tbody><tr><td colspan='6'>No jobs yet.</td></tr></tbody>"}
      </table>
    `;

    container.querySelectorAll(".job-row").forEach((row) => {
      row.addEventListener("click", (e) => {
        if (e.target.closest("[data-cancel]")) return;
        const jobId = row.dataset.jobId;
        const detailsRow = container.querySelector(`.job-details-row[data-job-id="${jobId}"]`);
        const job = sorted.find((j) => j.job_id === jobId);
        const cell = detailsRow.querySelector("td");
        if (detailsRow.hidden) {
          cell.innerHTML = renderResult(job);
        }
        detailsRow.hidden = !detailsRow.hidden;
      });
    });

    container.querySelectorAll("[data-cancel]").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        e.stopPropagation();
        cancel(btn.dataset.cancel);
      });
    });
  }

  async function refresh() {
    try {
      const jobs = await Api.listJobs();
      render(jobs);
    } catch (e) {
      document.getElementById("tab-jobs").innerHTML =
        `<div class="pf-v5-c-alert pf-m-danger"><div class="pf-v5-c-alert__title">Failed to load jobs: ${escapeHtml(e.message)}</div></div>`;
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
