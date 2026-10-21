const Outputs = (() => {
  const PAGE_SIZE = 24;
  let offset = 0;
  let total = 0;
  let currentPageCount = 0;

  async function showDetails(id) {
    try {
      const meta = await Api.getOutputMetadata(id);
      showModal(`Output ${id}`, `<pre>${escapeHtml(JSON.stringify(meta, null, 2))}</pre>`);
    } catch (e) {
      showModal("Error", `<p>${escapeHtml(e.message)}</p>`);
    }
  }

  async function deleteItem(id) {
    if (!confirm("Delete this output? This removes the image and metadata permanently.")) {
      return;
    }
    try {
      await Api.deleteOutput(id);
    } catch (e) {
      alert(`Could not delete output: ${e.message}`);
      return;
    }
    if (currentPageCount === 1 && offset > 0) {
      offset = Math.max(0, offset - PAGE_SIZE);
    }
    await refresh();
  }

  function render(data) {
    total = data.total;
    currentPageCount = data.items.length;
    const container = document.getElementById("tab-outputs");

    const cards = data.items.map((item) => `
      <div class="pf-v5-c-card">
        <div class="pf-v5-c-card__body">
          <img src="${Api.outputImageUrl(item.id)}" alt="${item.id}" loading="lazy">
        </div>
        <div class="pf-v5-c-card__footer">
          <small>${item.created_at}</small><br>
          <button class="pf-v5-c-button pf-m-link pf-m-inline" data-details="${item.id}">Details</button>
          <a class="pf-v5-c-button pf-m-link pf-m-inline" href="${item.image_url}" download>Image</a>
          <a class="pf-v5-c-button pf-m-link pf-m-inline" href="${item.json_url}" download>JSON</a>
          <button class="pf-v5-c-button pf-m-link pf-m-inline pf-m-danger" data-delete="${item.id}">Delete</button>
        </div>
      </div>
    `).join("");

    const hasPrev = offset > 0;
    const hasNext = offset + PAGE_SIZE < total;

    container.innerHTML = `
      <h2 class="pf-v5-c-title pf-m-xl">Outputs</h2>
      <div class="app-toolbar">
        <button class="pf-v5-c-button pf-m-secondary" id="outputs-prev" ${hasPrev ? "" : "disabled"}>Previous</button>
        <span>${total === 0 ? 0 : offset + 1}-${Math.min(offset + PAGE_SIZE, total)} of ${total}</span>
        <button class="pf-v5-c-button pf-m-secondary" id="outputs-next" ${hasNext ? "" : "disabled"}>Next</button>
      </div>
      <div class="output-gallery">${cards || "<p>No outputs persisted yet.</p>"}</div>
    `;

    container.querySelector("#outputs-prev")?.addEventListener("click", () => {
      offset = Math.max(0, offset - PAGE_SIZE);
      refresh();
    });
    container.querySelector("#outputs-next")?.addEventListener("click", () => {
      offset += PAGE_SIZE;
      refresh();
    });
    container.querySelectorAll("[data-details]").forEach((btn) => {
      btn.addEventListener("click", () => showDetails(btn.dataset.details));
    });
    container.querySelectorAll("[data-delete]").forEach((btn) => {
      btn.addEventListener("click", () => deleteItem(btn.dataset.delete));
    });
  }

  async function refresh() {
    try {
      const data = await Api.listOutputs(PAGE_SIZE, offset);
      render(data);
    } catch (e) {
      document.getElementById("tab-outputs").innerHTML =
        `<div class="pf-v5-c-alert pf-m-danger"><div class="pf-v5-c-alert__title">Failed to load outputs: ${escapeHtml(e.message)}</div></div>`;
    }
  }

  function activate() {
    refresh();
  }

  function deactivate() {}

  return { activate, deactivate };
})();
