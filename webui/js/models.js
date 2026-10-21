const Models = (() => {
  const MODEL_TYPES = ["sd15", "sdxl", "illustrious", "flux"];
  const RESOURCE_TYPES = ["checkpoints", "loras", "vaes"];
  let modelType = MODEL_TYPES[0];
  let resourceType = RESOURCE_TYPES[0];

  async function showMetadata(name) {
    try {
      const meta = await Api.getModelMetadata(modelType, resourceType, name);
      showModal(name, `<pre>${escapeHtml(JSON.stringify(meta.metadata, null, 2))}</pre>`);
    } catch (e) {
      showModal("Error", `<p>${escapeHtml(e.message)}</p>`);
    }
  }

  function renderTable(models) {
    const rows = models.map((m) => `
      <tr>
        <td>${escapeHtml(m.name)}</td>
        <td>
          <button class="pf-v5-c-button pf-m-link pf-m-inline" data-meta="${escapeHtml(m.name)}">Metadata</button>
          <a class="pf-v5-c-button pf-m-link pf-m-inline" href="${Api.modelDownloadUrl(modelType, resourceType, m.name)}" download>Download</a>
        </td>
      </tr>
    `).join("");

    return `
      <table class="pf-v5-c-table" role="grid">
        <thead><tr><th>Name</th><th></th></tr></thead>
        <tbody>${rows || "<tr><td colspan='2'>No models found.</td></tr>"}</tbody>
      </table>
    `;
  }

  function selectHtml(id, options, selected) {
    const opts = options.map((o) => `<option value="${o}" ${o === selected ? "selected" : ""}>${o}</option>`).join("");
    return `<select class="pf-v5-c-form-control" id="${id}">${opts}</select>`;
  }

  async function refresh() {
    const container = document.getElementById("tab-models");
    try {
      const data = await Api.listModels(modelType, resourceType);
      container.innerHTML = `
        <h2 class="pf-v5-c-title pf-m-xl">Models</h2>
        <div class="app-toolbar">
          ${selectHtml("model-type-select", MODEL_TYPES, modelType)}
          ${selectHtml("resource-type-select", RESOURCE_TYPES, resourceType)}
        </div>
        ${renderTable(data.models)}
      `;
      wireControls();
    } catch (e) {
      container.innerHTML = `
        <h2 class="pf-v5-c-title pf-m-xl">Models</h2>
        <div class="app-toolbar">
          ${selectHtml("model-type-select", MODEL_TYPES, modelType)}
          ${selectHtml("resource-type-select", RESOURCE_TYPES, resourceType)}
        </div>
        <div class="pf-v5-c-alert pf-m-danger"><div class="pf-v5-c-alert__title">Failed to load models: ${escapeHtml(e.message)}</div></div>
      `;
      wireControls();
    }
  }

  function wireControls() {
    const container = document.getElementById("tab-models");
    container.querySelector("#model-type-select").addEventListener("change", (e) => {
      modelType = e.target.value;
      refresh();
    });
    container.querySelector("#resource-type-select").addEventListener("change", (e) => {
      resourceType = e.target.value;
      refresh();
    });
    container.querySelectorAll("[data-meta]").forEach((btn) => {
      btn.addEventListener("click", () => showMetadata(btn.dataset.meta));
    });
  }

  function activate() {
    refresh();
  }

  function deactivate() {}

  return { activate, deactivate };
})();
