function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

function showModal(title, bodyHtml) {
  const root = document.getElementById("modal-root");
  root.innerHTML = `
    <div class="app-modal-backdrop">
      <div class="pf-v5-c-modal-box app-modal-box" role="dialog" aria-modal="true">
        <header class="pf-v5-c-modal-box__header">
          <h1 class="pf-v5-c-modal-box__title">${escapeHtml(title)}</h1>
        </header>
        <div class="pf-v5-c-modal-box__body">${bodyHtml}</div>
        <footer class="pf-v5-c-modal-box__footer">
          <button class="pf-v5-c-button pf-m-primary" id="modal-close-btn">Close</button>
        </footer>
      </div>
    </div>
  `;
  const close = () => { root.innerHTML = ""; };
  root.querySelector("#modal-close-btn").addEventListener("click", close);
  root.querySelector(".app-modal-backdrop").addEventListener("click", (e) => {
    if (e.target.classList.contains("app-modal-backdrop")) close();
  });
}

const Docs = { activate() {}, deactivate() {} };

const App = (() => {
  const tabs = ["jobs", "outputs", "models", "docs", "telemetry"];
  const controllers = { jobs: Jobs, outputs: Outputs, models: Models, docs: Docs, telemetry: Telemetry };

  function activate(tab) {
    if (!tabs.includes(tab)) tab = "jobs";

    tabs.forEach((t) => {
      document.getElementById(`tab-${t}`).classList.toggle("active", t === tab);
      const link = document.querySelector(`.app-nav-link[data-tab="${t}"]`);
      link.classList.toggle("pf-m-primary", t === tab);
      link.classList.toggle("pf-m-tertiary", t !== tab);
    });

    controllers[tab].activate();
    tabs.filter((t) => t !== tab).forEach((t) => controllers[t].deactivate());
  }

  function init() {
    document.querySelectorAll(".app-nav-link").forEach((link) => {
      link.addEventListener("click", (e) => {
        e.preventDefault();
        window.location.hash = link.dataset.tab;
      });
    });

    window.addEventListener("hashchange", () => {
      activate(window.location.hash.replace("#", ""));
    });

    activate(window.location.hash.replace("#", "") || "jobs");
  }

  return { init };
})();

document.addEventListener("DOMContentLoaded", App.init);
