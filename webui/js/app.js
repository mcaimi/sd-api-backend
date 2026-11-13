// Copies property descriptors rather than spreading, so getters stay live.
function tabComponent(tab, def) {
  return Object.defineProperties({
    init() {
      const sync = (current) => (current === tab ? this.activate() : this.deactivate());
      this.$watch("$store.nav.current", sync);
      sync(this.$store.nav.current);
    },
  }, Object.getOwnPropertyDescriptors(def));
}

// Backend timestamps are naive ISO strings in server-local time; show them as-is, minus the noise.
function formatTimestamp(iso) {
  return iso.replace("T", " ").replace(/\.\d+$/, "");
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return "?";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1000 && unit < units.length - 1) {
    value /= 1000;
    unit += 1;
  }
  return `${value.toFixed(unit && value < 10 ? 1 : 0)} ${units[unit]}`;
}

function formatDuration(seconds) {
  if (!Number.isFinite(seconds)) return "?";
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
  return `${Math.floor(seconds / 3600)}h ${Math.round((seconds % 3600) / 60)}m`;
}

function formatCount(n) {
  return new Intl.NumberFormat(undefined, { notation: "compact" }).format(n || 0);
}

// Plain text from untrusted HTML (e.g. Civitai descriptions). DOMParser doesn't run scripts or load images.
function htmlToText(html) {
  const doc = new DOMParser().parseFromString(html || "", "text/html");
  doc.querySelectorAll("br").forEach((br) => br.replaceWith("\n"));
  doc.querySelectorAll("p, li, div, h1, h2, h3, h4, h5, h6").forEach((el) => el.append("\n"));
  return doc.body.textContent.replace(/\n{3,}/g, "\n\n").trim();
}

function copyText(text) {
  navigator.clipboard.writeText(text).then(
    () => notify(`Copied "${text}"`, "success"),
    () => notify("Could not copy to the clipboard"),
  );
}

function notify(message, variant = "danger") {
  const icon = variant === "danger" ? "triangle-exclamation" : "circle-info";
  document.getElementById("toaster").create(message, { variant, icon });
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

document.addEventListener("alpine:init", () => {
  const TABS = ["jobs", "outputs", "models", "docs", "telemetry"];
  const fromHash = () => {
    const tab = window.location.hash.replace("#", "");
    return TABS.includes(tab) ? tab : "jobs";
  };

  Alpine.store("nav", { current: fromHash() });
  window.addEventListener("hashchange", () => {
    Alpine.store("nav").current = fromHash();
  });

  Alpine.store("modal", {
    open: false,
    title: "",
    body: "",
    show(title, body) {
      this.title = title;
      this.body = body;
      this.open = true;
    },
  });

  // Promise-based replacement for window.confirm().
  Alpine.store("confirm", {
    open: false,
    title: "",
    message: "",
    action: "",
    resolve: null,
    ask(title, message, action) {
      this.title = title;
      this.message = message;
      this.action = action;
      this.open = true;
      return new Promise((resolve) => { this.resolve = resolve; });
    },
    answer(value) {
      this.open = false;
      this.resolve?.(value);
      this.resolve = null;
    },
  });
});
