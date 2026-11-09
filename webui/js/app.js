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
