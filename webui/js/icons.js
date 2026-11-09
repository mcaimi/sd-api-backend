// Serve <wa-icon name="..."> from the vendored Font Awesome SVGs instead of the Font Awesome CDN,
// so the UI works offline. Add new icons to vendor/fontawesome/ from @fortawesome/fontawesome-free.
import { registerIconLibrary } from "../vendor/webawesome/webawesome.js";

registerIconLibrary("default", {
  resolver: (name) => new URL(`../vendor/fontawesome/${name}.svg`, import.meta.url).href,
  mutator: (svg) => svg.setAttribute("fill", "currentColor"),
});
