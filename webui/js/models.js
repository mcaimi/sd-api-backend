document.addEventListener("alpine:init", () => {
  const MODEL_TYPES = ["sd15", "sdxl", "illustrious", "flux"];
  const RESOURCE_TYPES = ["checkpoints", "loras", "vaes"];
  const LAYOUT_PARTS = { clip_l: "CLIP-L", t5xxl: "T5-XXL", vae: "VAE" };

  Alpine.data("models", () => tabComponent("models", {
    modelTypes: MODEL_TYPES,
    resourceTypes: RESOURCE_TYPES,
    modelType: MODEL_TYPES[0],
    resourceType: RESOURCE_TYPES[0],
    // "local" lists files on disk; "civitai" shows the Civitai browser (civitai.js) for the same filters
    view: "local",
    models: [],
    // model/resource type the listed models belong to; row URLs use these, not the (possibly newer) filters
    listed: { modelType: MODEL_TYPES[0], resourceType: RESOURCE_TYPES[0] },
    loading: true,
    error: null,
    identifying: {},

    downloadUrl(name) {
      return Api.modelDownloadUrl(this.listed.modelType, this.listed.resourceType, name);
    },

    previewUrl(name) {
      return Api.modelPreviewUrl(this.listed.modelType, this.listed.resourceType, name);
    },

    civitaiLabel(model) {
      const info = model.civitai;
      return [info.model_name, info.version_name, info.base_model].filter(Boolean).join(" · ");
    },

    layoutLabel(layout) {
      const parts = Object.entries(LAYOUT_PARTS).filter(([key]) => layout[key]).map(([, label]) => label);
      return parts.length ? `bundled: ${parts.join(", ")}` : "transformer only";
    },

    select(field, value) {
      if (!value || this[field] === value) return;
      this[field] = value;
      if (this.view === "local") {
        this.refresh();
      } else {
        // Hidden rows would otherwise build preview URLs for the new type; setView("local") reloads them
        this.models = [];
      }
    },

    setView(value) {
      if (!value || this.view === value) return;
      this.view = value;
      if (value === "local") this.refresh();
    },

    async showMetadata(name) {
      try {
        const meta = await Api.getModelMetadata(this.listed.modelType, this.listed.resourceType, name);
        this.$store.modal.show(name, JSON.stringify(meta.metadata, null, 2));
      } catch (e) {
        notify(`Could not load metadata: ${e.message}`);
      }
    },

    async identify(name) {
      this.identifying[name] = true;
      try {
        const result = await Api.civitaiIdentify({
          model_type: this.listed.modelType, resource_type: this.listed.resourceType, model_name: name,
        });
        if (result.matched) {
          notify(`${name}: found on Civitai as ${result.sidecar.model_name}`, "success");
          await this.refresh();
        } else {
          notify(`${name}: not found on Civitai`, "neutral");
        }
      } catch (e) {
        notify(`Could not identify ${name}: ${e.message}`);
      } finally {
        delete this.identifying[name];
      }
    },

    // Called when a Civitai download finishes (see civitai.js), so the new file shows up.
    onModelsChanged(detail) {
      if (this.view === "local" && detail.modelType === this.modelType && detail.resourceType === this.resourceType) {
        this.refresh();
      }
    },

    async refresh() {
      this.loading = true;
      try {
        const data = await Api.listModels(this.modelType, this.resourceType);
        if (data.model_type !== this.modelType || data.resource_type !== this.resourceType) return;
        this.listed = { modelType: data.model_type, resourceType: data.resource_type };
        this.models = data.models;
        this.error = null;
      } catch (e) {
        this.models = [];
        this.error = e.message;
      } finally {
        this.loading = false;
      }
    },

    activate() {
      if (this.view === "local") this.refresh();
    },

    deactivate() {},
  }));
});
