document.addEventListener("alpine:init", () => {
  const MODEL_TYPES = ["sd15", "sdxl", "illustrious", "flux"];
  const RESOURCE_TYPES = ["checkpoints", "loras", "vaes"];

  Alpine.data("models", () => tabComponent("models", {
    modelTypes: MODEL_TYPES,
    resourceTypes: RESOURCE_TYPES,
    modelType: MODEL_TYPES[0],
    resourceType: RESOURCE_TYPES[0],
    models: [],
    loading: true,
    error: null,

    downloadUrl(name) {
      return Api.modelDownloadUrl(this.modelType, this.resourceType, name);
    },

    select(field, value) {
      if (!value || this[field] === value) return;
      this[field] = value;
      this.refresh();
    },

    async showMetadata(name) {
      try {
        const meta = await Api.getModelMetadata(this.modelType, this.resourceType, name);
        this.$store.modal.show(name, JSON.stringify(meta.metadata, null, 2));
      } catch (e) {
        notify(`Could not load metadata: ${e.message}`);
      }
    },

    async refresh() {
      this.loading = true;
      try {
        const data = await Api.listModels(this.modelType, this.resourceType);
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
      this.refresh();
    },

    deactivate() {},
  }));
});
