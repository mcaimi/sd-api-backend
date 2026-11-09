document.addEventListener("alpine:init", () => {
  const PAGE_SIZE = 24;
  const REMOVE_ANIMATION_MS = 250;

  Alpine.data("outputs", () => tabComponent("outputs", {
    pageSize: PAGE_SIZE,
    items: [],
    total: 0,
    offset: 0,
    loading: true,
    error: null,
    removing: {},
    viewer: { open: false, item: null, meta: null, error: null },

    get page() {
      return Math.floor(this.offset / PAGE_SIZE) + 1;
    },

    // Splits persisted metadata into prompts (shown as blocks) and scalar settings (shown as a grid).
    get viewerFields() {
      const generation = this.viewer.meta?.generation;
      const fields = { prompts: [], settings: [], loras: [], vae: this.viewer.meta?.vae };
      if (generation && typeof generation === "object") {
        for (const [key, value] of Object.entries(generation)) {
          const label = key.replace(/_/g, " ");
          if (key.includes("prompt") && typeof value === "string") fields.prompts.push([label, value]);
          else if (value !== null && typeof value !== "object") fields.settings.push([label, String(value)]);
        }
      }
      for (const lora of Object.values(this.viewer.meta?.loras || {})) {
        fields.loras.push(`${lora.name} (${lora.merge_strength})`);
      }
      return fields;
    },

    get viewerJson() {
      return JSON.stringify(this.viewer.meta, null, 2);
    },

    imageUrl(id) {
      return Api.outputImageUrl(id);
    },

    formatTimestamp,

    goTo(page) {
      this.offset = (page - 1) * PAGE_SIZE;
      this.refresh();
    },

    async openViewer(item) {
      this.viewer = { open: true, item, meta: null, error: null };
      try {
        this.viewer.meta = await Api.getOutputMetadata(item.id);
      } catch (e) {
        this.viewer.error = e.message;
      }
    },

    async deleteItem(item) {
      const confirmed = await this.$store.confirm.ask(
        "Delete output?",
        "This removes the image and its metadata permanently.",
        "Delete",
      );
      if (!confirmed) return;
      try {
        await Api.deleteOutput(item.id);
      } catch (e) {
        notify(`Could not delete output: ${e.message}`);
        return;
      }
      if (this.viewer.item?.id === item.id) this.viewer.open = false;
      this.removing[item.id] = true;
      await sleep(REMOVE_ANIMATION_MS);
      if (this.items.length === 1 && this.offset > 0) {
        this.offset = Math.max(0, this.offset - PAGE_SIZE);
      }
      await this.refresh();
      delete this.removing[item.id];
    },

    async refresh() {
      try {
        const data = await Api.listOutputs(PAGE_SIZE, this.offset);
        this.items = data.items;
        this.total = data.total;
        this.error = null;
      } catch (e) {
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
