// Civitai browser, nested inside the "models" component: modelType, resourceType and view come from there.
document.addEventListener("alpine:init", () => {
  const PAGE_SIZE = 24;
  const SEARCH_DEBOUNCE_MS = 400;
  const JOB_POLL_MS = 1500;
  const THUMB_WIDTH = 450;
  const FINISHED = ["completed", "failed", "cancelled"];
  const LAYOUT_PARTS = { clip_l: "CLIP-L", t5xxl: "T5-XXL", vae: "VAE" };

  Alpine.data("civitai", () => ({
    status: null,
    statusError: null,
    query: "",
    sort: "Highest Rated",
    period: "AllTime",
    nsfw: false,
    items: [],
    nextCursor: null,
    loading: false,
    loadingMore: false,
    error: null,
    searchTimer: null,
    searchSeq: 0,
    // model/resource type the current results are for
    resultsFor: null,
    detail: { open: false, summary: null, model: null, versionId: null, error: null },
    layouts: {},
    downloads: {},
    pollHandle: null,

    init() {
      this.$watch("view", (view) => view === "civitai" && this.ensureLoaded());
      this.$watch("modelType", () => this.view === "civitai" && this.search());
      this.$watch("resourceType", () => this.view === "civitai" && this.search());
    },

    get version() {
      return this.detail.model?.versions.find((v) => v.id === this.detail.versionId) ?? null;
    },

    get description() {
      return htmlToText(this.detail.model?.description);
    },

    formatBytes,
    formatCount,

    // Civitai serves resized renditions: swap the original for a thumbnail-sized one.
    thumbUrl(preview) {
      return preview.url.replace("/original=true/", `/width=${THUMB_WIDTH}/`);
    },

    isBlurred(preview) {
      return Boolean(preview) && preview.nsfw_level > 1 && !this.nsfw;
    },

    civitaiUrl(model) {
      return `https://civitai.com/models/${model.id}`;
    },

    async ensureLoaded() {
      if (!this.status) {
        try {
          this.status = await Api.civitaiStatus();
          this.nsfw = this.status.nsfw_default;
          this.sort = this.status.sort_options[0];
          this.period = this.status.period_options[0];
        } catch (e) {
          this.statusError = e.message;
          return;
        }
      }
      if (this.resultsFor !== this.filterKey() && !this.loading) this.search();
    },

    filterKey() {
      return `${this.modelType}/${this.resourceType}`;
    },

    onQueryInput(value) {
      this.query = value;
      clearTimeout(this.searchTimer);
      this.searchTimer = setTimeout(() => this.search(), SEARCH_DEBOUNCE_MS);
    },

    setFilter(field, value) {
      if (this[field] === value) return;
      this[field] = value;
      this.search();
    },

    params(cursor) {
      return {
        model_type: this.modelType,
        resource_type: this.resourceType,
        query: this.query.trim(),
        sort: this.sort,
        period: this.period,
        nsfw: this.nsfw,
        limit: PAGE_SIZE,
        cursor,
      };
    },

    async search() {
      clearTimeout(this.searchTimer);
      if (!this.status?.enabled) return;
      // Results of an older search that finishes late are dropped
      const seq = ++this.searchSeq;
      this.loading = true;
      try {
        const data = await Api.civitaiSearch(this.params());
        if (seq !== this.searchSeq) return;
        this.items = data.items;
        this.nextCursor = data.next_cursor;
        this.resultsFor = this.filterKey();
        this.error = null;
      } catch (e) {
        if (seq !== this.searchSeq) return;
        this.items = [];
        this.nextCursor = null;
        this.error = e.message;
      } finally {
        if (seq === this.searchSeq) this.loading = false;
      }
    },

    async loadMore() {
      if (!this.nextCursor || this.loadingMore) return;
      const seq = this.searchSeq;
      this.loadingMore = true;
      try {
        const data = await Api.civitaiSearch(this.params(this.nextCursor));
        if (seq !== this.searchSeq) return;
        const seen = new Set(this.items.map((item) => item.id));
        this.items.push(...data.items.filter((item) => !seen.has(item.id)));
        this.nextCursor = data.next_cursor;
      } catch (e) {
        notify(`Could not load more results: ${e.message}`);
      } finally {
        this.loadingMore = false;
      }
    },

    async openDetail(summary) {
      this.detail = { open: true, summary, model: null, versionId: null, error: null };
      try {
        const model = await Api.civitaiModel(summary.id, this.modelType, this.nsfw);
        if (this.detail.summary?.id !== summary.id) return;
        this.detail.model = model;
        this.detail.versionId = model.versions[0]?.id ?? null;
      } catch (e) {
        this.detail.error = e.message;
      }
    },

    canCheckLayout(version, file) {
      return version.model_type === "flux" && file.target_resource_type === "checkpoints" && file.format === "SafeTensor";
    },

    layoutText(layout) {
      if (!layout.transformer) return "No Flux transformer found in this file";
      const parts = Object.entries(LAYOUT_PARTS).filter(([key]) => layout[key]).map(([, label]) => label);
      return parts.length ? `Bundled: transformer + ${parts.join(", ")}` : "Transformer only (uses the shared Flux components)";
    },

    async checkLayout(version, file) {
      this.layouts[file.id] = { loading: true };
      try {
        this.layouts[file.id] = { data: await Api.civitaiLayout(version.id, file.id) };
      } catch (e) {
        this.layouts[file.id] = { error: e.status === 401 ? "Needs a Civitai API key to read this file" : e.message };
      }
    },

    downloadState(file) {
      return this.downloads[file.id] ?? null;
    },

    async download(version, file, acceptMismatch = false) {
      try {
        const started = await Api.civitaiDownload({
          version_id: version.id,
          file_id: file.id,
          model_type: this.modelType,
          resource_type: file.target_resource_type,
          accept_mismatch: acceptMismatch,
        });
        this.downloads[file.id] = {
          jobId: started.job_id, status: "queued", progress: 0, filename: started.filename,
          modelType: this.modelType, resourceType: file.target_resource_type,
        };
        notify(`Downloading ${started.filename}: follow it here or in the Jobs tab`, "success");
        this.startPolling();
      } catch (e) {
        const detail = e.detail;
        if (e.status === 409 && detail?.warnings) {
          const confirmed = await this.$store.confirm.ask(
            "Different base model", `${detail.warnings.join(" ")} Download it anyway?`, "Download anyway",
          );
          if (confirmed) await this.download(version, file, true);
        } else if (e.status === 409 && detail?.path) {
          file.local_path = detail.path;
          notify(`${file.name} is already downloaded`, "neutral");
        } else if (e.status === 409 && detail?.job_id) {
          notify(`${file.name} is already being downloaded`, "neutral");
        } else {
          notify(`Could not download ${file.name}: ${typeof detail === "string" ? detail : e.message}`);
        }
      }
    },

    startPolling() {
      if (!this.pollHandle) this.pollHandle = setInterval(() => this.pollDownloads(), JOB_POLL_MS);
    },

    async pollDownloads() {
      const active = Object.entries(this.downloads).filter(([, d]) => !FINISHED.includes(d.status));
      if (!active.length) {
        clearInterval(this.pollHandle);
        this.pollHandle = null;
        return;
      }
      for (const [fileId, state] of active) {
        let job;
        try {
          job = await Api.getJob(state.jobId);
        } catch (e) {
          continue;
        }
        Object.assign(state, { status: job.status, progress: job.progress });
        if (job.status === "completed") {
          notify(`Downloaded ${state.filename}`, "success");
          this.markDownloaded(Number(fileId), job.result.path);
          window.dispatchEvent(new CustomEvent("models-changed", {
            detail: { modelType: state.modelType, resourceType: state.resourceType },
          }));
        } else if (job.status === "failed") {
          notify(`Download of ${state.filename} failed: ${job.error}`);
        }
      }
    },

    markDownloaded(fileId, path) {
      for (const version of this.detail.model?.versions ?? []) {
        for (const file of version.files) {
          if (file.id === fileId) file.local_path = path;
        }
      }
    },
  }));
});
