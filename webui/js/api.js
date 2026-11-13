const Api = (() => {
  const BASE = "/api/v1";

  // Errors carry the HTTP status and FastAPI's parsed `detail` (a string, or an object for structured errors).
  async function request(path, options) {
    const res = await fetch(`${BASE}${path}`, options);
    if (!res.ok) {
      const body = await res.text().catch(() => "");
      let detail = body;
      try {
        detail = JSON.parse(body).detail ?? body;
      } catch {}
      const text = typeof detail === "string" ? detail : detail?.message ?? body;
      const error = new Error(`${res.status} ${res.statusText}: ${text}`);
      error.status = res.status;
      error.detail = detail;
      throw error;
    }
    if (res.status === 204) return null;
    return res.json();
  }

  const postJson = (path, body) =>
    request(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

  // Drops null/undefined/empty values so optional query parameters fall back to server defaults.
  const query = (params) => new URLSearchParams(
    Object.entries(params).filter(([, value]) => value !== null && value !== undefined && value !== ""),
  ).toString();

  return {
    listJobs: () => request("/jobs/"),
    getJob: (jobId) => request(`/jobs/${jobId}`),
    cancelJob: (jobId) => request(`/jobs/${jobId}`, { method: "DELETE" }),

    listOutputs: (limit, offset) =>
      request(`/outputs/?limit=${limit}&offset=${offset}`),
    outputImageUrl: (id) => `${BASE}/outputs/${id}/image`,
    getOutputMetadata: (id) => request(`/outputs/${id}/metadata`),
    deleteOutput: (id) => request(`/outputs/${id}`, { method: "DELETE" }),

    listModels: (modelType, resourceType) =>
      request(`/models/${modelType}/${resourceType}`),
    getModelMetadata: (modelType, resourceType, name) =>
      request(`/models/${modelType}/${resourceType}/${encodeURIComponent(name)}/metadata`),
    modelDownloadUrl: (modelType, resourceType, name) =>
      `${BASE}/models/${modelType}/${resourceType}/${encodeURIComponent(name)}/download`,
    modelPreviewUrl: (modelType, resourceType, name) =>
      `${BASE}/models/${modelType}/${resourceType}/${encodeURIComponent(name)}/preview`,

    civitaiStatus: () => request("/civitai/status"),
    civitaiSearch: (params) => request(`/civitai/search?${query(params)}`),
    civitaiModel: (id, modelType, nsfw) =>
      request(`/civitai/models/${id}?${query({ model_type: modelType, nsfw })}`),
    civitaiLayout: (versionId, fileId) =>
      request(`/civitai/files/layout?${query({ version_id: versionId, file_id: fileId })}`),
    civitaiDownload: (body) => postJson("/civitai/downloads", body),
    civitaiIdentify: (body) => postJson("/civitai/identify", body),

    getSystemInfo: () => request("/system/info"),
  };
})();
