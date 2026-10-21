const Api = (() => {
  const BASE = "/api/v1";

  async function request(path, options) {
    const res = await fetch(`${BASE}${path}`, options);
    if (!res.ok) {
      const body = await res.text().catch(() => "");
      throw new Error(`${res.status} ${res.statusText}: ${body}`);
    }
    if (res.status === 204) return null;
    return res.json();
  }

  return {
    listJobs: () => request("/jobs/"),
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

    getSystemInfo: () => request("/system/info"),
  };
})();
