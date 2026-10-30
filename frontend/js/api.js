/**
 * API client. One place that knows the URLs and how errors surface, so views
 * only deal with data or a thrown Error with a readable message.
 */

const BASE = "";

async function request(path, { method = "GET", body, formData = null } = {}) {
  const options = { method, headers: {} };
  if (formData) {
    options.body = formData; // let the browser set the multipart boundary
  } else if (body !== undefined) {
    options.headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(body);
  }

  let response;
  try {
    response = await fetch(`${BASE}${path}`, options);
  } catch (cause) {
    throw new Error(`Cannot reach the API at ${path}. Is the server running?`);
  }

  if (response.status === 204) return null;

  const text = await response.text();
  let payload = null;
  if (text) {
    try { payload = JSON.parse(text); } catch { payload = { detail: text }; }
  }

  if (!response.ok) {
    let detail = payload?.detail ?? `HTTP ${response.status}`;
    if (Array.isArray(detail)) {
      // FastAPI validation errors
      detail = detail.map((d) => `${(d.loc || []).slice(1).join(".")}: ${d.msg}`).join("; ");
    }
    throw new Error(detail);
  }
  return payload;
}

export const api = {
  health: () => request("/api/health"),
  stats: () => request("/api/stats"),
  datasetPreview: (limit = 10) => request(`/api/dataset/preview?limit=${limit}`),
  defaults: () => request("/api/config/defaults"),

  listRuns: (params = {}) => {
    const qs = new URLSearchParams(
      Object.entries(params).filter(([, v]) => v !== undefined && v !== null && v !== ""),
    ).toString();
    return request(`/api/runs${qs ? `?${qs}` : ""}`);
  },
  getRun: (id) => request(`/api/runs/${encodeURIComponent(id)}`),
  createRun: (payload) => request("/api/runs", { method: "POST", body: payload }),
  evaluateRun: (id) => request(`/api/runs/${encodeURIComponent(id)}/evaluate`, { method: "POST" }),
  registerRun: (id, payload) =>
    request(`/api/runs/${encodeURIComponent(id)}/register`, { method: "POST", body: payload }),
  deleteRun: (id) => request(`/api/runs/${encodeURIComponent(id)}`, { method: "DELETE" }),

  listModels: (params = {}) => {
    const qs = new URLSearchParams(params).toString();
    return request(`/api/models${qs ? `?${qs}` : ""}`);
  },
  productionModel: () => request("/api/models/production"),
  promote: (id, payload) => request(`/api/models/${id}/promote`, { method: "POST", body: payload }),
  deleteVersion: (id) => request(`/api/models/${id}`, { method: "DELETE" }),

  predict: (payload) => request("/api/predict", { method: "POST", body: payload }),
  predictUpload: ({ file, threshold, stage, model_version_id }) => {
    const fd = new FormData();
    fd.append("file", file);
    fd.append("threshold", String(threshold));
    if (stage) fd.append("stage", stage);
    if (model_version_id) fd.append("model_version_id", String(model_version_id));
    return request("/api/predict/upload", { method: "POST", formData: fd });
  },
  listPredictions: (limit = 10) => request(`/api/predictions?limit=${limit}`),
};
