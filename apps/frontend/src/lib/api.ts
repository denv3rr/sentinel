const jsonHeaders = { "Content-Type": "application/json" };

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, init);
  if (!res.ok) {
    const body = await res.text();
    throw new Error(body || `Request failed (${res.status})`);
  }
  return (await res.json()) as T;
}

export const api = {
  getSettings: () => request("/api/settings"),
  listCameras: () => request<{ items: any[] }>("/api/cameras"),
  createCamera: (payload: any) => request("/api/cameras", { method: "POST", headers: jsonHeaders, body: JSON.stringify(payload) }),
  updateCamera: (id: string, payload: any) => request(`/api/cameras/${id}`, { method: "PUT", headers: jsonHeaders, body: JSON.stringify(payload) }),
  deleteCamera: (id: string) => request(`/api/cameras/${id}`, { method: "DELETE" }),
  testCamera: (payload: any) => request<{ ok: boolean; message: string }>("/api/cameras/test", { method: "POST", headers: jsonHeaders, body: JSON.stringify(payload) }),
  discoverOnvif: () => request<{ items: any[] }>("/api/cameras/discover/onvif"),
  detectDefaultWebcam: () => request<{ ok: boolean; index: string }>("/api/cameras/default-webcam"),
  getEvents: (params: Record<string, string | number | boolean | undefined>, init?: RequestInit) => {
    const qs = new URLSearchParams();
    Object.entries(params).forEach(([k, v]) => {
      if (v !== undefined && v !== "") qs.set(k, String(v));
    });
    return request(`/api/events?${qs.toString()}`, init);
  },
  markReviewed: (id: string, reviewed: boolean) => request(`/api/events/${id}/review`, { method: "POST", headers: jsonHeaders, body: JSON.stringify({ reviewed }) }),
  exportEvents: (payload: any) =>
    request<{ ok: boolean; path: string; count: number; format: string; download_url: string | null }>("/api/exports", {
      method: "POST",
      headers: jsonHeaders,
      body: JSON.stringify(payload)
    }),
  setDataDir: (path: string) => request("/api/settings/data-dir", { method: "POST", headers: jsonHeaders, body: JSON.stringify({ path }) }),
  setExportDir: (path: string | null) => request("/api/settings/export-dir", { method: "POST", headers: jsonHeaders, body: JSON.stringify({ path }) }),
  pickDataDir: () => request<{ path: string }>("/api/settings/pick-data-dir", { method: "POST" }),
  setRetention: (payload: { days: number; max_gb: number | null }) => request("/api/settings/retention", { method: "POST", headers: jsonHeaders, body: JSON.stringify(payload) }),
  setThresholds: (thresholds: Record<string, number>) => request("/api/settings/thresholds", { method: "POST", headers: jsonHeaders, body: JSON.stringify({ thresholds }) }),
  setLan: (allow_lan: boolean) => request("/api/settings/lan", { method: "POST", headers: jsonHeaders, body: JSON.stringify({ allow_lan }) }),
  setArm: (armed: boolean) => request("/api/settings/arm", { method: "POST", headers: jsonHeaders, body: JSON.stringify({ armed }) }),
  exitRuntime: () => request<{ ok: boolean; message: string }>("/api/settings/exit", { method: "POST" }),
  health: () => request("/api/health")
};
