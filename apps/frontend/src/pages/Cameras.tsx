import { useMemo, useState } from "react";

import ZoneEditor from "../components/ZoneEditor";
import { api } from "../lib/api";
import type { CameraConfig, WebcamDiscoveryItem } from "../lib/types";

interface Props {
  cameras: CameraConfig[];
  onChanged: () => Promise<void>;
}

const emptyForm: Omit<CameraConfig, "id"> = {
  name: "",
  kind: "webcam",
  source: "0",
  enabled: true,
  detection_enabled: false,
  recording_mode: "event_only",
  labels: ["person", "animal", "vehicle", "unknown"],
  min_confidence: { person: 0.35, animal: 0.3, vehicle: 0.35, unknown: 0.45 },
  cooldown_seconds: 8,
  motion_threshold: 0.012,
  zones: []
};

export default function CamerasPage({ cameras, onChanged }: Props) {
  const [form, setForm] = useState(emptyForm);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [onvif, setOnvif] = useState<Array<{ id: string; host: string; xaddr: string; rtsp_candidates: string[] }>>([]);
  const [webcams, setWebcams] = useState<WebcamDiscoveryItem[]>([]);
  const [discoveringWebcams, setDiscoveringWebcams] = useState(false);

  const isRtsp = useMemo(() => form.kind === "rtsp" || form.kind === "onvif", [form.kind]);
  const isWebcam = useMemo(() => form.kind === "webcam", [form.kind]);
  const discoveredSelection = useMemo(() => (webcams.some((item) => String(item.index) === form.source) ? form.source : ""), [form.source, webcams]);

  return (
    <section className="space-y-4">
      <article className="card rounded-xl p-4 shadow-panel">
        <h2 className="mb-3 text-lg font-semibold">Add Camera</h2>
        <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
          <label>
            <span className="mb-1 block text-xs opacity-70">Name</span>
            <input value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} className="w-full rounded border border-slate-400/40 bg-transparent px-2 py-1" />
          </label>

          <label>
            <span className="mb-1 block text-xs opacity-70">Kind</span>
            <select value={form.kind} onChange={(e) => setForm({ ...form, kind: e.target.value as CameraConfig["kind"] })} className="w-full rounded border border-slate-400/40 bg-transparent px-2 py-1">
              <option value="webcam">Webcam (OpenCV index)</option>
              <option value="rtsp">RTSP URL</option>
              <option value="onvif">ONVIF / RTSP</option>
            </select>
          </label>

          <label>
            <span className="mb-1 block text-xs opacity-70">{isRtsp ? "RTSP URL" : "Webcam source/index (manual fallback)"}</span>
            <input value={form.source} onChange={(e) => setForm({ ...form, source: e.target.value })} className="w-full rounded border border-slate-400/40 bg-transparent px-2 py-1" />
          </label>

          <label className="flex items-end gap-2 text-xs">
            <input type="checkbox" checked={form.enabled} onChange={(e) => setForm({ ...form, enabled: e.target.checked })} />
            Enabled
          </label>
        </div>

        {isWebcam ? (
          <div className="mt-3 rounded border border-slate-400/30 p-3">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div>
                <p className="text-xs font-semibold">Local Webcam Discovery</p>
                <p className="text-xs opacity-70">Find available local webcam sources and select one.</p>
              </div>
              <button
                className="rounded border border-slate-400/40 px-3 py-1 text-xs hover:bg-slate-500/10 disabled:cursor-not-allowed disabled:opacity-60"
                disabled={discoveringWebcams}
                onClick={async () => {
                  setError("");
                  setMessage("");
                  setDiscoveringWebcams(true);
                  try {
                    const res = await api.discoverWebcams();
                    const items = (res as { items?: WebcamDiscoveryItem[] }).items ?? [];
                    setWebcams(items);
                    if (items.length === 0) {
                      setMessage("No webcams discovered. Enter a source manually to continue.");
                    } else {
                      const preferred = items.find((item) => item.is_default) ?? items[0];
                      setForm((prev) => ({ ...prev, source: String(preferred.index) }));
                      setMessage(`Discovered ${items.length} webcam source${items.length === 1 ? "" : "s"}.`);
                    }
                  } catch (e) {
                    setError((e as Error).message || "Webcam discovery failed");
                  } finally {
                    setDiscoveringWebcams(false);
                  }
                }}
              >
                {discoveringWebcams ? "Discovering..." : "Discover Local Webcams"}
              </button>
            </div>

            {webcams.length > 0 ? (
              <label className="mt-3 block">
                <span className="mb-1 block text-xs opacity-70">Discovered webcams</span>
                <select
                  value={discoveredSelection}
                  onChange={(e) => {
                    const nextSource = e.target.value;
                    if (!nextSource) return;
                    setForm((prev) => ({ ...prev, source: nextSource }));
                  }}
                  className="w-full rounded border border-slate-400/40 bg-transparent px-2 py-1"
                >
                  <option value="">Select discovered webcam...</option>
                  {webcams.map((item) => (
                    <option key={`${item.index}-${item.label}`} value={String(item.index)}>
                      {item.label} (index {item.index}) - {item.status}
                      {item.is_default ? " | default" : ""}
                    </option>
                  ))}
                </select>
              </label>
            ) : null}
          </div>
        ) : null}

        <details className="mt-3 rounded border border-slate-400/30 p-3">
          <summary className="cursor-pointer text-xs font-semibold">Advanced camera settings</summary>
          <div className="mt-3 grid grid-cols-1 gap-3 md:grid-cols-2">
            <label className="flex items-end gap-2 text-xs">
              <input
                type="checkbox"
                checked={form.detection_enabled}
                onChange={(e) => setForm({ ...form, detection_enabled: e.target.checked })}
              />
              Detector enabled
            </label>

            <label>
              <span className="mb-1 block text-xs opacity-70">Recording mode</span>
              <select
                value={form.recording_mode}
                onChange={(e) =>
                  setForm({
                    ...form,
                    recording_mode: e.target.value as CameraConfig["recording_mode"]
                  })
                }
                className="w-full rounded border border-slate-400/40 bg-transparent px-2 py-1"
              >
                <option value="event_only">Event only (default)</option>
                <option value="full">Full recording (5 min segments)</option>
                <option value="live">Live recording (1 min segments)</option>
              </select>
            </label>

            <label>
              <span className="mb-1 block text-xs opacity-70">Cooldown (sec)</span>
              <input
                type="number"
                value={form.cooldown_seconds}
                onChange={(e) => setForm({ ...form, cooldown_seconds: Number(e.target.value) })}
                className="w-full rounded border border-slate-400/40 bg-transparent px-2 py-1"
              />
            </label>

            <label>
              <span className="mb-1 block text-xs opacity-70">Motion threshold</span>
              <input
                type="number"
                min={0}
                max={1}
                step={0.001}
                value={form.motion_threshold}
                onChange={(e) => setForm({ ...form, motion_threshold: Number(e.target.value) })}
                className="w-full rounded border border-slate-400/40 bg-transparent px-2 py-1"
              />
            </label>
          </div>
          <div className="mt-3">
            <p className="mb-2 text-xs opacity-70">Detection zones</p>
            <ZoneEditor value={form.zones} onChange={(zones) => setForm({ ...form, zones })} />
          </div>
        </details>

        <div className="mt-3 flex flex-wrap gap-2">
          <button
            className="rounded border border-slate-400/40 px-3 py-1 text-xs hover:bg-slate-500/10"
            onClick={async () => {
              setError("");
              setMessage("");
              try {
                const result = await api.testCamera(form);
                setMessage((result as any).message);
              } catch (e) {
                setError((e as Error).message || "Camera test failed");
              }
            }}
          >
            Test source
          </button>
          <button
            className="rounded bg-teal-600 px-3 py-1 text-xs font-medium text-white hover:bg-teal-500"
            onClick={async () => {
              setError("");
              setMessage("");
              try {
                await api.createCamera(form);
                setForm(emptyForm);
                setMessage("Camera saved");
                await onChanged();
              } catch (e) {
                setError((e as Error).message || "Failed to save camera");
              }
            }}
          >
            Save camera
          </button>
          <button
            className="rounded border border-slate-400/40 px-3 py-1 text-xs hover:bg-slate-500/10"
            onClick={async () => {
              setError("");
              try {
                const res = await api.discoverOnvif();
                setOnvif((res as any).items ?? []);
              } catch (e) {
                setError((e as Error).message || "ONVIF discovery failed");
              }
            }}
          >
            Discover ONVIF
          </button>
        </div>

        {message ? <p className="mt-2 text-xs text-emerald-400">{message}</p> : null}
        {error ? <p className="mt-2 text-xs text-rose-400">{error}</p> : null}

        {onvif.length > 0 ? (
          <div className="mt-3 rounded border border-slate-400/30 p-2 text-xs">
            <p className="mb-2 font-semibold">ONVIF endpoints</p>
            {onvif.map((endpoint) => (
              <div key={endpoint.id} className="mb-2 rounded border border-slate-500/30 p-2">
                <p>{endpoint.host}</p>
                <p className="opacity-70">{endpoint.xaddr}</p>
                <p className="mt-1 opacity-80">Candidates: {endpoint.rtsp_candidates.join(" | ")}</p>
              </div>
            ))}
          </div>
        ) : null}
      </article>

      <article className="card rounded-xl p-4 shadow-panel">
        <h3 className="mb-2 text-sm font-semibold">Configured Cameras</h3>
        {cameras.length === 0 ? <p className="text-xs opacity-70">No cameras configured.</p> : null}
        <div className="space-y-2">
          {cameras.map((camera) => (
            <div key={camera.id} className="flex flex-wrap items-center justify-between gap-2 rounded border border-slate-400/30 p-2 text-xs">
              <div>
                <p className="font-semibold">{camera.name}</p>
                <p className="opacity-70">
                  {camera.kind} | {camera.source}
                </p>
                <p className="opacity-70">
                  detector: {camera.detection_enabled ? "on" : "off"} | recording: {camera.recording_mode}
                </p>
              </div>
              <button
                className="rounded border border-rose-500/50 px-2 py-1 text-rose-400"
                onClick={async () => {
                  await api.deleteCamera(camera.id);
                  await onChanged();
                }}
              >
                Remove
              </button>
            </div>
          ))}
        </div>
      </article>
    </section>
  );
}
