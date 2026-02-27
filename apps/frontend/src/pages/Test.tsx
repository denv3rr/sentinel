import { useEffect, useMemo, useState } from "react";

import { api } from "../lib/api";
import { formatTimestamp, relativeTime } from "../lib/time";
import type { CameraConfig, EventItem } from "../lib/types";

interface Props {
  cameras: CameraConfig[];
  statuses: Record<string, { online?: boolean; fps?: number; last_error?: string }>;
  onChanged: () => Promise<void>;
  armed: boolean;
}

type RuntimeStatus = { online?: boolean; fps?: number; last_error?: string };

function buildDefaultCameraPayload(name: string, source: string): Omit<CameraConfig, "id"> {
  return {
    name,
    kind: "webcam",
    source,
    enabled: true,
    detection_enabled: true,
    recording_mode: "event_only",
    labels: ["person", "animal", "vehicle", "unknown"],
    min_confidence: { person: 0.35, animal: 0.3, vehicle: 0.35, unknown: 0.45 },
    cooldown_seconds: 8,
    motion_threshold: 0.012,
    zones: []
  };
}

function choosePreferredCamera(cameras: CameraConfig[], statuses: Record<string, RuntimeStatus>): CameraConfig | null {
  if (cameras.length === 0) {
    return null;
  }

  const onlineWebcam = cameras.find((camera) => camera.kind === "webcam" && statuses[camera.id]?.online);
  if (onlineWebcam) {
    return onlineWebcam;
  }

  const onlineCamera = cameras.find((camera) => statuses[camera.id]?.online);
  if (onlineCamera) {
    return onlineCamera;
  }

  return cameras.find((camera) => camera.kind === "webcam") ?? cameras[0];
}

export default function TestPage({ cameras, statuses, onChanged, armed }: Props) {
  const [cameraId, setCameraId] = useState("");
  const [manualSelection, setManualSelection] = useState(false);
  const [note, setNote] = useState("Preparing test monitor...");
  const [error, setError] = useState("");
  const [events, setEvents] = useState<EventItem[]>([]);
  const [busy, setBusy] = useState(false);
  const [toggleBusy, setToggleBusy] = useState(false);

  const selectedCamera = useMemo(() => cameras.find((c) => c.id === cameraId), [cameras, cameraId]);
  const status = selectedCamera ? statuses[selectedCamera.id] : undefined;

  async function bootstrapQuickTest(): Promise<void> {
    if (busy) return;
    setBusy(true);
    setError("");
    setNote("Detecting default webcam...");

    try {
      const detected = await api.detectDefaultWebcam();
      const webcamIndex = (detected as { index: string }).index;
      const exactMatch = cameras.find((camera) => camera.kind === "webcam" && camera.source === webcamIndex);
      const reusableDefault = cameras.find((camera) => camera.kind === "webcam" && camera.name.toLowerCase() === "default webcam");
      let activeCamera: CameraConfig | null = null;

      if (exactMatch) {
        if (!exactMatch.enabled || !exactMatch.detection_enabled) {
          setNote(`Using webcam ${webcamIndex}. Enabling monitor and detector...`);
          const updated = {
            ...exactMatch,
            enabled: true,
            detection_enabled: true
          };
          const result = (await api.updateCamera(exactMatch.id, updated)) as { camera?: CameraConfig };
          activeCamera = result.camera ?? updated;
        } else {
          setNote(`Using existing webcam ${webcamIndex}.`);
          activeCamera = exactMatch;
        }
      } else if (reusableDefault) {
        setNote(`Updating existing default webcam to source ${webcamIndex}...`);
        const updated = {
          ...reusableDefault,
          source: webcamIndex,
          enabled: true,
          detection_enabled: true
        };
        const result = (await api.updateCamera(reusableDefault.id, updated)) as { camera?: CameraConfig };
        activeCamera = result.camera ?? updated;
      } else {
        setNote(`Webcam ${webcamIndex} detected. Creating quick monitor camera...`);
        const payload = buildDefaultCameraPayload("Default Webcam", webcamIndex);
        const created = (await api.createCamera(payload)) as { camera: CameraConfig };
        activeCamera = created.camera;
      }

      if (activeCamera) {
        setManualSelection(false);
        setCameraId(activeCamera.id);
      }

      if (!armed) {
        setNote("Arming detection so event logs are active...");
        await api.setArm(true);
      }

      await onChanged();
      setNote("Quick Test is active. Move in frame to confirm box overlays and event logs.");
    } catch (e) {
      setError((e as Error).message || "Failed to start quick test");
      setNote("");
    } finally {
      setBusy(false);
    }
  }

  async function loadEvents(currentCameraId: string): Promise<void> {
    try {
      const res = (await api.getEvents({ camera_id: currentCameraId, limit: 30, sort: "desc" })) as {
        items: EventItem[];
      };
      setEvents(res.items ?? []);
    } catch {
      // Keep existing list when refresh fails.
    }
  }

  useEffect(() => {
    if (cameras.length === 0) {
      if (!cameraId) {
        setNote("No cameras configured yet. Run auto setup to detect and add your default webcam.");
      }
      return;
    }

    if (cameraId && cameras.some((camera) => camera.id === cameraId)) {
      return;
    }

    const preferredCamera = choosePreferredCamera(cameras, statuses);
    if (preferredCamera) {
      setManualSelection(false);
      setCameraId(preferredCamera.id);
      if (statuses[preferredCamera.id]?.online) {
        setNote(`Using active camera "${preferredCamera.name}".`);
      } else {
        setNote(`Using configured camera "${preferredCamera.name}".`);
      }
    } else {
      setNote("No cameras configured yet. Run auto setup to detect and add your default webcam.");
    }
  }, [cameraId, cameras, statuses]);

  useEffect(() => {
    if (!cameraId || manualSelection) return;
    const current = cameras.find((camera) => camera.id === cameraId);
    if (!current) return;
    if (statuses[current.id]?.online) return;

    const onlineWebcam = cameras.find(
      (camera) => camera.id !== current.id && camera.kind === "webcam" && statuses[camera.id]?.online
    );
    const onlineFallback = onlineWebcam ?? cameras.find((camera) => camera.id !== current.id && statuses[camera.id]?.online);
    if (!onlineFallback) return;

    setCameraId(onlineFallback.id);
    if (onlineFallback.kind === "webcam") {
      setNote(`"${current.name}" is offline. Switched to active webcam "${onlineFallback.name}".`);
    } else {
      setNote(`"${current.name}" is offline. Switched to active input "${onlineFallback.name}".`);
    }
  }, [cameraId, manualSelection, cameras, statuses]);

  useEffect(() => {
    if (!cameraId) return;
    void loadEvents(cameraId);
    const timer = window.setInterval(() => {
      void loadEvents(cameraId);
    }, 4000);
    return () => window.clearInterval(timer);
  }, [cameraId]);

  return (
    <section className="space-y-4">
      <article className="card rounded-xl p-4 shadow-panel">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div>
            <h2 className="text-lg font-semibold">Test Monitor</h2>
            <p className="text-sm opacity-80">
              Auto-detects your default webcam and runs regular monitoring with overlays + event logs.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <label className="flex items-center gap-2 text-xs">
              <span className="opacity-70">Input</span>
              <select
                value={cameraId}
                onChange={(e) => {
                  const nextCameraId = e.target.value;
                  setManualSelection(true);
                  setCameraId(nextCameraId);
                  const nextCamera = cameras.find((camera) => camera.id === nextCameraId);
                  if (nextCamera) {
                    setNote(`Switched to "${nextCamera.name}".`);
                  }
                }}
                className="min-w-[220px] rounded border border-slate-400/40 bg-transparent px-2 py-1"
                disabled={cameras.length === 0}
              >
                {cameras.length === 0 ? <option value="">No inputs available</option> : null}
                {cameras.map((camera) => {
                  const cameraStatus = statuses[camera.id];
                  const availability = cameraStatus?.online ? "online" : "offline";
                  return (
                    <option key={camera.id} value={camera.id}>
                      {camera.name} ({camera.kind}) - {availability}
                    </option>
                  );
                })}
              </select>
            </label>
            <button
              className="rounded border border-slate-400/40 px-3 py-1 text-xs hover:bg-slate-500/10"
              onClick={() => void bootstrapQuickTest()}
              disabled={busy}
            >
              {busy ? "Starting..." : "Run auto setup"}
            </button>
            <button
              className="rounded border border-slate-400/40 px-3 py-1 text-xs hover:bg-slate-500/10"
              onClick={async () => {
                await api.setArm(!armed);
                await onChanged();
              }}
            >
              {armed ? "Disarm System" : "Arm System"}
            </button>
            <button
              className="rounded border border-slate-400/40 px-3 py-1 text-xs hover:bg-slate-500/10 disabled:cursor-not-allowed disabled:opacity-50"
              disabled={!selectedCamera || toggleBusy}
              onClick={async () => {
                if (!selectedCamera || toggleBusy) return;
                setToggleBusy(true);
                setError("");
                try {
                  const updated = {
                    ...selectedCamera,
                    detection_enabled: !selectedCamera.detection_enabled,
                    enabled: true
                  };
                  if (updated.detection_enabled && !armed) {
                    await api.setArm(true);
                  }
                  await api.updateCamera(selectedCamera.id, updated);
                  await onChanged();
                  if (updated.detection_enabled && !armed) {
                    setNote("Detector enabled and system armed for selected camera.");
                  } else {
                    setNote(updated.detection_enabled ? "Detector enabled for selected camera." : "Detector disabled for selected camera.");
                  }
                } catch (e) {
                  setError((e as Error).message || "Failed to update detector state");
                } finally {
                  setToggleBusy(false);
                }
              }}
            >
              {selectedCamera?.detection_enabled ? "Detector Off" : "Detector On"}
            </button>
          </div>
        </div>

        <div className="mt-3 grid grid-cols-1 gap-2 text-xs md:grid-cols-4">
          <div className="rounded border border-slate-400/30 p-2">
            <p className="opacity-70">Armed</p>
            <p className={`font-semibold ${armed ? "text-emerald-400" : "text-amber-300"}`}>{armed ? "Yes" : "No"}</p>
          </div>
          <div className="rounded border border-slate-400/30 p-2">
            <p className="opacity-70">Camera</p>
            <p className="font-semibold">{selectedCamera?.name ?? "Pending"}</p>
          </div>
          <div className="rounded border border-slate-400/30 p-2">
            <p className="opacity-70">Detector</p>
            <p className="font-semibold">{selectedCamera?.detection_enabled ? "On" : "Off"}</p>
          </div>
          <div className="rounded border border-slate-400/30 p-2">
            <p className="opacity-70">Status</p>
            <p className={`font-semibold ${status?.online ? "text-emerald-400" : "text-rose-400"}`}>
              {status?.online ? `Online ${status?.fps ? `${status.fps.toFixed(1)} fps` : ""}` : "Offline"}
            </p>
          </div>
        </div>

        {note ? <p className="mt-3 rounded border border-sky-500/40 bg-sky-500/10 p-2 text-xs text-sky-300">{note}</p> : null}
        {selectedCamera?.detection_enabled && !armed ? (
          <p className="mt-3 rounded border border-amber-500/40 bg-amber-500/10 p-2 text-xs text-amber-300">
            Detector is enabled, but the system is disarmed. Arm the system to restore object boxing and recognition logs.
          </p>
        ) : null}
        {error ? <p className="mt-3 rounded border border-rose-500/40 bg-rose-500/10 p-2 text-xs text-rose-300">{error}</p> : null}
      </article>

      <article className="card rounded-xl p-3 shadow-panel">
        <h3 className="mb-2 text-sm font-semibold">Live Overlay</h3>
        {cameraId ? (
          <img
            src={`/api/cameras/${cameraId}/stream.mjpeg`}
            alt="Test monitor live overlay"
            className="h-[480px] w-full rounded-lg bg-black object-contain"
          />
        ) : (
          <div className="rounded-lg border border-dashed border-slate-400/40 p-8 text-center text-sm opacity-80">
            Waiting for default webcam setup...
          </div>
        )}
      </article>

      <article className="card rounded-xl p-3 shadow-panel">
        <h3 className="mb-2 text-sm font-semibold">Recent Detection Logs</h3>
        {events.length === 0 ? (
          <p className="text-xs opacity-70">No events yet. Move in frame to trigger detection boxes and event entries.</p>
        ) : (
          <div className="space-y-2">
            {events.slice(0, 12).map((event) => (
              <div key={event.id} className="flex flex-wrap items-center justify-between gap-2 rounded border border-slate-400/30 p-2 text-xs">
                <div>
                  <p className="font-semibold uppercase">{event.label}</p>
                  <p className="opacity-70">
                    {formatTimestamp(event.created_at)} ({relativeTime(event.created_at)})
                  </p>
                </div>
                <div className="text-right">
                  <p>conf: {event.confidence.toFixed(2)}</p>
                  <p>track: #{event.track_id ?? "-"}</p>
                </div>
              </div>
            ))}
          </div>
        )}
      </article>
    </section>
  );
}
