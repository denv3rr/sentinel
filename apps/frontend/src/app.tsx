import { useCallback, useEffect, useMemo, useState } from "react";

import { api } from "./lib/api";
import type { CameraConfig, SettingsResponse } from "./lib/types";
import Dashboard from "./pages/Dashboard";
import EventsPage from "./pages/Events";
import CamerasPage from "./pages/Cameras";
import TestPage from "./pages/Test";
import SettingsPage from "./pages/Settings";

type Page = "monitor" | "controls";

export default function App() {
  const [page, setPage] = useState<Page>("monitor");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [exiting, setExiting] = useState(false);
  const [settingsResponse, setSettingsResponse] = useState<SettingsResponse | null>(null);
  const [cameras, setCameras] = useState<CameraConfig[]>([]);
  const [statuses, setStatuses] = useState<Record<string, { online?: boolean; fps?: number; last_error?: string }>>({});
  const [theme, setTheme] = useState<"light" | "dark">(() => (localStorage.getItem("sentinel-theme") as "light" | "dark") || "dark");
  const [wizardDataDir, setWizardDataDir] = useState("");
  const [wizardCamName, setWizardCamName] = useState("Front Camera");
  const [wizardCamSource, setWizardCamSource] = useState("0");

  const loadAll = useCallback(async () => {
    if (exiting) return;
    setLoading(true);
    setError("");
    try {
      const [settings, cameraList, health] = await Promise.all([
        api.getSettings() as Promise<SettingsResponse>,
        api.listCameras() as Promise<{ items: CameraConfig[] }>,
        api.health() as Promise<{ runtime?: Record<string, any> }>
      ]);

      setSettingsResponse(settings);
      setCameras(cameraList.items ?? settings.cameras ?? []);
      setStatuses((health.runtime ?? {}) as Record<string, { online?: boolean; fps?: number; last_error?: string }>);
    } catch (e) {
      setError((e as Error).message || "Failed to load Sentinel UI state");
    } finally {
      setLoading(false);
    }
  }, [exiting]);

  useEffect(() => {
    void loadAll();
  }, [loadAll]);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("sentinel-theme", theme);
  }, [theme]);

  useEffect(() => {
    if (exiting) return;
    const poll = window.setInterval(async () => {
      try {
        const health = (await api.health()) as { runtime?: Record<string, any> };
        setStatuses((health.runtime ?? {}) as Record<string, { online?: boolean; fps?: number; last_error?: string }>);
      } catch {
        // Keep previous status if polling fails.
      }
    }, 3000);
    return () => window.clearInterval(poll);
  }, [exiting]);

  useEffect(() => {
    if (settingsResponse?.settings.data_dir) {
      setWizardDataDir(settingsResponse.settings.data_dir);
    }
  }, [settingsResponse?.settings.data_dir]);

  const firstRun = useMemo(() => Boolean(settingsResponse?.first_run), [settingsResponse]);

  return (
    <div className="min-h-screen pb-10">
      <header className="mx-auto flex w-full max-w-[1400px] flex-wrap items-center justify-between gap-2 px-4 py-4">
        <div>
          <h1 className="text-xl font-semibold tracking-wide">Sentinel</h1>
          <p className="text-xs opacity-70">Local-first security recognition</p>
        </div>

        <nav className="flex flex-wrap gap-2" aria-label="Primary navigation">
          {(["monitor", "controls"] as Page[]).map((target) => (
            <button
              key={target}
              onClick={() => setPage(target)}
              className={`rounded px-3 py-1 text-sm ${page === target ? "bg-teal-600 text-white" : "card"}`}
            >
              {target[0].toUpperCase() + target.slice(1)}
            </button>
          ))}
        </nav>

        <div className="flex items-center gap-2">
          <button className="rounded border border-slate-400/40 px-3 py-1 text-xs" onClick={() => void loadAll()}>
            Refresh
          </button>
          <button
            className="rounded border border-slate-400/40 px-3 py-1 text-xs"
            onClick={() => setTheme((prev) => (prev === "dark" ? "light" : "dark"))}
          >
            Theme: {theme}
          </button>
          <button
            className="rounded border border-rose-500/50 px-3 py-1 text-xs text-rose-300 hover:bg-rose-500/10 disabled:cursor-not-allowed disabled:opacity-50"
            disabled={exiting}
            onClick={async () => {
              if (exiting) return;
              setExiting(true);
              setNotice("Shutting down Sentinel runtime...");
              setError("");
              try {
                // Let UI unmount active stream elements before requesting runtime shutdown.
                await new Promise((resolve) => window.setTimeout(resolve, 200));
                await api.exitRuntime();
                setNotice("Sentinel is shutting down. You can close this window.");
                window.setTimeout(() => {
                  window.close();
                }, 250);
              } catch (e) {
                setExiting(false);
                setError((e as Error).message || "Failed to request shutdown");
                setNotice("");
              }
            }}
          >
            Exit Sentinel
          </button>
        </div>
      </header>

      <main className="mx-auto w-full max-w-[1400px] space-y-4 px-4">
        {firstRun ? (
          <section className="rounded-xl border border-amber-500/40 bg-amber-500/10 p-3 text-sm text-amber-100">
            <h2 className="mb-2 font-semibold">First-Run Wizard</h2>
            <p className="mb-3 text-xs opacity-90">
              Step 1: choose data directory. Step 2: add first camera. Detection starts disarmed by default.
            </p>
            <div className="grid grid-cols-1 gap-2 lg:grid-cols-[2fr_1fr_1fr]">
              <input
                value={wizardDataDir}
                onChange={(e) => setWizardDataDir(e.target.value)}
                className="rounded border border-amber-300/40 bg-transparent px-2 py-1 text-xs"
                placeholder="Data directory path"
              />
              <button
                className="rounded border border-amber-300/50 px-2 py-1 text-xs hover:bg-amber-300/10"
                onClick={async () => {
                  await api.setDataDir(wizardDataDir);
                  setNotice("Data directory saved");
                  await loadAll();
                }}
              >
                Save data dir
              </button>
              <button
                className="rounded border border-amber-300/50 px-2 py-1 text-xs hover:bg-amber-300/10"
                onClick={async () => {
                  const picked = (await api.pickDataDir()) as { path: string };
                  setWizardDataDir(picked.path);
                }}
              >
                Native picker
              </button>
            </div>
            <div className="mt-2 grid grid-cols-1 gap-2 lg:grid-cols-[2fr_1fr_1fr]">
              <input
                value={wizardCamName}
                onChange={(e) => setWizardCamName(e.target.value)}
                className="rounded border border-amber-300/40 bg-transparent px-2 py-1 text-xs"
                placeholder="Camera name"
              />
              <input
                value={wizardCamSource}
                onChange={(e) => setWizardCamSource(e.target.value)}
                className="rounded border border-amber-300/40 bg-transparent px-2 py-1 text-xs"
                placeholder="Webcam index"
              />
              <button
                className="rounded bg-teal-600 px-2 py-1 text-xs text-white hover:bg-teal-500"
                onClick={async () => {
                  await api.createCamera({
                    name: wizardCamName || "Front Camera",
                    kind: "webcam",
                    source: wizardCamSource || "0",
                    enabled: true,
                    detection_enabled: false,
                    recording_mode: "event_only",
                    labels: ["person", "animal", "vehicle", "unknown"],
                    min_confidence: { person: 0.35, animal: 0.3, vehicle: 0.35, unknown: 0.45 },
                    cooldown_seconds: 8,
                    motion_threshold: 0.012,
                    zones: []
                  });
                  setNotice("First camera added");
                  setPage("monitor");
                  await loadAll();
                }}
              >
                Add webcam
              </button>
            </div>
          </section>
        ) : null}
        {settingsResponse?.settings.allow_lan ? (
          <section className="rounded-xl border border-rose-500/40 bg-rose-500/10 p-3 text-sm text-rose-300">
            LAN mode is enabled. Keep Sentinel behind trusted network controls and avoid public exposure.
          </section>
        ) : null}
        {settingsResponse && !settingsResponse.settings.armed ? (
          <section className="rounded-xl border border-amber-500/40 bg-amber-500/10 p-3 text-sm text-amber-200">
            Detection is currently disarmed. Live preview continues, but detection event logs are paused.
          </section>
        ) : null}

        {loading ? <section className="card rounded-xl p-4">Loading Sentinel...</section> : null}
        {error ? <section className="rounded-xl border border-rose-500/40 bg-rose-500/10 p-3 text-sm text-rose-400">{error}</section> : null}
        {notice ? <section className="rounded-xl border border-emerald-500/40 bg-emerald-500/10 p-3 text-sm text-emerald-400">{notice}</section> : null}
        {!loading && settingsResponse && !exiting ? (
          <>
            {page === "monitor" ? (
              <>
                <Dashboard cameras={cameras} statuses={statuses} />
                <TestPage
                  cameras={cameras}
                  statuses={statuses}
                  onChanged={loadAll}
                  armed={Boolean(settingsResponse.settings.armed)}
                />
              </>
            ) : null}
            {page === "controls" ? (
              <>
                <CamerasPage cameras={cameras} onChanged={loadAll} />
                <SettingsPage settings={settingsResponse.settings} onRefresh={loadAll} />
                <EventsPage cameras={cameras} />
              </>
            ) : null}
          </>
        ) : null}
        {exiting ? <section className="card rounded-xl p-4 text-sm">Stopping in-process runtime and releasing camera resources...</section> : null}
      </main>
    </div>
  );
}
