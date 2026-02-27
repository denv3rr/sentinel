import { useEffect, useState } from "react";

import StoragePicker from "../components/StoragePicker";
import { api } from "../lib/api";
import type { SettingsResponse } from "../lib/types";

interface Props {
  settings: SettingsResponse["settings"];
  onRefresh: () => Promise<void>;
}

export default function SettingsPage({ settings, onRefresh }: Props) {
  const [retentionDays, setRetentionDays] = useState(settings.retention.days);
  const [retentionMaxGb, setRetentionMaxGb] = useState(settings.retention.max_gb ?? 0);
  const [thresholds, setThresholds] = useState(settings.label_thresholds);
  const [allowLan, setAllowLan] = useState(settings.allow_lan);
  const [armed, setArmed] = useState(settings.armed);
  const [exportDir, setExportDir] = useState(settings.export_dir ?? "");
  const [notice, setNotice] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    setRetentionDays(settings.retention.days);
    setRetentionMaxGb(settings.retention.max_gb ?? 0);
    setThresholds(settings.label_thresholds);
    setAllowLan(settings.allow_lan);
    setArmed(settings.armed);
    setExportDir(settings.export_dir ?? "");
  }, [settings]);

  return (
    <section className="space-y-4">
      <StoragePicker
        currentPath={settings.data_dir}
        onPickNative={async () => {
          const result = await api.pickDataDir();
          return (result as { path: string }).path;
        }}
        onSave={async (path) => {
          await api.setDataDir(path);
          setNotice("Data directory updated");
          await onRefresh();
        }}
      />

      <article className="card rounded-xl p-3 shadow-panel">
        <h3 className="mb-2 text-sm font-semibold">Export Directory</h3>
        <p className="mb-3 text-xs opacity-70">
          Choose where exported files are saved. Leave empty to use the default inside your data directory.
        </p>
        <div className="grid grid-cols-1 gap-2 md:grid-cols-[1fr_auto_auto_auto]">
          <input
            className="rounded border border-slate-400/40 bg-transparent px-2 py-1"
            value={exportDir}
            onChange={(e) => setExportDir(e.target.value)}
            aria-label="Export directory path"
            placeholder={settings.data_dir + "/exports"}
          />
          <button
            className="rounded border border-slate-400/40 px-3 py-1 text-xs hover:bg-slate-500/10"
            onClick={async () => {
              try {
                const result = await api.pickDataDir();
                setExportDir((result as { path: string }).path);
              } catch (e) {
                setError((e as Error).message || "Native picker failed");
              }
            }}
          >
            Native Picker
          </button>
          <button
            className="rounded border border-slate-400/40 px-3 py-1 text-xs hover:bg-slate-500/10"
            onClick={async () => {
              try {
                await api.setExportDir(null);
                setNotice("Export directory reset to default");
                await onRefresh();
              } catch (e) {
                setError((e as Error).message || "Failed to reset export directory");
              }
            }}
          >
            Use Default
          </button>
          <button
            className="rounded bg-teal-600 px-3 py-1 text-xs font-medium text-white hover:bg-teal-500"
            onClick={async () => {
              try {
                await api.setExportDir(exportDir.trim() || null);
                setNotice("Export directory updated");
                await onRefresh();
              } catch (e) {
                setError((e as Error).message || "Failed to save export directory");
              }
            }}
          >
            Save
          </button>
        </div>
      </article>

      <article className="card rounded-xl p-4 shadow-panel">
        <h3 className="mb-2 text-sm font-semibold">Retention</h3>
        <div className="grid grid-cols-1 gap-2 md:grid-cols-2">
          <label>
            <span className="mb-1 block text-xs opacity-70">Days</span>
            <input
              type="number"
              min={1}
              value={retentionDays}
              onChange={(e) => setRetentionDays(Number(e.target.value))}
              className="w-full rounded border border-slate-400/40 bg-transparent px-2 py-1"
            />
          </label>
          <label>
            <span className="mb-1 block text-xs opacity-70">Max media size (GB)</span>
            <input
              type="number"
              min={0}
              step={1}
              value={retentionMaxGb}
              onChange={(e) => setRetentionMaxGb(Number(e.target.value))}
              className="w-full rounded border border-slate-400/40 bg-transparent px-2 py-1"
            />
          </label>
        </div>
        <button
          className="mt-3 rounded bg-teal-600 px-3 py-1 text-xs font-medium text-white hover:bg-teal-500"
          onClick={async () => {
            try {
              await api.setRetention({ days: retentionDays, max_gb: retentionMaxGb });
              setNotice("Retention updated");
              await onRefresh();
            } catch (e) {
              setError((e as Error).message || "Failed to save retention");
            }
          }}
        >
          Save retention
        </button>
      </article>

      <article className="card rounded-xl p-4 shadow-panel">
        <h3 className="mb-2 text-sm font-semibold">Per-label Thresholds</h3>
        <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
          {Object.entries(thresholds).map(([key, value]) => (
            <label key={key}>
              <span className="mb-1 block text-xs uppercase opacity-70">{key}</span>
              <input
                type="number"
                min={0}
                max={1}
                step={0.01}
                value={value}
                onChange={(e) => setThresholds((prev) => ({ ...prev, [key]: Number(e.target.value) }))}
                className="w-full rounded border border-slate-400/40 bg-transparent px-2 py-1"
              />
            </label>
          ))}
        </div>
        <button
          className="mt-3 rounded border border-slate-400/40 px-3 py-1 text-xs hover:bg-slate-500/10"
          onClick={async () => {
            try {
              await api.setThresholds(thresholds);
              setNotice("Thresholds updated");
              await onRefresh();
            } catch (e) {
              setError((e as Error).message || "Failed to save thresholds");
            }
          }}
        >
          Save thresholds
        </button>
      </article>

      <article className="card rounded-xl p-4 shadow-panel">
        <h3 className="mb-2 text-sm font-semibold">Detection Arming</h3>
        <label className="flex items-center gap-2 text-sm">
          <input type="checkbox" checked={armed} onChange={(e) => setArmed(e.target.checked)} />
          Arm detection engine (event logging + detector actions)
        </label>
        <button
          className="mt-3 rounded border border-slate-400/40 px-3 py-1 text-xs hover:bg-slate-500/10"
          onClick={async () => {
            await api.setArm(armed);
            setNotice(armed ? "Detection armed" : "Detection disarmed");
            await onRefresh();
          }}
        >
          Save arm state
        </button>
      </article>

      <article className="card rounded-xl p-4 shadow-panel">
        <h3 className="mb-2 text-sm font-semibold">LAN Access</h3>
        <label className="flex items-center gap-2 text-sm">
          <input type="checkbox" checked={allowLan} onChange={(e) => setAllowLan(e.target.checked)} />
          Allow LAN access flag (requires restart with --bind 0.0.0.0)
        </label>
        <button
          className="mt-3 rounded border border-amber-500/50 px-3 py-1 text-xs text-amber-300 hover:bg-amber-500/10"
          onClick={async () => {
            await api.setLan(allowLan);
            setNotice("LAN preference saved");
            await onRefresh();
          }}
        >
          Save LAN preference
        </button>
        {allowLan ? <p className="mt-2 text-xs text-amber-300">Warning: LAN mode increases exposure risk. Keep on trusted networks.</p> : null}
      </article>

      {notice ? <p className="rounded border border-emerald-500/40 bg-emerald-500/10 p-2 text-xs text-emerald-400">{notice}</p> : null}
      {error ? <p className="rounded border border-rose-500/40 bg-rose-500/10 p-2 text-xs text-rose-400">{error}</p> : null}
    </section>
  );
}
