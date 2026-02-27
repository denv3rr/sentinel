import { useState } from "react";

interface Props {
  currentPath: string;
  onSave: (path: string) => Promise<void>;
  onPickNative: () => Promise<string>;
}

export default function StoragePicker({ currentPath, onSave, onPickNative }: Props) {
  const [path, setPath] = useState(currentPath);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  return (
    <section className="card rounded-xl p-3 shadow-panel">
      <h3 className="mb-2 text-sm font-semibold">Data Directory</h3>
      <p className="mb-3 text-xs opacity-70">
        Pick storage for DB/media/exports. Sentinel never writes runtime data to the repo directory.
      </p>
      <div className="grid grid-cols-1 gap-2 md:grid-cols-[1fr_auto_auto]">
        <input
          className="rounded border border-slate-400/40 bg-transparent px-2 py-1"
          value={path}
          onChange={(e) => setPath(e.target.value)}
          aria-label="Data directory path"
          placeholder="C:/Users/<you>/SentinelData"
        />
        <button
          className="rounded border border-slate-400/40 px-3 py-1 text-xs hover:bg-slate-500/10"
          onClick={async () => {
            setBusy(true);
            setError("");
            try {
              const selected = await onPickNative();
              if (selected) setPath(selected);
            } catch (e) {
              setError((e as Error).message || "Native picker failed");
            } finally {
              setBusy(false);
            }
          }}
          disabled={busy}
        >
          {busy ? "Picking..." : "Native Picker"}
        </button>
        <button
          className="rounded bg-teal-600 px-3 py-1 text-xs font-medium text-white hover:bg-teal-500"
          onClick={async () => {
            setBusy(true);
            setError("");
            try {
              await onSave(path);
            } catch (e) {
              setError((e as Error).message || "Failed to save");
            } finally {
              setBusy(false);
            }
          }}
          disabled={busy || !path.trim()}
        >
          Save
        </button>
      </div>
      {error ? <p className="mt-2 text-xs text-rose-400">{error}</p> : null}
    </section>
  );
}