import { useEffect, useState } from "react";

interface Props {
  filters: {
    search: string;
    camera_id: string;
    label: string;
    min_confidence: number;
    zone: string;
    start: string;
    end: string;
    sort: "asc" | "desc";
    reviewed?: boolean;
    exported?: boolean;
  };
  cameraOptions: Array<{ id: string; name: string }>;
  onChange: (next: Partial<Props["filters"]>) => void;
}

export default function FiltersBar({ filters, cameraOptions, onChange }: Props) {
  const [searchDraft, setSearchDraft] = useState(filters.search ?? "");
  const [showAdvanced, setShowAdvanced] = useState(false);

  useEffect(() => {
    if (searchDraft === (filters.search ?? "")) {
      return;
    }
    const timer = window.setTimeout(() => onChange({ search: searchDraft }), 260);
    return () => window.clearTimeout(timer);
  }, [searchDraft, filters.search, onChange]);

  useEffect(() => {
    setSearchDraft(filters.search ?? "");
  }, [filters.search]);

  return (
    <section className="card space-y-3 rounded-xl p-3 shadow-panel">
      <div className="grid grid-cols-1 gap-3 md:grid-cols-5">
        <label className="md:col-span-2">
          <span className="mb-1 block text-xs opacity-70">Search</span>
          <input
            value={searchDraft}
            onChange={(e) => setSearchDraft(e.target.value)}
            className="w-full rounded border border-slate-400/40 bg-transparent px-2 py-1"
            placeholder="person, vehicle, camera name..."
          />
        </label>

        <label>
          <span className="mb-1 block text-xs opacity-70">Camera</span>
          <select value={filters.camera_id} onChange={(e) => onChange({ camera_id: e.target.value })} className="w-full rounded border border-slate-400/40 bg-transparent px-2 py-1">
            <option value="">All</option>
            {cameraOptions.map((camera) => (
              <option key={camera.id} value={camera.id}>
                {camera.name}
              </option>
            ))}
          </select>
        </label>

        <label>
          <span className="mb-1 block text-xs opacity-70">Start</span>
          <input
            type="datetime-local"
            value={filters.start}
            onChange={(e) => onChange({ start: e.target.value })}
            className="w-full rounded border border-slate-400/40 bg-transparent px-2 py-1"
          />
        </label>

        <label>
          <span className="mb-1 block text-xs opacity-70">End</span>
          <input
            type="datetime-local"
            value={filters.end}
            onChange={(e) => onChange({ end: e.target.value })}
            className="w-full rounded border border-slate-400/40 bg-transparent px-2 py-1"
          />
        </label>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <label className="w-full max-w-[220px]">
          <span className="mb-1 block text-xs opacity-70">Sort</span>
          <select
            value={filters.sort}
            onChange={(e) => onChange({ sort: e.target.value as "asc" | "desc" })}
            className="w-full rounded border border-slate-400/40 bg-transparent px-2 py-1"
          >
            <option value="desc">Newest first</option>
            <option value="asc">Oldest first</option>
          </select>
        </label>
        <button
          className="rounded border border-slate-400/40 px-3 py-1 text-xs hover:bg-slate-500/10"
          onClick={() => setShowAdvanced((prev) => !prev)}
        >
          {showAdvanced ? "Hide advanced filters" : "Show advanced filters"}
        </button>
        <button
          className="rounded border border-slate-400/40 px-3 py-1 text-xs hover:bg-slate-500/10"
          onClick={() => {
            setSearchDraft("");
            onChange({
              search: "",
              camera_id: "",
              label: "",
              min_confidence: 0,
              zone: "",
              start: "",
              end: "",
              sort: "desc",
              reviewed: undefined,
              exported: undefined
            });
          }}
        >
          Clear filters
        </button>
      </div>

      {showAdvanced ? (
        <div className="grid grid-cols-1 gap-3 rounded border border-slate-400/30 p-3 md:grid-cols-5">
          <label>
            <span className="mb-1 block text-xs opacity-70">Label</span>
            <select value={filters.label} onChange={(e) => onChange({ label: e.target.value })} className="w-full rounded border border-slate-400/40 bg-transparent px-2 py-1">
              <option value="">All</option>
              <option value="person">Person</option>
              <option value="animal">Animal</option>
              <option value="vehicle">Vehicle</option>
              <option value="unknown">Unknown</option>
            </select>
          </label>

          <label>
            <span className="mb-1 block text-xs opacity-70">Min confidence</span>
            <input
              type="number"
              min={0}
              max={1}
              step={0.05}
              value={filters.min_confidence}
              onChange={(e) => onChange({ min_confidence: Number(e.target.value) })}
              className="w-full rounded border border-slate-400/40 bg-transparent px-2 py-1"
            />
          </label>

          <label>
            <span className="mb-1 block text-xs opacity-70">Review state</span>
            <select
              value={String(filters.reviewed ?? "")}
              onChange={(e) => {
                const value = e.target.value;
                if (value === "") onChange({ reviewed: undefined });
                else onChange({ reviewed: value === "true" });
              }}
              className="w-full rounded border border-slate-400/40 bg-transparent px-2 py-1"
            >
              <option value="">All</option>
              <option value="false">Unreviewed</option>
              <option value="true">Reviewed</option>
            </select>
          </label>

          <label>
            <span className="mb-1 block text-xs opacity-70">Exported</span>
            <select
              value={String(filters.exported ?? "")}
              onChange={(e) => {
                const value = e.target.value;
                if (value === "") onChange({ exported: undefined });
                else onChange({ exported: value === "true" });
              }}
              className="w-full rounded border border-slate-400/40 bg-transparent px-2 py-1"
            >
              <option value="">All</option>
              <option value="false">Not exported</option>
              <option value="true">Exported</option>
            </select>
          </label>

          <label>
            <span className="mb-1 block text-xs opacity-70">Zone</span>
            <input
              value={filters.zone}
              onChange={(e) => onChange({ zone: e.target.value })}
              className="w-full rounded border border-slate-400/40 bg-transparent px-2 py-1"
              placeholder="zone id"
            />
          </label>
        </div>
      ) : null}
    </section>
  );
}
