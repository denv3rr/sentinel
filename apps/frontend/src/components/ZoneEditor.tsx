import { useMemo } from "react";

interface Zone {
  id: string;
  name: string;
  mode: "include" | "ignore" | "line";
  points: Array<[number, number]>;
}

interface Props {
  value: Zone[];
  onChange: (zones: Zone[]) => void;
}

export default function ZoneEditor({ value, onChange }: Props) {
  const zones = useMemo(() => value ?? [], [value]);

  const addZone = () => {
    onChange([
      ...zones,
      {
        id: `zone-${Math.random().toString(36).slice(2, 8)}`,
        name: `Zone ${zones.length + 1}`,
        mode: "include",
        points: [
          [0.2, 0.2],
          [0.8, 0.2],
          [0.8, 0.8],
          [0.2, 0.8]
        ]
      }
    ]);
  };

  return (
    <section className="card rounded-xl p-3 shadow-panel">
      <div className="mb-2 flex items-center justify-between">
        <h3 className="text-sm font-semibold">Zones</h3>
        <button className="rounded border border-slate-400/40 px-2 py-1 text-xs" onClick={addZone}>
          Add zone
        </button>
      </div>
      {zones.length === 0 ? <p className="text-xs opacity-70">No zones configured.</p> : null}
      <div className="space-y-2">
        {zones.map((zone, index) => (
          <article key={zone.id} className="rounded border border-slate-400/30 p-2 text-xs">
            <div className="grid grid-cols-1 gap-2 md:grid-cols-3">
              <input
                className="rounded border border-slate-400/40 bg-transparent px-2 py-1"
                value={zone.name}
                onChange={(e) => {
                  const next = [...zones];
                  next[index] = { ...zone, name: e.target.value };
                  onChange(next);
                }}
              />
              <select
                className="rounded border border-slate-400/40 bg-transparent px-2 py-1"
                value={zone.mode}
                onChange={(e) => {
                  const next = [...zones];
                  next[index] = { ...zone, mode: e.target.value as Zone["mode"] };
                  onChange(next);
                }}
              >
                <option value="include">Include</option>
                <option value="ignore">Ignore</option>
                <option value="line">Line</option>
              </select>
              <button
                className="rounded border border-rose-500/50 px-2 py-1 text-rose-400"
                onClick={() => {
                  const next = zones.filter((z) => z.id !== zone.id);
                  onChange(next);
                }}
              >
                Remove
              </button>
            </div>
            <p className="mt-2 opacity-70">Points (normalized): {zone.points.map(([x, y]) => `(${x.toFixed(2)}, ${y.toFixed(2)})`).join(" ")}</p>
          </article>
        ))}
      </div>
    </section>
  );
}