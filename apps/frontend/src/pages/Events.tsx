import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import EventTable from "../components/EventTable";
import FiltersBar from "../components/FiltersBar";
import { api } from "../lib/api";
import type { CameraConfig, EventItem } from "../lib/types";

interface Props {
  cameras: CameraConfig[];
}

function toDateTimeLocalValue(value: Date): string {
  const year = value.getFullYear();
  const month = String(value.getMonth() + 1).padStart(2, "0");
  const day = String(value.getDate()).padStart(2, "0");
  const hour = String(value.getHours()).padStart(2, "0");
  const minute = String(value.getMinutes()).padStart(2, "0");
  return `${year}-${month}-${day}T${hour}:${minute}`;
}

interface TimelineBucket {
  start: Date;
  end: Date;
  label: string;
  count: number;
}

export default function EventsPage({ cameras }: Props) {
  const [items, setItems] = useState<EventItem[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const activeRequestRef = useRef<AbortController | null>(null);
  const [filters, setFilters] = useState({
    search: "",
    camera_id: "",
    label: "",
    min_confidence: 0,
    zone: "",
    start: "",
    end: "",
    sort: "desc" as "asc" | "desc",
    reviewed: undefined as boolean | undefined,
    exported: undefined as boolean | undefined
  });

  const handleFiltersChange = useCallback((next: Partial<typeof filters>) => {
    setFilters((prev) => {
      const keys = Object.keys(next) as Array<keyof typeof prev>;
      const changed = keys.some((key) => prev[key] !== next[key]);
      if (!changed) {
        return prev;
      }
      return { ...prev, ...next };
    });
  }, []);

  const fetchEvents = useCallback(async () => {
    activeRequestRef.current?.abort();
    const controller = new AbortController();
    activeRequestRef.current = controller;

    setLoading(true);
    setError("");
    try {
      const data = (await api.getEvents({
        ...filters,
        start: filters.start ? new Date(filters.start).toISOString() : undefined,
        end: filters.end ? new Date(filters.end).toISOString() : undefined,
        limit: 300,
        sort: filters.sort
      }, { signal: controller.signal })) as { items: EventItem[]; total: number };
      setItems(data.items ?? []);
      setTotal(data.total ?? 0);
    } catch (e) {
      if (e instanceof DOMException && e.name === "AbortError") {
        return;
      }
      setError((e as Error).message || "Failed to load events");
    } finally {
      if (activeRequestRef.current === controller) {
        setLoading(false);
      }
    }
  }, [filters]);

  useEffect(() => {
    void fetchEvents();
    return () => {
      activeRequestRef.current?.abort();
      activeRequestRef.current = null;
    };
  }, [fetchEvents]);

  const cameraOptions = useMemo(() => cameras.map((c) => ({ id: c.id, name: c.name })), [cameras]);
  const timelineBuckets = useMemo<TimelineBucket[]>(() => {
    const byHour = new Map<number, number>();
    for (const item of items) {
      const at = new Date(item.created_at);
      at.setMinutes(0, 0, 0);
      const key = at.getTime();
      byHour.set(key, (byHour.get(key) ?? 0) + 1);
    }
    return Array.from(byHour.entries())
      .sort((a, b) => a[0] - b[0])
      .slice(-24)
      .map(([hourStart, count]) => {
        const start = new Date(hourStart);
        const end = new Date(hourStart + 60 * 60 * 1000);
        return {
          start,
          end,
          label: `${start.getHours().toString().padStart(2, "0")}:00`,
          count
        };
      });
  }, [items]);
  const maxBucketCount = useMemo(() => Math.max(1, ...timelineBuckets.map((bucket) => bucket.count)), [timelineBuckets]);

  const setQuickRange = useCallback((hours: number) => {
    const end = new Date();
    const start = new Date(end.getTime() - hours * 60 * 60 * 1000);
    handleFiltersChange({ start: toDateTimeLocalValue(start), end: toDateTimeLocalValue(end) });
  }, [handleFiltersChange]);

  return (
    <section className="space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 className="text-lg font-semibold">Events</h2>
        <button
          className="rounded bg-teal-600 px-3 py-1 text-xs font-medium text-white hover:bg-teal-500"
          onClick={async () => {
            try {
              const result = await api.exportEvents({
                format: "jsonl",
                ...filters,
                start: filters.start ? new Date(filters.start).toISOString() : undefined,
                end: filters.end ? new Date(filters.end).toISOString() : undefined
              });
              const exportResult = result as { path: string; count: number; download_url: string | null };
              if (exportResult.download_url) {
                window.open(exportResult.download_url, "_blank");
              }
              setNotice(`Exported ${exportResult.count} events to ${exportResult.path}`);
            } catch (e) {
              setError((e as Error).message || "Export failed");
            }
          }}
        >
          Export current filter
        </button>
      </div>

      <FiltersBar
        filters={filters}
        cameraOptions={cameraOptions}
        onChange={handleFiltersChange}
      />

      <section className="card space-y-3 rounded-xl p-3 shadow-panel">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div>
            <h3 className="text-sm font-semibold">Quick Time Range</h3>
            <p className="text-xs opacity-70">Use shortcuts, then fine-tune with filters above if needed.</p>
          </div>
          <div className="flex flex-wrap gap-2">
            <button className="rounded border border-slate-400/40 px-3 py-1 text-xs hover:bg-slate-500/10" onClick={() => setQuickRange(1)}>
              Last hour
            </button>
            <button className="rounded border border-slate-400/40 px-3 py-1 text-xs hover:bg-slate-500/10" onClick={() => setQuickRange(24)}>
              Last 24 hours
            </button>
            <button className="rounded border border-slate-400/40 px-3 py-1 text-xs hover:bg-slate-500/10" onClick={() => setQuickRange(24 * 7)}>
              Last 7 days
            </button>
            <button
              className="rounded border border-slate-400/40 px-3 py-1 text-xs hover:bg-slate-500/10"
              onClick={() => handleFiltersChange({ start: "", end: "" })}
            >
              Clear range
            </button>
          </div>
        </div>
        <div>
          <p className="mb-2 text-xs opacity-70">Activity Timeline (last 24 loaded hours)</p>
          {timelineBuckets.length === 0 ? (
            <p className="rounded border border-dashed border-slate-400/30 p-3 text-xs opacity-70">No events available for timeline yet.</p>
          ) : (
            <div className="flex items-end gap-1 overflow-x-auto rounded border border-slate-400/30 p-2">
              {timelineBuckets.map((bucket) => {
                const barHeight = Math.max(12, Math.round((bucket.count / maxBucketCount) * 70));
                return (
                  <button
                    key={bucket.start.toISOString()}
                    className="flex min-w-[34px] flex-col items-center rounded px-1 py-1 hover:bg-slate-500/10"
                    onClick={() => handleFiltersChange({ start: toDateTimeLocalValue(bucket.start), end: toDateTimeLocalValue(bucket.end) })}
                    title={`${bucket.label} - ${bucket.count} event${bucket.count === 1 ? "" : "s"}`}
                  >
                    <div className="w-5 rounded-sm bg-teal-500/60" style={{ height: `${barHeight}px` }} />
                    <span className="mt-1 text-[10px] opacity-70">{bucket.label}</span>
                  </button>
                );
              })}
            </div>
          )}
        </div>
      </section>

      {notice ? <p className="rounded border border-emerald-500/40 bg-emerald-500/10 p-2 text-xs text-emerald-400">{notice}</p> : null}
      {error ? <p className="rounded border border-rose-500/40 bg-rose-500/10 p-2 text-xs text-rose-400">{error}</p> : null}

      <EventTable
        events={items}
        total={total}
        loading={loading}
        onToggleReviewed={async (eventId, reviewed) => {
          await api.markReviewed(eventId, reviewed);
          void fetchEvents();
        }}
      />
    </section>
  );
}
