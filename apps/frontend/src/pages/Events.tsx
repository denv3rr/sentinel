import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import EventTable from "../components/EventTable";
import FiltersBar from "../components/FiltersBar";
import { api } from "../lib/api";
import type { CameraConfig, EventItem } from "../lib/types";

interface Props {
  cameras: CameraConfig[];
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
