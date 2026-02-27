import { useMemo, useRef } from "react";
import { useVirtualizer } from "@tanstack/react-virtual";

import type { EventItem } from "../lib/types";
import { formatTimestamp } from "../lib/time";

interface Props {
  events: EventItem[];
  total: number;
  loading: boolean;
  onToggleReviewed: (eventId: string, reviewed: boolean) => void;
}

export default function EventTable({ events, total, loading, onToggleReviewed }: Props) {
  const parentRef = useRef<HTMLDivElement | null>(null);

  const virtualizer = useVirtualizer({
    count: events.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 88,
    overscan: 10
  });

  const rows = useMemo(() => virtualizer.getVirtualItems(), [virtualizer]);

  return (
    <section className="card rounded-xl p-3 shadow-panel">
      <div className="mb-3 flex items-center justify-between">
        <h2 className="text-base font-semibold">Events</h2>
        <p className="text-xs opacity-70">{loading ? "Loading..." : `${events.length} loaded / ${total} total`}</p>
      </div>

      {events.length === 0 && !loading ? (
        <div className="rounded-lg border border-dashed border-slate-500/40 p-8 text-center text-sm opacity-80">
          No events match the current filters.
        </div>
      ) : (
        <div ref={parentRef} className="h-[520px] overflow-auto rounded-lg border border-slate-500/30" role="region" aria-label="Event list">
          <div style={{ height: `${virtualizer.getTotalSize()}px`, position: "relative" }}>
            {rows.map((virtualRow) => {
              const event = events[virtualRow.index];
              return (
                <div
                  key={event.id}
                  className="grid grid-cols-12 items-center gap-2 border-b border-slate-500/20 px-3 py-2 text-xs"
                  style={{
                    position: "absolute",
                    top: 0,
                    left: 0,
                    width: "100%",
                    height: `${virtualRow.size}px`,
                    transform: `translateY(${virtualRow.start}px)`
                  }}
                >
                  <div className="col-span-2 font-medium">{formatTimestamp(event.created_at)}</div>
                  <div className="col-span-1">{event.camera_id}</div>
                  <div className="col-span-1 uppercase">{event.label}</div>
                  <div className="col-span-1">{event.confidence.toFixed(2)}</div>
                  <div className="col-span-1">#{event.track_id ?? "-"}</div>
                  <div className="col-span-1">{event.zone ?? "-"}</div>
                  <div className="col-span-2 truncate">{event.search_text}</div>
                  <div className="col-span-1">
                    {event.thumbnail_url ? <img src={event.thumbnail_url} alt="Event thumbnail" className="h-12 w-20 rounded object-cover" /> : <span className="opacity-70">No thumb</span>}
                  </div>
                  <div className="col-span-1 text-right">
                    {event.clip_url ? (
                      <a className="text-teal-500 underline" href={event.clip_url} target="_blank" rel="noreferrer">
                        Clip
                      </a>
                    ) : (
                      <span className="opacity-70">-</span>
                    )}
                  </div>
                  <div className="col-span-1 text-right">
                    <button
                      className="rounded border border-slate-400/40 px-2 py-1 hover:bg-slate-500/10"
                      onClick={() => onToggleReviewed(event.id, !event.reviewed)}
                    >
                      {event.reviewed ? "Unreview" : "Review"}
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </section>
  );
}