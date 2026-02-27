import { useEffect, useRef } from "react";

import type { CameraConfig } from "../lib/types";

interface Props {
  camera: CameraConfig;
  status?: { online?: boolean; fps?: number; last_error?: string };
}

export default function VideoTile({ camera, status }: Props) {
  const streamUrl = `/api/cameras/${camera.id}/stream.mjpeg`;
  const imageRef = useRef<HTMLImageElement | null>(null);

  useEffect(() => {
    return () => {
      if (imageRef.current) {
        imageRef.current.src = "";
      }
    };
  }, []);

  return (
    <article className="card rounded-xl p-3 shadow-panel">
      <header className="mb-2 flex items-center justify-between gap-2">
        <h3 className="text-sm font-semibold">{camera.name}</h3>
        <span className={`rounded px-2 py-1 text-xs ${status?.online ? "bg-emerald-500/20 text-emerald-400" : "bg-rose-500/20 text-rose-400"}`}>
          {status?.online ? `Online ${status?.fps ? `${status.fps.toFixed(1)} fps` : ""}` : "Offline"}
        </span>
      </header>
      <img ref={imageRef} src={streamUrl} alt={`Live feed for ${camera.name}`} className="h-48 w-full rounded-lg bg-black object-cover" loading="eager" />
      {status?.last_error ? <p className="mt-2 text-xs text-rose-400">{status.last_error}</p> : null}
    </article>
  );
}
