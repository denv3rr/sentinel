import type { CameraConfig, OperatingMode } from "../lib/types";
import VideoTile from "../components/VideoTile";

interface Props {
  cameras: CameraConfig[];
  statuses: Record<string, { online?: boolean; fps?: number; last_error?: string }>;
  operatingMode: OperatingMode;
}

export default function Dashboard({ cameras, statuses, operatingMode }: Props) {
  const online = cameras.filter((c) => statuses[c.id]?.online).length;

  return (
    <section className="space-y-4">
      <article className="card rounded-xl p-4 shadow-panel">
        <h2 className="text-lg font-semibold">Live Dashboard</h2>
        <div className="mt-2 flex flex-wrap items-center gap-2 text-sm">
          <span className="opacity-80">{online}/{cameras.length} cameras online</span>
          <span className="rounded border border-teal-500/40 bg-teal-500/10 px-2 py-0.5 text-[11px] uppercase tracking-wide text-teal-300">
            Mode: {operatingMode}
          </span>
        </div>
      </article>

      {cameras.length === 0 ? (
        <article className="card rounded-xl p-6 text-sm shadow-panel">
          No cameras added yet. Go to Cameras and add a webcam or RTSP source.
        </article>
      ) : (
        <div className="grid grid-cols-1 gap-3 lg:grid-cols-2 2xl:grid-cols-3">
          {cameras.map((camera) => (
            <VideoTile key={camera.id} camera={camera} status={statuses[camera.id]} />
          ))}
        </div>
      )}
    </section>
  );
}
