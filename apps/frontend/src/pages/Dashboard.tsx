import type { CameraConfig } from "../lib/types";
import VideoTile from "../components/VideoTile";

interface Props {
  cameras: CameraConfig[];
  statuses: Record<string, { online?: boolean; fps?: number; last_error?: string }>;
}

export default function Dashboard({ cameras, statuses }: Props) {
  const online = cameras.filter((c) => statuses[c.id]?.online).length;

  return (
    <section className="space-y-4">
      <article className="card rounded-xl p-4 shadow-panel">
        <h2 className="text-lg font-semibold">Live Dashboard</h2>
        <p className="mt-1 text-sm opacity-80">
          {online}/{cameras.length} cameras online
        </p>
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