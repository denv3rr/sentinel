# Performance Notes

## Strategy

- One capture/processing worker per enabled camera.
- Frame drop strategy when input exceeds target processing interval.
- Efficient in-memory ring buffer for clip creation.
- Lightweight MJPEG streaming for browser preview.
- Viewer-aware stream encoding to avoid wasted JPEG work when no clients are subscribed.
- Batched event persistence to reduce SQLite commit overhead under detection bursts.

## CPU/GPU

- Runs CPU-only by default.
- Uses Ultralytics YOLO when available.
- Degrades gracefully to fallback unknown-object detector if model load fails.

## Tuning

- Per-camera motion threshold.
- Per-label confidence thresholds.
- Event cooldown seconds.
- Per-camera inference resolution cap (`inference_max_side`).
- Retention to control disk pressure.
