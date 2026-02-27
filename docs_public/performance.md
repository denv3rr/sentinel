# Performance Notes

## Strategy

- One capture/processing worker per enabled camera.
- Frame drop strategy when input exceeds target processing interval.
- Efficient in-memory ring buffer for clip creation.
- Lightweight MJPEG streaming for browser preview.

## CPU/GPU

- Runs CPU-only by default.
- Uses Ultralytics YOLO when available.
- Degrades gracefully to fallback unknown-object detector if model load fails.

## Tuning

- Per-camera motion threshold.
- Per-label confidence thresholds.
- Event cooldown seconds.
- Retention to control disk pressure.