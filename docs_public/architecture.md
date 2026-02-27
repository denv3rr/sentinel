# Architecture (Public)

## Overview

Sentinel is a local-first, cross-platform security-recognition system.
It runs as a single runtime instance from one `sentinel` command.

- In-process runtime: Python 3.11+, FastAPI, OpenCV, Ultralytics YOLO (default detector).
- Frontend: React + Vite + TypeScript + Tailwind.
- Storage: SQLite index + filesystem media in user-selected data directory.
- No subprocess workers, brokers, or external task services.

## Runtime Flow

1. Camera workers capture frames from webcam/RTSP/ONVIF-RTSP.
2. Detector classifies person/animal/vehicle/unknown.
3. Tracker assigns stable IDs.
4. Event rules apply motion + detection thresholds + cooldown + zone logic.
5. Recorder writes thumbnails/clips.
6. SQLite indexes metadata for fast filtering/export.
7. UI reads API + MJPEG streams from localhost.

## Data Layout

```
data/
  db/sentinel.db
  media/<camera_id>/<YYYY>/<MM>/<DD>/...
  exports/...
  logs/...
  config/settings.json
```

## Security Model

- Local bind by default (`127.0.0.1`).
- LAN exposure is explicit opt-in.
- RTSP credentials are redacted in logs/API.
- Secret storage uses OS keychain when available, encrypted fallback otherwise.

## Expandability

- Detector interface supports model swap (bring-your-own-model).
- Tracker interface allows alternative algorithms.
- ONVIF module is best-effort and structured for deeper profile support later.

## Web App Standards Coverage

- Responsive layouts across desktop and mobile breakpoints.
- Accessibility basics: semantic landmarks, labels, keyboard-friendly controls.
- Virtualized event table rendering for large logs.
- Explicit loading, error, and empty states on major pages.
- Debounced event search and safe state updates.
- Input validation and user notifications for settings/camera actions.
- Light/dark theming with consistent typography and visual tokens.
- Versioned settings and migration-safe startup defaults.
- Telemetry disabled by default.
