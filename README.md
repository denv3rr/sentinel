# Sentinel

A local-run movement detector and logger for webcams or other media inputs (capture cards, security cameras, etc). It runs on Windows, macOS, and Linux, processes video fully on-device by default, and provides a simple UI for camera management, live views, event review, and exports.

## Install

### Primary (pip)

```bash
pip install -e .
sentinel
```

## Run

```bash
sentinel --data-dir "C:/Users/<you>/SentinelData" --bind 127.0.0.1 --port 8765
```

Defaults:
- `--bind 127.0.0.1`
- `--port 8765`
- `--data-dir` defaults to a platform-specific user data location (never repo root)
- Browser opens automatically unless `--no-open` is passed
- Detection starts **disarmed** by default for safer first run

## Camera Setup

- Webcam: use camera type `webcam` and index `0`, `1`, etc.
- RTSP: add full RTSP URL; credentials are sanitized in UI/logs.
- ONVIF: use discovery in Cameras page; if discovery is unavailable, paste/test RTSP directly.

Detailed instructions: `docs_public/camera_setup.md`.

## Data Directory & Retention

Sentinel stores runtime data under your chosen data directory:

- `db/sentinel.db`
- `media/<camera_id>/<YYYY>/<MM>/<DD>/...`
- `exports/...`
- `logs/...`

Retention policies:
- Event/media age-based retention (days)
- Optional size-based trimming (GB)
- Safe deletion that removes DB rows and media references together
- Per-camera recording mode:
  - `event_only` (default): log only detection events
  - `full`: continuous recording segments + segment log entries
  - `live`: short continuous recording segments + segment log entries

## Security Posture

- Local-only bind by default (`127.0.0.1`).
- LAN access requires explicit opt-in (`--bind 0.0.0.0`) and warning banner.
- RTSP credentials are redacted in logs and API payloads.
- No hard-coded credentials.
- Secret storage uses OS keychain where available, encrypted local fallback otherwise.
- Telemetry is OFF by default (none collected unless explicitly added and opted in).

See `SECURITY.md` for details.

## Development

```bash
# Backend + frontend dev
make dev
# or explicitly:
./scripts/dev.sh
# or on Windows
./scripts/dev.ps1
```

CI runs lint/typecheck/tests/build for backend and frontend.

## Public vs Internal Docs

Public docs live in `docs_public/` and are committed.
Internal planning lives in `agents/` and is intentionally untracked.

## Known Limitations (v1)

- ONVIF is best-effort discovery (network conditions/camera vendor behavior vary).
- Browser-native directory picker may be restricted by browser sandbox; Sentinel also provides a backend-assisted native picker endpoint.
- Clip writing uses OpenCV codec availability on host OS.
