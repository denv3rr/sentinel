# Scaling and Hardening Roadmap

## Purpose

This document tracks the multi-phase execution plan for camera-input scaling, detection quality, security hardening, and operational robustness.

## Implemented So Far

### Throughput and Input Handling

- Added viewer-aware stream encoding:
  - The runtime now skips annotation/JPEG encoding work when no stream subscribers are connected.
- Added inference downscaling:
  - `inference_max_side` camera setting controls the max inference dimension.
  - Bounding boxes are rescaled back to original frame coordinates.
- Improved duplicate webcam-source normalization:
  - Numeric-equivalent sources (for example `0` and `00`) are treated as the same source.

### Persistence and Runtime Robustness

- Added batched event inserts:
  - Runtime persists emitted events in batches using one `executemany` call.
- Added batch-write fallback:
  - If a batch insert fails, runtime falls back to per-event writes to avoid event loss.
- Tuned SQLite contention behavior:
  - Added `busy_timeout=5000` and `synchronous=NORMAL`.

### Security Hardening

- Added robust path containment checks:
  - Static and media routes now use canonical path checks.
  - `/api/media` is restricted to media/export directories.
- Added camera-ID validation:
  - Camera IDs are validated at API boundaries and recorder path usage.
- Added RTSP source validation:
  - Strict scheme/host/format checks with support for `rtsp` and `rtsps`.
  - Safe handling for redacted credentials when an existing secret reference is present.
- Hardened container defaults:
  - Non-root container user.
  - Explicit runtime data directory.
  - Compose hardening flags (`read_only`, `tmpfs`, `cap_drop`, `no-new-privileges`).

## Remaining Phases

### Phase A: Async Media I/O Queue

Goal:
- Remove media encode/write work from the hot camera loop.

Planned:
- Add bounded queue + worker thread(s) for thumbnail/clip writes.
- Add backpressure metrics and safe drop behavior.

### Phase B: Runtime Metrics and Capacity Harness

Goal:
- Make scaling measurable and repeatable.

Planned:
- Per-camera metrics:
  - inference latency
  - encode latency
  - event persistence latency
  - reconnect/backoff counters
- Add scripted benchmark profiles and baseline report format.

### Phase C: Auth Guardrails

Goal:
- Protect mutating endpoints when exposed beyond localhost.

Planned:
- Add local auth/session or token guard for mutating APIs.
- Preserve simple first-run UX for local-only mode.

### Phase D: Detection Quality Continuation

Goal:
- Improve precision/recall stability under noisy motion scenes.

Planned:
- Two-stage gating (motion gate + detector).
- Temporal confidence smoothing.
- Min object area / scene-change suppression controls.

### Phase E: UX Polish for Non-Technical Users

Goal:
- Keep primary workflows simple and recovery actions obvious.

Planned:
- Stronger no-signal guidance and one-click input recovery prompts.
- Maintain strict Basic vs Advanced separation for high-complexity controls.

## Validation Gates

Before each release:

- Backend:
  - full test suite passes
  - lint/ruff passes
- Frontend:
  - typecheck passes
  - lint passes (no new warnings)
  - production build passes
- Runtime:
  - no-signal fallback behavior validated
  - multi-camera burst tests show no event-loss regressions

