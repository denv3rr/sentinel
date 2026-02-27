# Data Model

## SQLite Tables

- `cameras`
  - `id`, `name`, `kind`, `source` (sanitized), `enabled`, `settings_json`, timestamps.
- `events`
  - `id`, `created_at`, `local_time`, `monotonic_ns`, `camera_id`, `event_type`, `label`, `confidence`, `track_id`, `zone`, `motion`, `thumbnail_path`, `clip_path`, `reviewed`, `exported`, `search_text`, `metadata_json`.
- `settings_history`
  - append-only audit of mutable settings.

## Query Filters

Events API supports filtering by:
- Time range
- Camera
- Label
- Confidence threshold
- Zone
- Reviewed/exported
- Free-text search
- Pagination/sort