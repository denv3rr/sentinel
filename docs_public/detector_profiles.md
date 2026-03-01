# Detector Profiles

Sentinel exposes detector presets through `create_default_detector(..., profile=...)`.

This system is for generic research/simulation testbench use and excludes weapon-targeting functionality.

## Profiles

- `balanced` (default): general-purpose webcam profile with moderate smoothing and conservative motion assist.
- `fast_motion`: tuned for quick hand/limb movement and rapid gestures with stronger hysteresis and lower motion-area threshold.
- `small_object`: increased image size and max boxes for small/remote targets.
- `capture_card_reforger`: tuned for capture-card footage with profile-specific motion filtering and temporal stabilization.

## Tunable Knobs

All available via `create_default_detector`:

- `confidence`: minimum detector confidence
- `profile`: one of the presets above
- `image_size`: YOLO inference image size
- `max_boxes`: max detections per frame
- `motion_threshold`: frame-difference threshold
- `min_motion_area`: minimum contour area for motion assist
- `smoothing_alpha`: temporal EMA smoothing
- `hysteresis_frames`: carry-forward frames when a track briefly disappears
- `child_attach_mode`: `off|strict|balanced|aggressive` for sub-box nesting behavior

Tracker/object-memory knobs are configured at camera runtime (API/config), not detector construction:

- `tracker_profile`: `balanced|fast_motion|small_object|capture_card_reforger|memory_strong`
- `tracker_profile_auto`: aligns tracker defaults with detector/source context while respecting manual overrides
- `tracker_iou_threshold`: base overlap gate
- `tracker_max_age`: active missed-frame budget before archiving
- `tracker_memory_age`: inactive-memory TTL for ID revival
- `tracker_reid_distance_scale`: center-distance tolerance for re-association
- `tracker_min_size_similarity`: minimum bbox-size similarity for fallback linking
- `tracker_use_appearance`: enables lightweight color-histogram appearance cues
- `tracker_appearance_weight`: how strongly appearance affects matching score
- `tracker_min_appearance_similarity`: minimum appearance agreement for weak-spatial matches
- `tracker_appearance_ema_alpha`: smoothing factor for per-track appearance memory

## Contract Notes

- The detector still returns `Detection` dataclass objects (`bbox`, `confidence`, `label`, optional `raw_label`).
- Sub-boxes are serialized as `children` under the parent object body when overlap indicates containment.
- Track memory now uses IoU + center-distance + size-aware re-association with inactive-memory revival so IDs survive brief misses more reliably.
- Appearance-assisted re-identification is enabled by default to reduce wrong ID revival after occlusion/fast motion.
- `sentinel.vision.yolo_ultralytics.create_default_detector(model_name, confidence)` remains backward-compatible.
- If the ML model is unavailable, Sentinel follows a deterministic motion-assisted fallback path.
