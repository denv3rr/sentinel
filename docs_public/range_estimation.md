# Range Estimation Modes

Sentinel provides a generic 2D range/depth estimator at `sentinel.modules.range_estimator`.

This module is for research/testbench use and does not include weapon-targeting behavior.

## Modes

1. `relative_depth` (default)
- Uses inverse bbox-height proxy.
- Outputs `depth_rel` and uncertainty.
- `range_est_m` stays `null`.

2. `relative_depth_first_observation`
- Baseline-normalized depth:
  - `depth_rel = bbox_height_first / bbox_height_current`
- Applies EMA smoothing for continuity.
- `range_est_m` stays `null`.

3. `calibrated_metric`
- Uses a pinhole approximation with optional priors:
  - focal length in pixels
  - target/reference height in meters
  - camera tilt
- Outputs `range_est_m` and `range_sigma_m` when calibrated.
- If no calibration priors are provided, metric range remains `null`.

## Uncertainty Factors

Sentinel includes uncertainty terms for:

- bbox jitter
- motion blur proxy (center displacement + area heuristic)
- perspective/tilt penalty

All estimates include provenance metadata:

- method
- calibrated flag
- priors used
- per-factor uncertainty contribution

## Camera Presets

Per-camera configuration supports:

- `range_mode`
- `range_calibration`
- `range_calibration_preset`
- `range_calibration_presets`

This allows saved calibration presets and per-stream preset selection.

## Important Limitation

For 2D-only feeds, default output is relative depth. Metric range requires calibration assumptions and should be interpreted as an estimate, not ground truth.
