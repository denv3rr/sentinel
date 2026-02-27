# Model Licensing

Sentinel defaults to Ultralytics YOLO through the `Detector` interface.

## Important

- Model and framework licensing may vary by version and use case.
- Verify current Ultralytics and model weight licensing before commercial use.
- Sentinel includes a detector interface (`vision/detect_base.py`) so you can plug in alternatives.

## Bring Your Own Model

Implement the `Detector` interface and return normalized detections:
- `bbox`
- `confidence`
- `label` (`person`/`animal`/`vehicle`/`unknown`)
- optional `raw_label`

Then wire your detector into runtime initialization.