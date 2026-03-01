# Convergence Integration Contract

Convergence imports Sentinel detector logic as an external consumer.

## Stable Entry Point

Convergence-facing import path remains:

```python
from sentinel.vision.yolo_ultralytics import create_default_detector
```

Back-compat callable:

```python
detector = create_default_detector(model_name="yolov8n.pt", confidence=0.25)
```

## Output Contract

Detector outputs are compatible with Sentinel `Detection`:

- `bbox`: `(x1, y1, x2, y2)` integers
- `confidence`: `float`
- `label`: generic category string
- `raw_label`: optional source label string
- `children`: optional list of sub-boxes nested under parent object bodies

Sentinel may attach additional internal fields (for example appearance-memory hints) but the Convergence-consumed fields above remain stable.

Generic labels are intentionally non-weapon semantics (`person`, `animal`, `vehicle`, `unknown`, `motion`, `limb`, etc.).

## Compatibility Policy

- No breaking change to `create_default_detector(model_name, confidence)` without a compatibility shim.
- Graceful fallback behavior is preserved when model loading fails.
- Hybrid runtime enhancements remain internal and do not change contract shape.
- When sub-boxes are present (e.g. limb/motion inside body box), they are serialized as `children` on the parent object.
- Event APIs support filtering by `child_label` and include serialized child hierarchy in metadata.

## Scope Guardrail

Sentinel and Convergence integration here is for generic research/simulation testbench usage only and excludes weapon-targeting and weapons employment logic.
