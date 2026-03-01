from __future__ import annotations

import sentinel.vision.detector_stack as detector_stack
from sentinel.ml.runners import run_benchmark


def test_benchmark_guard_warm_path_faster_than_cold(monkeypatch) -> None:
    # Keep benchmark deterministic and lightweight in CI by forcing no-model fallback.
    monkeypatch.setattr(detector_stack, "_load_yolo_model", lambda model_name: None)

    balanced = run_benchmark(profile="balanced", frames=40, frame_width=320, frame_height=180)
    fast = run_benchmark(profile="fast_motion", frames=40, frame_width=320, frame_height=180)

    assert balanced.cold_start_ms > balanced.warm_inference_ms
    assert fast.cold_start_ms > fast.warm_inference_ms
    assert balanced.fast_motion_fps > 0.0
    assert fast.fast_motion_fps > 0.0
    # Regression guard: fast-motion profile should not regress drastically vs balanced.
    assert fast.warm_inference_ms <= (balanced.warm_inference_ms * 1.5)

