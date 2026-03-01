from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import sentinel.vision.detector_stack as detector_stack
from sentinel.vision.detect_base import Detection
from sentinel.vision.yolo_ultralytics import create_default_detector


def test_create_default_detector_contract_shape() -> None:
    detector = create_default_detector(model_name="yolov8n.pt", confidence=0.2, profile="fast_motion")
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    _ = detector.detect(frame)
    detections = detector.detect(frame)

    assert isinstance(detections, list)
    for item in detections:
        assert isinstance(item, Detection)
        x1, y1, x2, y2 = item.bbox
        assert isinstance(x1, int)
        assert isinstance(y1, int)
        assert isinstance(x2, int)
        assert isinstance(y2, int)
        assert isinstance(item.confidence, float)
        assert isinstance(item.label, str)
        assert item.raw_label is None or isinstance(item.raw_label, str)
        assert isinstance(item.children, list)
        for child in item.children:
            assert child.child_id is None or isinstance(child.child_id, str)


def test_detector_fallback_is_deterministic_without_model() -> None:
    detector = create_default_detector(model_name="nonexistent-weights.pt", confidence=0.2, profile="fast_motion")
    frame_a = np.zeros((180, 280, 3), dtype=np.uint8)
    frame_b = frame_a.copy()
    frame_b[50:90, 60:80] = 255
    frame_c = frame_a.copy()
    frame_c[55:95, 165:185] = 255

    assert detector.detect(frame_a) == []
    first = detector.detect(frame_b)
    second = detector.detect(frame_c)

    assert first
    assert second
    assert all(isinstance(item, Detection) for item in first + second)


def test_detector_fallback_sequence_is_repeatable() -> None:
    detector_a = create_default_detector(model_name="nonexistent-weights.pt", confidence=0.2, profile="fast_motion")
    detector_b = create_default_detector(model_name="nonexistent-weights.pt", confidence=0.2, profile="fast_motion")

    frame_a = np.zeros((180, 280, 3), dtype=np.uint8)
    frame_b = frame_a.copy()
    frame_b[40:80, 40:60] = 255
    frame_c = frame_a.copy()
    frame_c[100:140, 180:205] = 255
    sequence = [frame_a, frame_b, frame_c]

    output_a = [detector_a.detect(frame) for frame in sequence]
    output_b = [detector_b.detect(frame) for frame in sequence]
    assert output_a == output_b


def test_detector_can_resolve_model_from_manifest_alias(tmp_path, monkeypatch) -> None:
    weights = tmp_path / "weights" / "best.pt"
    weights.parent.mkdir(parents=True, exist_ok=True)
    weights.touch(exist_ok=True)

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "model_id": "demo",
                "version": "1.0.0",
                "task": "detect",
                "labels": ["person"],
                "thresholds": {"person": 0.25},
                "input_size": 640,
                "training_data_tag": "demo",
                "created_at": "2026-03-01T00:00:00+00:00",
                "weights_path": str(weights),
            }
        ),
        encoding="utf-8",
    )
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps({"aliases": {"demo-alias": str(manifest_path)}}), encoding="utf-8")

    captured: dict[str, str] = {}

    def _fake_loader(model_name: str):
        captured["model_name"] = model_name
        return None

    monkeypatch.setattr(detector_stack, "_load_yolo_model", _fake_loader)
    _ = create_default_detector(model_name="demo-alias", model_registry_path=str(registry_path))

    assert Path(captured["model_name"]).resolve() == weights.resolve()
