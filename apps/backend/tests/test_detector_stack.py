from __future__ import annotations

import numpy as np

from sentinel.vision.detect_base import Detection
from sentinel.vision.detector_stack import BaseDetector, DetectorSettings, HybridDetector


class _SequenceDetector(BaseDetector):
    def __init__(self, settings: DetectorSettings, sequence: list[list[Detection]]) -> None:
        super().__init__(settings)
        self._sequence = sequence
        self._index = 0

    def detect_raw(self, frame: np.ndarray) -> list[Detection]:
        _ = frame
        if self._index >= len(self._sequence):
            return []
        output = self._sequence[self._index]
        self._index += 1
        return output


def _frame() -> np.ndarray:
    return np.zeros((240, 320, 3), dtype=np.uint8)


def test_hybrid_detector_merges_motion_with_yolo_when_non_overlapping() -> None:
    settings = DetectorSettings.from_profile(
        model_name="unused",
        confidence=0.1,
        profile_name="fast_motion",
        smoothing_alpha=0.0,
        hysteresis_frames=0,
    )
    yolo = _SequenceDetector(
        settings,
        sequence=[[Detection(bbox=(20, 20, 100, 180), confidence=0.91, label="person", raw_label="person")]],
    )
    motion = _SequenceDetector(
        settings,
        sequence=[[Detection(bbox=(180, 60, 260, 200), confidence=0.55, label="motion", raw_label="motion")]],
    )
    hybrid = HybridDetector(settings=settings, yolo_detector=yolo, motion_detector=motion)

    detections = hybrid.detect(_frame())

    labels = {item.label for item in detections}
    assert "person" in labels
    assert "motion" in labels
    assert len(detections) == 2


def test_hybrid_detector_nests_overlapping_subboxes_under_parent() -> None:
    settings = DetectorSettings.from_profile(
        model_name="unused",
        confidence=0.1,
        profile_name="fast_motion",
        smoothing_alpha=0.0,
        hysteresis_frames=0,
    )
    yolo = _SequenceDetector(
        settings,
        sequence=[[Detection(bbox=(30, 30, 200, 220), confidence=0.93, label="person", raw_label="person")]],
    )
    motion = _SequenceDetector(
        settings,
        sequence=[[Detection(bbox=(60, 120, 120, 170), confidence=0.61, label="limb", raw_label="limb")]],
    )
    hybrid = HybridDetector(settings=settings, yolo_detector=yolo, motion_detector=motion)

    detections = hybrid.detect(_frame())
    assert len(detections) == 1
    assert detections[0].label == "person"
    assert len(detections[0].children) == 1
    assert detections[0].children[0].label == "limb"


def test_hybrid_detector_attach_mode_off_keeps_subbox_as_peer() -> None:
    settings = DetectorSettings.from_profile(
        model_name="unused",
        confidence=0.1,
        profile_name="fast_motion",
        smoothing_alpha=0.0,
        hysteresis_frames=0,
        child_attach_mode="off",
    )
    yolo = _SequenceDetector(
        settings,
        sequence=[[Detection(bbox=(30, 30, 200, 220), confidence=0.93, label="person", raw_label="person")]],
    )
    motion = _SequenceDetector(
        settings,
        sequence=[[Detection(bbox=(60, 120, 120, 170), confidence=0.61, label="limb", raw_label="limb")]],
    )
    hybrid = HybridDetector(settings=settings, yolo_detector=yolo, motion_detector=motion)
    detections = hybrid.detect(_frame())
    assert len(detections) == 2
    assert sum(1 for item in detections if item.label == "person") == 1
    assert sum(1 for item in detections if item.label == "limb") == 1


def test_hybrid_continuity_uses_hysteresis_for_quick_miss() -> None:
    settings = DetectorSettings.from_profile(
        model_name="unused",
        confidence=0.1,
        profile_name="fast_motion",
        smoothing_alpha=0.45,
        hysteresis_frames=2,
    )
    yolo = _SequenceDetector(
        settings,
        sequence=[
            [Detection(bbox=(80, 80, 140, 170), confidence=0.88, label="person", raw_label="person")],
            [],
            [],
            [],
        ],
    )
    motion = _SequenceDetector(settings, sequence=[[], [], [], []])
    hybrid = HybridDetector(settings=settings, yolo_detector=yolo, motion_detector=motion)

    first = hybrid.detect(_frame())
    second = hybrid.detect(_frame())
    third = hybrid.detect(_frame())
    fourth = hybrid.detect(_frame())

    assert first and first[0].label == "person"
    assert second and second[0].label == "person"
    assert third and third[0].label == "person"
    assert fourth == []


def test_fast_motion_regression_detects_quick_limb_movement_from_synthetic_frames() -> None:
    settings = DetectorSettings.from_profile(
        model_name="unused",
        confidence=0.1,
        profile_name="fast_motion",
    )
    yolo = _SequenceDetector(settings, sequence=[[], [], []])
    # Use the real motion assist detector for this regression path.
    hybrid = HybridDetector(settings=settings, yolo_detector=yolo, motion_detector=None)

    frame_a = np.zeros((240, 320, 3), dtype=np.uint8)
    frame_b = frame_a.copy()
    frame_c = frame_a.copy()
    frame_b[90:120, 40:55] = 255
    frame_c[92:124, 170:188] = 255

    assert hybrid.detect(frame_a) == []
    detections_b = hybrid.detect(frame_b)
    detections_c = hybrid.detect(frame_c)

    assert detections_b
    assert detections_c
    assert any(item.raw_label in {"motion", "limb"} for item in detections_c)
