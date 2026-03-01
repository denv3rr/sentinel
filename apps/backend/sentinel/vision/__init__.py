"""Computer vision helpers for Sentinel."""

from .detect_base import Detection, DetectionChild, Detector
from .detector_stack import (
    BaseDetector,
    DETECTOR_PROFILES,
    DetectorProfile,
    DetectorSettings,
    HybridDetector,
    MotionAssistDetector,
    UltralyticsDetector,
    create_default_detector,
)

__all__ = [
    "Detection",
    "DetectionChild",
    "Detector",
    "BaseDetector",
    "DetectorProfile",
    "DetectorSettings",
    "DETECTOR_PROFILES",
    "UltralyticsDetector",
    "MotionAssistDetector",
    "HybridDetector",
    "create_default_detector",
]
