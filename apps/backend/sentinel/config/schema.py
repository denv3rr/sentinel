from __future__ import annotations

from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

from .defaults import (
    APP_VERSION,
    DEFAULT_BIND,
    DEFAULT_EVENT_COOLDOWN_SECONDS,
    DEFAULT_LABEL_THRESHOLDS,
    DEFAULT_MOTION_THRESHOLD,
    DEFAULT_PORT,
    DEFAULT_RETENTION_DAYS,
    DEFAULT_RETENTION_MAX_GB,
)


class ZoneConfig(BaseModel):
    id: str = Field(default_factory=lambda: f"zone-{uuid4().hex[:8]}")
    name: str = "Zone"
    mode: Literal["include", "ignore", "line"] = "include"
    points: list[tuple[float, float]] = Field(default_factory=list)


class CameraConfig(BaseModel):
    id: str = Field(default_factory=lambda: f"cam-{uuid4().hex[:8]}")
    name: str = "Camera"
    kind: Literal["webcam", "rtsp", "onvif"] = "webcam"
    source: str = "0"
    enabled: bool = True
    detection_enabled: bool = False
    recording_mode: Literal["event_only", "full", "live"] = "event_only"
    labels: list[str] = Field(default_factory=lambda: ["person", "animal", "vehicle", "unknown"])
    min_confidence: dict[str, float] = Field(default_factory=lambda: dict(DEFAULT_LABEL_THRESHOLDS))
    cooldown_seconds: int = DEFAULT_EVENT_COOLDOWN_SECONDS
    motion_threshold: float = DEFAULT_MOTION_THRESHOLD
    inference_max_side: int = 960
    detector_profile: Literal["balanced", "fast_motion", "small_object", "capture_card_reforger"] = "balanced"
    detector_profile_auto: bool = True
    child_attach_mode: Literal["off", "strict", "balanced", "aggressive"] = "balanced"
    detector_confidence: float = 0.25
    detector_image_size: int = 960
    detector_max_boxes: int = 80
    detector_motion_threshold: int = 28
    detector_min_motion_area: int = 900
    detector_smoothing_alpha: float = 0.42
    detector_hysteresis_frames: int = 2
    tracker_profile: Literal["balanced", "fast_motion", "small_object", "capture_card_reforger", "memory_strong"] = "balanced"
    tracker_profile_auto: bool = True
    tracker_iou_threshold: float = 0.35
    tracker_max_age: int = 18
    tracker_memory_age: int = 45
    tracker_reid_distance_scale: float = 2.4
    tracker_min_size_similarity: float = 0.2
    tracker_use_appearance: bool = True
    tracker_appearance_weight: float = 0.32
    tracker_min_appearance_similarity: float = 0.14
    tracker_appearance_ema_alpha: float = 0.55
    overlay_show_children: bool = True
    overlay_parent_thickness: int = 2
    overlay_child_thickness: int = 1
    overlay_child_shade_delta: float = 0.30
    hierarchy_debug_export: bool = False
    range_mode: Literal["relative_depth", "relative_depth_first_observation", "calibrated_metric"] = "relative_depth"
    range_calibration: dict[str, float] | None = None
    range_calibration_preset: str | None = None
    range_calibration_presets: dict[str, dict[str, float]] = Field(default_factory=dict)
    zones: list[ZoneConfig] = Field(default_factory=list)
    secret_ref: dict[str, str] | None = None

    @field_validator("inference_max_side")
    @classmethod
    def clamp_inference_max_side(cls, value: int) -> int:
        return min(4096, max(0, value))

    @field_validator("detector_image_size")
    @classmethod
    def clamp_detector_image_size(cls, value: int) -> int:
        return min(4096, max(64, value))

    @field_validator("detector_max_boxes")
    @classmethod
    def clamp_detector_max_boxes(cls, value: int) -> int:
        return min(512, max(1, value))

    @field_validator("detector_motion_threshold")
    @classmethod
    def clamp_detector_motion_threshold(cls, value: int) -> int:
        return min(255, max(1, value))

    @field_validator("detector_min_motion_area")
    @classmethod
    def clamp_detector_min_motion_area(cls, value: int) -> int:
        return min(200_000, max(1, value))

    @field_validator("detector_smoothing_alpha")
    @classmethod
    def clamp_detector_smoothing_alpha(cls, value: float) -> float:
        return min(1.0, max(0.0, value))

    @field_validator("detector_hysteresis_frames")
    @classmethod
    def clamp_detector_hysteresis_frames(cls, value: int) -> int:
        return min(60, max(0, value))

    @field_validator("tracker_iou_threshold")
    @classmethod
    def clamp_tracker_iou_threshold(cls, value: float) -> float:
        return min(0.95, max(0.01, value))

    @field_validator("tracker_max_age")
    @classmethod
    def clamp_tracker_max_age(cls, value: int) -> int:
        return min(300, max(1, value))

    @field_validator("tracker_memory_age")
    @classmethod
    def clamp_tracker_memory_age(cls, value: int) -> int:
        return min(900, max(1, value))

    @field_validator("tracker_reid_distance_scale")
    @classmethod
    def clamp_tracker_reid_distance_scale(cls, value: float) -> float:
        return min(8.0, max(0.5, value))

    @field_validator("tracker_min_size_similarity")
    @classmethod
    def clamp_tracker_min_size_similarity(cls, value: float) -> float:
        return min(1.0, max(0.0, value))

    @field_validator("tracker_appearance_weight")
    @classmethod
    def clamp_tracker_appearance_weight(cls, value: float) -> float:
        return min(2.0, max(0.0, value))

    @field_validator("tracker_min_appearance_similarity")
    @classmethod
    def clamp_tracker_min_appearance_similarity(cls, value: float) -> float:
        return min(1.0, max(0.0, value))

    @field_validator("tracker_appearance_ema_alpha")
    @classmethod
    def clamp_tracker_appearance_ema_alpha(cls, value: float) -> float:
        return min(1.0, max(0.0, value))

    @field_validator("overlay_parent_thickness", "overlay_child_thickness")
    @classmethod
    def clamp_overlay_thickness(cls, value: int) -> int:
        return min(8, max(1, value))

    @field_validator("overlay_child_shade_delta")
    @classmethod
    def clamp_overlay_child_shade_delta(cls, value: float) -> float:
        return min(1.0, max(-1.0, value))


class RetentionConfig(BaseModel):
    days: int = DEFAULT_RETENTION_DAYS
    max_gb: float | None = DEFAULT_RETENTION_MAX_GB

    @field_validator("days")
    @classmethod
    def positive_days(cls, value: int) -> int:
        return max(1, value)


class AppSettings(BaseModel):
    version: int = APP_VERSION
    data_dir: str
    export_dir: str | None = None
    bind: str = DEFAULT_BIND
    port: int = DEFAULT_PORT
    allow_lan: bool = False
    operating_mode: Literal["home", "away", "night"] = "home"
    armed: bool = False
    telemetry_opt_in: bool = False
    onboarding_completed: bool = False
    retention: RetentionConfig = Field(default_factory=RetentionConfig)
    label_thresholds: dict[str, float] = Field(default_factory=lambda: dict(DEFAULT_LABEL_THRESHOLDS))
    cameras: list[CameraConfig] = Field(default_factory=list)

    @field_validator("data_dir")
    @classmethod
    def data_dir_not_empty(cls, value: str) -> str:
        if not value.strip():
            msg = "data_dir cannot be empty"
            raise ValueError(msg)
        return value
