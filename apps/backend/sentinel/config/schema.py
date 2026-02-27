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
    zones: list[ZoneConfig] = Field(default_factory=list)
    secret_ref: dict[str, str] | None = None

    @field_validator("inference_max_side")
    @classmethod
    def clamp_inference_max_side(cls, value: int) -> int:
        return min(4096, max(0, value))


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
