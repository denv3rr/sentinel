from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CameraRecord:
    id: str
    name: str
    kind: str
    source: str
    enabled: bool
    settings: dict[str, Any]


@dataclass
class EventRecord:
    id: str
    created_at: str
    local_time: str
    camera_id: str
    event_type: str
    label: str
    confidence: float
    track_id: int | None
    zone: str | None
    motion: float | None
    thumbnail_path: str | None
    clip_path: str | None
    reviewed: bool
    exported: bool
    metadata: dict[str, Any]