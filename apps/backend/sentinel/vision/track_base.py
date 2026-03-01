from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from .detect_base import Detection, DetectionChild


@dataclass
class TrackedObject:
    track_id: int
    bbox: tuple[int, int, int, int]
    confidence: float
    label: str
    raw_label: str | None = None
    children: list[DetectionChild] = field(default_factory=list)


class Tracker(ABC):
    @abstractmethod
    def update(self, detections: list[Detection]) -> list[TrackedObject]:
        raise NotImplementedError
