from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class DetectionChild:
    bbox: tuple[int, int, int, int]
    confidence: float
    label: str
    raw_label: str | None = None
    child_id: str | None = None


@dataclass
class Detection:
    bbox: tuple[int, int, int, int]
    confidence: float
    label: str
    raw_label: str | None = None
    children: list[DetectionChild] = field(default_factory=list)
    appearance_signature: tuple[float, ...] | None = None


class Detector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[Detection]:
        raise NotImplementedError
