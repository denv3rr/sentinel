from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class FramePacket:
    frame: np.ndarray
    wall_time_iso: str
    local_time_iso: str
    monotonic_ns: int


class CameraSource(ABC):
    @abstractmethod
    def connect(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def read_frame(self) -> FramePacket | None:
        raise NotImplementedError

    @abstractmethod
    def health(self) -> dict[str, object]:
        raise NotImplementedError

    @abstractmethod
    def reconnect(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError