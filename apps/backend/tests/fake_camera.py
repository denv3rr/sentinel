from __future__ import annotations

import cv2
import numpy as np

from sentinel.camera.base import CameraSource, FramePacket
from sentinel.util.time import monotonic_ns, now_local_iso, now_utc_iso


class FakeCameraSource(CameraSource):
    """Deterministic test camera source from generated frames or a video file."""

    def __init__(self, frames: list[np.ndarray] | None = None, video_path: str | None = None) -> None:
        self.frames = frames or []
        self.video_path = video_path
        self._capture: cv2.VideoCapture | None = None
        self._idx = 0
        self._connected = False

    def connect(self) -> bool:
        if self.video_path:
            self._capture = cv2.VideoCapture(self.video_path)
            self._connected = bool(self._capture.isOpened())
            return self._connected
        self._connected = bool(self.frames)
        return self._connected

    def read_frame(self) -> FramePacket | None:
        if not self._connected:
            return None

        if self._capture is not None:
            ok, frame = self._capture.read()
            if not ok or frame is None:
                return None
            return FramePacket(frame=frame, wall_time_iso=now_utc_iso(), local_time_iso=now_local_iso(), monotonic_ns=monotonic_ns())

        if self._idx >= len(self.frames):
            return None

        frame = self.frames[self._idx]
        self._idx += 1
        return FramePacket(frame=frame, wall_time_iso=now_utc_iso(), local_time_iso=now_local_iso(), monotonic_ns=monotonic_ns())

    def health(self) -> dict[str, object]:
        return {
            "connected": self._connected,
            "index": self._idx,
            "remaining": max(0, len(self.frames) - self._idx),
        }

    def reconnect(self) -> bool:
        self.close()
        self._idx = 0
        return self.connect()

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
        self._capture = None
        self._connected = False