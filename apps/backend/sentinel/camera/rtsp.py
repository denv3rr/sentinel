from __future__ import annotations

from typing import Any

import cv2

from sentinel.util.security import sanitize_rtsp_url
from sentinel.util.time import monotonic_ns, now_local_iso, now_utc_iso

from .base import CameraSource, FramePacket


class RTSPCameraSource(CameraSource):
    def __init__(self, rtsp_url: str, name: str, options: dict[str, Any] | None = None) -> None:
        self.rtsp_url = rtsp_url
        self.safe_url = sanitize_rtsp_url(rtsp_url)
        self.name = name
        self.options = options or {}
        self.capture: cv2.VideoCapture | None = None
        self.connected = False
        self.failures = 0

    def connect(self) -> bool:
        if self.capture is not None:
            self.capture.release()
        self.capture = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
            self.capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 1500)
        if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
            self.capture.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 1500)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.connected = bool(self.capture.isOpened())
        self.failures = 0
        return self.connected

    def read_frame(self) -> FramePacket | None:
        if not self.capture or not self.connected:
            return None
        ok, frame = self.capture.read()
        if not ok or frame is None:
            self.failures += 1
            return None
        return FramePacket(
            frame=frame,
            wall_time_iso=now_utc_iso(),
            local_time_iso=now_local_iso(),
            monotonic_ns=monotonic_ns(),
        )

    def health(self) -> dict[str, object]:
        return {
            "connected": self.connected,
            "failures": self.failures,
            "source": self.safe_url,
        }

    def reconnect(self) -> bool:
        self.close()
        return self.connect()

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
        self.capture = None
        self.connected = False
