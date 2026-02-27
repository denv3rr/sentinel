from __future__ import annotations

import sys
import threading
from typing import Any, Literal, TypedDict

import cv2

from sentinel.util.time import monotonic_ns, now_local_iso, now_utc_iso

from .base import CameraSource, FramePacket

WebcamStatus = Literal["online", "offline", "no_signal"]


class WebcamDiscoveryItem(TypedDict):
    index: int
    status: WebcamStatus


class OpenCVCameraSource(CameraSource):
    def __init__(self, source_index: int | str, name: str, options: dict[str, Any] | None = None) -> None:
        self.source_index = source_index
        self.name = name
        self.options = options or {}
        self.capture: cv2.VideoCapture | None = None
        self.connected = False
        self.failures = 0
        self._lock = threading.Lock()

    def connect(self) -> bool:
        with self._lock:
            if self.capture is not None:
                self.capture.release()
            self.capture = self._open_capture()
            width = self.options.get("width")
            height = self.options.get("height")
            fps = self.options.get("fps")
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if width:
                self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
            if height:
                self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
            if fps:
                self.capture.set(cv2.CAP_PROP_FPS, float(fps))
            if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
                self.capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 1500)
            if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
                self.capture.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 1500)
            self.connected = bool(self.capture.isOpened())
            self.failures = 0
            return self.connected

    def read_frame(self) -> FramePacket | None:
        with self._lock:
            capture = self.capture
            connected = self.connected
        if not capture or not connected:
            return None
        ok, frame = capture.read()
        if not ok or frame is None:
            with self._lock:
                self.failures += 1
            return None
        return FramePacket(
            frame=frame,
            wall_time_iso=now_utc_iso(),
            local_time_iso=now_local_iso(),
            monotonic_ns=monotonic_ns(),
        )

    def health(self) -> dict[str, object]:
        with self._lock:
            connected = self.connected
            failures = self.failures
        return {
            "connected": connected,
            "failures": failures,
            "source": str(self.source_index),
        }

    def reconnect(self) -> bool:
        self.close()
        return self.connect()

    def close(self) -> None:
        with self._lock:
            if self.capture is not None:
                self.capture.release()
            self.capture = None
            self.connected = False

    def _open_capture(self) -> cv2.VideoCapture:
        # On Windows, DirectShow avoids MSMF hangs when a webcam device is already busy.
        if isinstance(self.source_index, int) and sys.platform.startswith("win"):
            return cv2.VideoCapture(self.source_index, cv2.CAP_DSHOW)
        return cv2.VideoCapture(self.source_index)


def detect_default_webcam_index(max_index: int = 5) -> int | None:
    for item in discover_webcam_indices(max_index=max_index):
        if item["status"] == "online":
            return item["index"]
    return None


def classify_webcam_index(index: int) -> WebcamStatus:
    source = OpenCVCameraSource(index, name=f"Probe Webcam {index}")
    try:
        if not source.connect():
            return "offline"
        frame = source.read_frame()
        if frame is None:
            return "no_signal"
        return "online"
    finally:
        source.close()


def discover_webcam_indices(max_index: int = 10) -> list[WebcamDiscoveryItem]:
    search_max = max(0, min(max_index, 20))
    return [{"index": index, "status": classify_webcam_index(index)} for index in range(search_max + 1)]
