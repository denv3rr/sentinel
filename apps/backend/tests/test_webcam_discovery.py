from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from sentinel.api import routes_cameras
from sentinel.camera import opencv_cam


class _FakeProbeSource:
    statuses: dict[int, str] = {}
    closed: list[int] = []

    def __init__(self, source_index: int | str, name: str, options: dict[str, object] | None = None) -> None:
        _ = name
        _ = options
        self.source_index = int(source_index)

    def connect(self) -> bool:
        return self.statuses.get(self.source_index, "offline") != "offline"

    def read_frame(self) -> object | None:
        status = self.statuses.get(self.source_index, "offline")
        if status == "online":
            return object()
        return None

    def close(self) -> None:
        self.closed.append(self.source_index)


def test_discover_webcam_indices_classifies_statuses(monkeypatch) -> None:
    _FakeProbeSource.statuses = {0: "offline", 1: "no_signal", 2: "online"}
    _FakeProbeSource.closed = []
    monkeypatch.setattr(opencv_cam, "OpenCVCameraSource", _FakeProbeSource)

    items = opencv_cam.discover_webcam_indices(max_index=2)

    assert items == [
        {"index": 0, "status": "offline"},
        {"index": 1, "status": "no_signal"},
        {"index": 2, "status": "online"},
    ]
    assert _FakeProbeSource.closed == [0, 1, 2]


def test_detect_default_webcam_uses_discovery(monkeypatch) -> None:
    called: dict[str, int] = {}

    def _fake_discover(max_index: int = 10):
        called["max_index"] = max_index
        return [
            {"index": 0, "status": "offline"},
            {"index": 1, "status": "no_signal"},
            {"index": 2, "status": "online"},
        ]

    monkeypatch.setattr(opencv_cam, "discover_webcam_indices", _fake_discover)

    assert opencv_cam.detect_default_webcam_index(max_index=7) == 2
    assert called["max_index"] == 7


def test_discover_webcams_route_returns_status_and_default(monkeypatch) -> None:
    called: dict[str, int] = {}

    def _fake_discover(max_index: int = 10):
        called["max_index"] = max_index
        return [
            {"index": 0, "status": "offline"},
            {"index": 1, "status": "online"},
            {"index": 2, "status": "online"},
        ]

    monkeypatch.setattr(routes_cameras, "discover_webcam_indices", _fake_discover)

    app = FastAPI()
    app.include_router(routes_cameras.router, prefix="/api")
    client = TestClient(app)

    response = client.get("/api/cameras/discover/webcams?max_index=3")
    assert response.status_code == 200
    assert called["max_index"] == 3
    assert response.json()["items"] == [
        {"index": 0, "label": "Webcam 0", "status": "offline", "is_default": False},
        {"index": 1, "label": "Webcam 1", "status": "online", "is_default": True},
        {"index": 2, "label": "Webcam 2", "status": "online", "is_default": False},
    ]
