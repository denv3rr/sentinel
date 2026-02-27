from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from sentinel.api import routes_cameras
from sentinel.main import SentinelState
from sentinel.pipeline.recorder import MediaRecorder
from sentinel.pipeline.runtime import PipelineRuntime
from sentinel.util.security import validate_camera_id


def _build_state(tmp_path, monkeypatch) -> SentinelState:
    bootstrap_path = tmp_path / "bootstrap.json"
    data_dir = tmp_path / "data"
    monkeypatch.setattr("sentinel.config.migrate.bootstrap_config_path", lambda: bootstrap_path)
    monkeypatch.setattr("sentinel.main.detect_default_webcam_index", lambda max_index=5: None)
    monkeypatch.setattr(PipelineRuntime, "sync_cameras", lambda self, cameras: None)
    return SentinelState.create(data_dir=str(data_dir), log_level="warning")


def _camera_payload(*, camera_id: str | None = None) -> dict[str, object]:
    return {
        "id": camera_id,
        "name": "Front Cam",
        "kind": "webcam",
        "source": "0",
        "enabled": True,
        "detection_enabled": False,
        "recording_mode": "event_only",
        "labels": ["person", "animal", "vehicle", "unknown"],
        "min_confidence": {"person": 0.35, "animal": 0.3, "vehicle": 0.35, "unknown": 0.45},
        "cooldown_seconds": 8,
        "motion_threshold": 0.012,
        "zones": [],
    }


def test_validate_camera_id_accepts_common_ids() -> None:
    assert validate_camera_id("cam-1") == "cam-1"
    assert validate_camera_id("cam_2") == "cam_2"
    assert validate_camera_id("cam.3") == "cam.3"


@pytest.mark.parametrize(
    "camera_id",
    ["", "../cam", "cam/1", r"cam\1", "cam 1", "cam$1", ".cam"],
)
def test_validate_camera_id_rejects_unsafe_values(camera_id: str) -> None:
    with pytest.raises(ValueError):
        validate_camera_id(camera_id)


def test_camera_routes_reject_invalid_camera_id(tmp_path, monkeypatch) -> None:
    state = _build_state(tmp_path, monkeypatch)
    app = FastAPI()
    app.state.sentinel = state
    app.include_router(routes_cameras.router, prefix="/api")
    client = TestClient(app)

    try:
        create_response = client.post("/api/cameras", json=_camera_payload(camera_id="../escape"))
        assert create_response.status_code == 400
        assert create_response.json()["detail"] == "Invalid camera id"

        update_response = client.put("/api/cameras/cam$bad", json=_camera_payload())
        assert update_response.status_code == 400
        assert update_response.json()["detail"] == "Invalid camera id"

        stream_response = client.get("/api/cameras/cam$bad/stream.mjpeg")
        assert stream_response.status_code == 400
        assert stream_response.json()["detail"] == "Invalid camera id"
    finally:
        state.shutdown()


def test_recorder_uses_camera_id_validation(tmp_path) -> None:
    recorder = MediaRecorder(tmp_path / "data")
    timestamp = "2026-01-01T00:00:00+00:00"

    valid = recorder.prepare_continuous_clip_path("cam-1", timestamp, "live")
    assert valid is not None
    path, _ = valid
    assert path.is_relative_to(tmp_path / "data")

    invalid = recorder.prepare_continuous_clip_path("../escape", timestamp, "live")
    assert invalid is None
    assert not (tmp_path / "escape").exists()
