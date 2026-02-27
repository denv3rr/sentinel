from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from sentinel.api import routes_cameras
from sentinel.main import SentinelState
from sentinel.pipeline.runtime import PipelineRuntime
from sentinel.util.security import validate_rtsp_url


def _build_state(tmp_path, monkeypatch) -> SentinelState:
    bootstrap_path = tmp_path / "bootstrap.json"
    data_dir = tmp_path / "data"
    monkeypatch.setattr("sentinel.config.migrate.bootstrap_config_path", lambda: bootstrap_path)
    monkeypatch.setattr("sentinel.main.detect_default_webcam_index", lambda max_index=5: None)
    monkeypatch.setattr(PipelineRuntime, "sync_cameras", lambda self, cameras: None)
    return SentinelState.create(data_dir=str(data_dir), log_level="warning")


def _rtsp_payload(source: str) -> dict[str, object]:
    return {
        "name": "RTSP Cam",
        "kind": "rtsp",
        "source": source,
        "enabled": True,
        "detection_enabled": False,
        "recording_mode": "event_only",
        "labels": ["person", "animal", "vehicle", "unknown"],
        "min_confidence": {"person": 0.35, "animal": 0.3, "vehicle": 0.35, "unknown": 0.45},
        "cooldown_seconds": 8,
        "motion_threshold": 0.012,
        "zones": [],
    }


def test_validate_rtsp_url_accepts_standard_source() -> None:
    source = "rtsp://admin:secret@192.168.1.25:554/stream1"
    assert validate_rtsp_url(source) == source


def test_validate_rtsp_url_accepts_rtsps_source() -> None:
    source = "rtsps://admin:secret@camera.local:322/stream1"
    assert validate_rtsp_url(source) == source


@pytest.mark.parametrize(
    "source",
    [
        "http://192.168.1.25/stream1",
        "rtsp:///stream1",
        "rtsp://camera.local/stream1#fragment",
        "rtsp://camera.local/stream 1",
        "rtsp://admin:***@camera.local/stream1",
    ],
)
def test_validate_rtsp_url_rejects_invalid_sources(source: str) -> None:
    with pytest.raises(ValueError):
        validate_rtsp_url(source)


def test_validate_rtsp_url_allows_redacted_password_for_existing_secret_ref() -> None:
    source = "rtsp://admin:***@camera.local/stream1"
    assert validate_rtsp_url(source, allow_redacted_password=True) == source


def test_api_rejects_invalid_rtsp_source(tmp_path, monkeypatch) -> None:
    state = _build_state(tmp_path, monkeypatch)
    app = FastAPI()
    app.state.sentinel = state
    app.include_router(routes_cameras.router, prefix="/api")
    client = TestClient(app)

    try:
        invalid_scheme = client.post("/api/cameras", json=_rtsp_payload("http://camera.local/stream1"))
        assert invalid_scheme.status_code == 400
        assert invalid_scheme.json()["detail"] == "Invalid RTSP source"

        redacted_without_secret = client.post("/api/cameras", json=_rtsp_payload("rtsp://admin:***@camera.local/stream1"))
        assert redacted_without_secret.status_code == 400
        assert redacted_without_secret.json()["detail"] == "Invalid RTSP source"
    finally:
        state.shutdown()


def test_api_allows_redacted_rtsp_update_when_secret_ref_exists(tmp_path, monkeypatch) -> None:
    state = _build_state(tmp_path, monkeypatch)
    app = FastAPI()
    app.state.sentinel = state
    app.include_router(routes_cameras.router, prefix="/api")
    client = TestClient(app)

    try:
        created = client.post("/api/cameras", json=_rtsp_payload("rtsp://admin:secret@camera.local/stream1"))
        assert created.status_code == 200
        created_camera = created.json()["camera"]
        assert created_camera["source"] == "rtsp://admin:***@camera.local/stream1"
        assert isinstance(created_camera.get("secret_ref"), dict)

        updated_payload = dict(created_camera)
        updated_payload["name"] = "Updated RTSP Cam"
        updated_payload["source"] = "rtsp://admin:***@camera.local/stream1"
        updated = client.put(f"/api/cameras/{created_camera['id']}", json=updated_payload)

        assert updated.status_code == 200
        updated_camera = updated.json()["camera"]
        assert updated_camera["name"] == "Updated RTSP Cam"
        assert updated_camera.get("secret_ref") == created_camera.get("secret_ref")
    finally:
        state.shutdown()
