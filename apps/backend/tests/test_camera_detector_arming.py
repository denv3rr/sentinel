from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from sentinel.api import routes_cameras
from sentinel.main import SentinelState
from sentinel.pipeline.runtime import PipelineRuntime


def _build_state(tmp_path, monkeypatch) -> SentinelState:
    bootstrap_path = tmp_path / "bootstrap.json"
    data_dir = tmp_path / "data"
    monkeypatch.setattr("sentinel.config.migrate.bootstrap_config_path", lambda: bootstrap_path)
    monkeypatch.setattr("sentinel.main.detect_default_webcam_index", lambda max_index=5: None)
    monkeypatch.setattr(PipelineRuntime, "sync_cameras", lambda self, cameras: None)
    return SentinelState.create(data_dir=str(data_dir), log_level="warning")


def _camera_payload(detection_enabled: bool) -> dict[str, object]:
    return {
        "name": "Front Cam",
        "kind": "webcam",
        "source": "0",
        "enabled": True,
        "detection_enabled": detection_enabled,
        "recording_mode": "event_only",
        "labels": ["person", "animal", "vehicle", "unknown"],
        "min_confidence": {"person": 0.35, "animal": 0.3, "vehicle": 0.35, "unknown": 0.45},
        "cooldown_seconds": 8,
        "motion_threshold": 0.012,
        "zones": [],
    }


def test_create_camera_with_detector_enabled_auto_arms(tmp_path, monkeypatch) -> None:
    state = _build_state(tmp_path, monkeypatch)
    app = FastAPI()
    app.state.sentinel = state
    app.include_router(routes_cameras.router, prefix="/api")

    client = TestClient(app)
    try:
        assert state.settings_store.settings.armed is False
        response = client.post("/api/cameras", json=_camera_payload(detection_enabled=True))
        assert response.status_code == 200
        assert state.settings_store.settings.armed is True
        assert state.runtime.global_armed is True
    finally:
        state.shutdown()


def test_update_camera_detector_on_auto_arms(tmp_path, monkeypatch) -> None:
    state = _build_state(tmp_path, monkeypatch)
    app = FastAPI()
    app.state.sentinel = state
    app.include_router(routes_cameras.router, prefix="/api")

    client = TestClient(app)
    try:
        created = client.post("/api/cameras", json=_camera_payload(detection_enabled=False))
        assert created.status_code == 200
        camera = created.json()["camera"]

        state.settings_store.update(armed=False)
        state.runtime.global_armed = False
        assert state.settings_store.settings.armed is False

        camera["detection_enabled"] = True
        updated = client.put(f"/api/cameras/{camera['id']}", json=camera)
        assert updated.status_code == 200
        assert state.settings_store.settings.armed is True
        assert state.runtime.global_armed is True
    finally:
        state.shutdown()
