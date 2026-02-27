from __future__ import annotations

import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from sentinel.api import routes_settings
from sentinel.main import SentinelState
from sentinel.pipeline.runtime import PipelineRuntime


def _build_state(tmp_path, monkeypatch) -> SentinelState:
    bootstrap_path = tmp_path / "bootstrap.json"
    data_dir = tmp_path / "data"
    monkeypatch.setattr("sentinel.config.migrate.bootstrap_config_path", lambda: bootstrap_path)
    monkeypatch.setattr("sentinel.main.detect_default_webcam_index", lambda max_index=5: None)
    monkeypatch.setattr(PipelineRuntime, "sync_cameras", lambda self, cameras: None)
    return SentinelState.create(data_dir=str(data_dir), log_level="warning")


def test_operating_mode_maps_to_armed_and_persists_history(tmp_path, monkeypatch) -> None:
    state = _build_state(tmp_path, monkeypatch)
    app = FastAPI()
    app.state.sentinel = state
    app.include_router(routes_settings.router, prefix="/api")
    client = TestClient(app)

    try:
        assert state.settings_store.settings.operating_mode == "home"
        assert state.settings_store.settings.armed is False

        response = client.post("/api/settings/operating-mode", json={"mode": "away"})
        assert response.status_code == 200
        assert response.json() == {"ok": True, "mode": "away", "armed": True}
        assert state.settings_store.settings.operating_mode == "away"
        assert state.settings_store.settings.armed is True
        assert state.runtime.global_armed is True

        latest = state.db.query_one("SELECT key, value_json FROM settings_history ORDER BY id DESC LIMIT 1")
        assert latest is not None
        assert latest["key"] == "operating_mode"
        assert json.loads(str(latest["value_json"])) == {"mode": "away", "armed": True}

        response = client.post("/api/settings/operating-mode", json={"mode": "home"})
        assert response.status_code == 200
        assert response.json() == {"ok": True, "mode": "home", "armed": False}
        assert state.settings_store.settings.operating_mode == "home"
        assert state.settings_store.settings.armed is False
        assert state.runtime.global_armed is False
    finally:
        state.shutdown()


def test_arm_route_still_works_with_operating_mode_enabled(tmp_path, monkeypatch) -> None:
    state = _build_state(tmp_path, monkeypatch)
    app = FastAPI()
    app.state.sentinel = state
    app.include_router(routes_settings.router, prefix="/api")
    client = TestClient(app)

    try:
        mode_response = client.post("/api/settings/operating-mode", json={"mode": "night"})
        assert mode_response.status_code == 200
        assert state.settings_store.settings.operating_mode == "night"
        assert state.settings_store.settings.armed is True

        arm_response = client.post("/api/settings/arm", json={"armed": False})
        assert arm_response.status_code == 200
        assert arm_response.json() == {"ok": True, "armed": False}
        assert state.settings_store.settings.armed is False
        assert state.runtime.global_armed is False
        assert state.settings_store.settings.operating_mode == "night"
    finally:
        state.shutdown()
