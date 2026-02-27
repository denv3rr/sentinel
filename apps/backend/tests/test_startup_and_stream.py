from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from sentinel.api import routes_cameras
from sentinel.main import SentinelState
from sentinel.pipeline.runtime import PipelineRuntime


class _DummyRuntime:
    def stream(self, camera_id: str):
        _ = camera_id
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\nfake\r\n"


class _DummyState:
    runtime = _DummyRuntime()


def test_stream_endpoint_sets_no_cache_headers() -> None:
    app = FastAPI()
    app.state.sentinel = _DummyState()
    app.include_router(routes_cameras.router, prefix="/api")

    client = TestClient(app)
    with client.stream("GET", "/api/cameras/cam-1/stream.mjpeg") as response:
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store, no-cache, must-revalidate, max-age=0"
        assert response.headers["pragma"] == "no-cache"
        assert response.headers["expires"] == "0"
        assert response.headers["x-accel-buffering"] == "no"
        chunk = next(response.iter_bytes())
        assert b"--frame" in chunk


def test_state_bootstraps_default_webcam_when_no_cameras(tmp_path, monkeypatch) -> None:
    bootstrap_path = tmp_path / "bootstrap.json"
    data_dir = tmp_path / "data"

    monkeypatch.setattr("sentinel.config.migrate.bootstrap_config_path", lambda: bootstrap_path)
    monkeypatch.setattr("sentinel.main.detect_default_webcam_index", lambda max_index=5: 0)
    monkeypatch.setattr(PipelineRuntime, "sync_cameras", lambda self, cameras: None)

    state = SentinelState.create(data_dir=str(data_dir), log_level="warning")
    try:
        cameras = state.repo.list_cameras()
        assert len(cameras) == 1
        assert cameras[0]["name"] == "Default Webcam"
        assert cameras[0]["kind"] == "webcam"
        assert cameras[0]["source"] == "0"
        assert cameras[0]["enabled"] is True
        assert cameras[0]["detection_enabled"] is False
    finally:
        state.shutdown()


def test_state_keeps_empty_when_no_default_webcam(tmp_path, monkeypatch) -> None:
    bootstrap_path = tmp_path / "bootstrap.json"
    data_dir = tmp_path / "data"

    monkeypatch.setattr("sentinel.config.migrate.bootstrap_config_path", lambda: bootstrap_path)
    monkeypatch.setattr("sentinel.main.detect_default_webcam_index", lambda max_index=5: None)
    monkeypatch.setattr(PipelineRuntime, "sync_cameras", lambda self, cameras: None)

    state = SentinelState.create(data_dir=str(data_dir), log_level="warning")
    try:
        assert state.repo.list_cameras() == []
    finally:
        state.shutdown()


def test_state_shutdown_is_idempotent(tmp_path, monkeypatch) -> None:
    bootstrap_path = tmp_path / "bootstrap.json"
    data_dir = tmp_path / "data"

    monkeypatch.setattr("sentinel.config.migrate.bootstrap_config_path", lambda: bootstrap_path)
    monkeypatch.setattr("sentinel.main.detect_default_webcam_index", lambda max_index=5: None)
    monkeypatch.setattr(PipelineRuntime, "sync_cameras", lambda self, cameras: None)

    state = SentinelState.create(data_dir=str(data_dir), log_level="warning")
    stop_calls: list[int] = []
    close_calls: list[int] = []
    monkeypatch.setattr(state.runtime, "stop_all", lambda: stop_calls.append(1))
    monkeypatch.setattr(state.db, "close", lambda: close_calls.append(1))

    state.begin_shutdown()
    state.begin_shutdown()
    assert len(stop_calls) == 1

    state.shutdown()
    state.shutdown()
    assert len(stop_calls) == 1
    assert len(close_calls) == 1
