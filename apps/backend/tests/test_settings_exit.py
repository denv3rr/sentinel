from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from sentinel.api import routes_settings


class _DummyState:
    def __init__(self) -> None:
        self.begin_shutdown_calls = 0

    def begin_shutdown(self) -> None:
        self.begin_shutdown_calls += 1


def test_settings_exit_requests_runtime_shutdown() -> None:
    app = FastAPI()
    state = _DummyState()
    exit_calls: list[int] = []
    app.state.sentinel = state
    app.state.request_exit = lambda: exit_calls.append(1)
    app.include_router(routes_settings.router, prefix="/api")

    client = TestClient(app)
    response = client.post("/api/settings/exit")

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert state.begin_shutdown_calls == 1
    assert len(exit_calls) == 1


def test_settings_exit_returns_503_without_runtime_exit_hook() -> None:
    app = FastAPI()
    state = _DummyState()
    app.state.sentinel = state
    app.include_router(routes_settings.router, prefix="/api")

    client = TestClient(app)
    response = client.post("/api/settings/exit")

    assert response.status_code == 503
    assert state.begin_shutdown_calls == 1
