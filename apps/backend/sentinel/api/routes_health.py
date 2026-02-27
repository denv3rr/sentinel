from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter(prefix="/health", tags=["health"])


@router.get("")
def get_health(request: Request) -> dict[str, object]:
    state = request.app.state.sentinel
    settings = state.settings_store.settings
    return {
        "ok": True,
        "version": "0.1.0",
        "bind": settings.bind,
        "port": settings.port,
        "allow_lan": settings.allow_lan,
        "armed": settings.armed,
        "data_dir": settings.data_dir,
        "cameras_online": sum(1 for s in state.runtime.statuses().values() if s.get("online")),
        "runtime": state.runtime.statuses(),
    }
