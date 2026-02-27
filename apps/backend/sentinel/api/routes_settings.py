from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

router = APIRouter(prefix="/settings", tags=["settings"])


class DataDirPayload(BaseModel):
    path: str


class ExportDirPayload(BaseModel):
    path: str | None = None


class RetentionPayload(BaseModel):
    days: int = Field(ge=1, default=30)
    max_gb: float | None = Field(default=50.0)


class ThresholdPayload(BaseModel):
    thresholds: dict[str, float]


class LanPayload(BaseModel):
    allow_lan: bool


class ArmPayload(BaseModel):
    armed: bool


class RuntimeExitResponse(BaseModel):
    ok: bool
    message: str


def _pick_native_directory() -> str | None:
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askdirectory(title="Select Sentinel data directory")
        root.destroy()
        return selected or None
    except Exception:
        return None


@router.get("")
def get_settings(request: Request) -> dict[str, object]:
    state = request.app.state.sentinel
    settings = state.settings_store.settings
    cameras = state.repo.list_cameras()
    exports_root = Path(settings.export_dir).resolve() if settings.export_dir else Path(settings.data_dir) / "exports"
    return {
        "settings": settings.model_dump(mode="json"),
        "cameras": cameras,
        "first_run": not settings.onboarding_completed,
        "data_tree": {
            "db": str(Path(settings.data_dir) / "db" / "sentinel.db"),
            "media": str(Path(settings.data_dir) / "media"),
            "exports": str(exports_root),
            "logs": str(Path(settings.data_dir) / "logs"),
        },
    }


@router.post("/pick-data-dir")
def pick_data_dir() -> dict[str, object]:
    picked = _pick_native_directory()
    if not picked:
        raise HTTPException(status_code=400, detail="Native folder picker unavailable on this host/session")
    return {"path": picked}


@router.post("/data-dir")
def set_data_dir(payload: DataDirPayload, request: Request) -> dict[str, object]:
    state = request.app.state.sentinel
    if not payload.path.strip():
        raise HTTPException(status_code=400, detail="Path cannot be empty")

    state.change_data_dir(payload.path)
    state.repo.append_settings_history("data_dir", {"path": payload.path})
    return {"ok": True, "settings": state.settings_store.settings.model_dump(mode="json")}


@router.post("/export-dir")
def set_export_dir(payload: ExportDirPayload, request: Request) -> dict[str, object]:
    state = request.app.state.sentinel
    raw = payload.path.strip() if isinstance(payload.path, str) else ""
    target: str | None = None
    if raw:
        resolved = Path(raw).expanduser().resolve()
        resolved.mkdir(parents=True, exist_ok=True)
        target = str(resolved)

    state.settings_store.update(export_dir=target)
    state.repo.append_settings_history("export_dir", {"path": target})
    return {"ok": True, "export_dir": target}


@router.post("/retention")
def set_retention(payload: RetentionPayload, request: Request) -> dict[str, object]:
    state = request.app.state.sentinel
    current = state.settings_store.settings.retention.model_dump()
    current.update(payload.model_dump())
    state.settings_store.update(retention=current)
    state.repo.append_settings_history("retention", current)
    summary = state.retention_service.enforce(days=current["days"], max_gb=current.get("max_gb"))
    return {"ok": True, "retention": current, "summary": summary}


@router.post("/thresholds")
def set_thresholds(payload: ThresholdPayload, request: Request) -> dict[str, object]:
    state = request.app.state.sentinel
    cleaned = {k: float(v) for k, v in payload.thresholds.items()}
    state.settings_store.update(label_thresholds=cleaned)
    state.repo.append_settings_history("label_thresholds", cleaned)
    state.runtime.global_thresholds = cleaned
    state.refresh_workers()
    return {"ok": True, "thresholds": cleaned}


@router.post("/lan")
def set_lan(payload: LanPayload, request: Request) -> dict[str, object]:
    state = request.app.state.sentinel
    state.settings_store.update(allow_lan=payload.allow_lan)
    state.repo.append_settings_history("allow_lan", {"allow_lan": payload.allow_lan})
    return {
        "ok": True,
        "allow_lan": payload.allow_lan,
        "note": "Runtime bind host is controlled by CLI flags; restart with --bind 0.0.0.0 to expose on LAN.",
    }


@router.post("/arm")
def set_arm(payload: ArmPayload, request: Request) -> dict[str, object]:
    state = request.app.state.sentinel
    state.settings_store.update(armed=payload.armed)
    state.repo.append_settings_history("armed", {"armed": payload.armed})
    state.runtime.global_armed = payload.armed
    return {"ok": True, "armed": payload.armed}


@router.post("/exit", response_model=RuntimeExitResponse)
def request_runtime_exit(request: Request) -> RuntimeExitResponse:
    state = request.app.state.sentinel
    state.begin_shutdown()

    request_exit = getattr(request.app.state, "request_exit", None)
    if callable(request_exit):
        request_exit()
        return RuntimeExitResponse(ok=True, message="Sentinel runtime shutdown requested.")

    raise HTTPException(status_code=503, detail="Runtime exit is unavailable in this launch mode.")
