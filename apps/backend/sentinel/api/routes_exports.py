from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request
from pydantic import BaseModel

from sentinel.storage.repo import EventQuery

router = APIRouter(prefix="/exports", tags=["exports"])


class ExportPayload(BaseModel):
    format: str = "jsonl"
    start: str | None = None
    end: str | None = None
    camera_id: str | None = None
    label: str | None = None
    min_confidence: float | None = None
    zone: str | None = None
    child_label: str | None = None
    reviewed: bool | None = None
    exported: bool | None = None
    search: str | None = None


@router.post("")
def export_events(payload: ExportPayload, request: Request) -> dict[str, object]:
    state = request.app.state.sentinel
    query = EventQuery(
        start=payload.start,
        end=payload.end,
        camera_id=payload.camera_id,
        label=payload.label,
        min_confidence=payload.min_confidence,
        zone=payload.zone,
        child_label=payload.child_label,
        reviewed=payload.reviewed,
        exported=payload.exported,
        search=payload.search,
        limit=10_000,
        offset=0,
        sort="desc",
    )
    configured_export_dir = state.settings_store.settings.export_dir
    export_dir_path = Path(configured_export_dir).resolve() if configured_export_dir else None
    result = state.export_service.export(payload.format, query, export_dir=export_dir_path)
    download_url = f"/api/media/{result['path']}" if bool(result.get("inside_data_dir")) else None
    return {
        "ok": True,
        "format": result["format"],
        "count": result["count"],
        "path": result["path"],
        "download_url": download_url,
    }
