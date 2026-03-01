from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel

from sentinel.storage.repo import EventQuery

router = APIRouter(prefix="/events", tags=["events"])


class ReviewPayload(BaseModel):
    reviewed: bool = True


@router.get("")
def list_events(
    request: Request,
    start: str | None = None,
    end: str | None = None,
    camera_id: str | None = None,
    label: str | None = None,
    min_confidence: float | None = None,
    zone: str | None = None,
    child_label: str | None = None,
    reviewed: bool | None = None,
    exported: bool | None = None,
    search: str | None = None,
    limit: int = 200,
    offset: int = 0,
    sort: str = "desc",
) -> dict[str, object]:
    state = request.app.state.sentinel
    query = EventQuery(
        start=start,
        end=end,
        camera_id=camera_id,
        label=label,
        min_confidence=min_confidence,
        zone=zone,
        child_label=child_label,
        reviewed=reviewed,
        exported=exported,
        search=search,
        limit=limit,
        offset=offset,
        sort=sort,
    )
    rows = state.repo.query_events(query)
    for row in rows:
        thumb = row.get("thumbnail_path")
        clip = row.get("clip_path")
        row["thumbnail_url"] = f"/api/media/{thumb}" if thumb else None
        row["clip_url"] = f"/api/media/{clip}" if clip else None
    total = state.repo.count_events(query)
    return {"items": rows, "total": total}


@router.post("/{event_id}/review")
def mark_reviewed(event_id: str, payload: ReviewPayload, request: Request) -> dict[str, object]:
    state = request.app.state.sentinel
    state.repo.mark_event_reviewed(event_id, reviewed=payload.reviewed)
    return {"ok": True, "event_id": event_id, "reviewed": payload.reviewed}
