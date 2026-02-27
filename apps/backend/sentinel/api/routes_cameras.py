from __future__ import annotations

import asyncio
from typing import Literal
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from sentinel.camera.onvif import discover_onvif, guess_rtsp_candidates
from sentinel.camera.opencv_cam import detect_default_webcam_index, discover_webcam_indices
from sentinel.util.security import sanitize_rtsp_url, validate_camera_id, validate_rtsp_url

router = APIRouter(prefix="/cameras", tags=["cameras"])


class SafeStreamingResponse(StreamingResponse):
    async def __call__(self, scope, receive, send):  # type: ignore[override]
        try:
            await super().__call__(scope, receive, send)
        except asyncio.CancelledError:
            return


class CameraPayload(BaseModel):
    id: str | None = None
    name: str = Field(min_length=1)
    kind: Literal["webcam", "rtsp", "onvif"] = "webcam"
    source: str = "0"
    enabled: bool = True
    detection_enabled: bool = False
    recording_mode: Literal["event_only", "full", "live"] = "event_only"
    labels: list[str] = Field(default_factory=lambda: ["person", "animal", "vehicle", "unknown"])
    min_confidence: dict[str, float] = Field(default_factory=dict)
    cooldown_seconds: int = 8
    motion_threshold: float = 0.012
    inference_max_side: int = Field(default=960, ge=0, le=4096)
    zones: list[dict[str, object]] = Field(default_factory=list)


def _sanitized_camera(camera: dict[str, object]) -> dict[str, object]:
    out = dict(camera)
    out.setdefault("detection_enabled", False)
    out.setdefault("recording_mode", "event_only")
    out.setdefault("inference_max_side", 960)
    source = str(out.get("source", ""))
    if str(out.get("kind", "")) in {"rtsp", "onvif"}:
        out["source"] = sanitize_rtsp_url(source)
    return out


def _to_camera_dict(
    payload: CameraPayload,
    request: Request,
    existing_id: str | None = None,
    existing: dict[str, object] | None = None,
) -> dict[str, object]:
    state = request.app.state.sentinel

    try:
        camera_id = validate_camera_id(existing_id or payload.id or f"cam-{uuid4().hex[:8]}")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    camera = payload.model_dump()
    camera["id"] = camera_id

    if payload.kind in {"rtsp", "onvif"}:
        keep_existing_secret = bool(existing and isinstance(existing.get("secret_ref"), dict))
        try:
            source_value = validate_rtsp_url(
                payload.source,
                allow_redacted_password=keep_existing_secret,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        source_has_inline_creds = "@" in source_value and ":" in source_value.split("@", 1)[0]
        if source_has_inline_creds and "***" not in source_value:
            secret = state.secret_store.store(f"rtsp:{camera_id}", source_value)
            camera["secret_ref"] = secret.as_dict()
            camera["source"] = sanitize_rtsp_url(source_value)
        elif keep_existing_secret:
            camera["secret_ref"] = existing["secret_ref"]
            camera["source"] = sanitize_rtsp_url(source_value)
        else:
            camera["source"] = sanitize_rtsp_url(source_value)
    else:
        camera["source"] = str(payload.source)

    return camera


def _sync_after_camera_change(request: Request) -> None:
    state = request.app.state.sentinel
    cameras = state.repo.list_cameras()
    state.settings_store.update(cameras=cameras, onboarding_completed=bool(cameras))
    state.refresh_workers()


def _auto_arm_if_detector_enabled(camera: dict[str, object], request: Request) -> bool:
    if not bool(camera.get("detection_enabled", False)):
        return False

    state = request.app.state.sentinel
    if bool(state.settings_store.settings.armed):
        return False

    state.settings_store.update(armed=True)
    state.repo.append_settings_history("armed", {"armed": True, "reason": "detector_enabled"})
    state.runtime.global_armed = True
    return True


@router.get("")
def list_cameras(request: Request) -> dict[str, object]:
    state = request.app.state.sentinel
    cameras = [_sanitized_camera(c) for c in state.repo.list_cameras()]
    return {"items": cameras}


@router.post("")
def create_camera(payload: CameraPayload, request: Request) -> dict[str, object]:
    state = request.app.state.sentinel
    camera = _to_camera_dict(payload, request)
    state.repo.upsert_camera(camera)
    _auto_arm_if_detector_enabled(camera, request)
    _sync_after_camera_change(request)
    return {"ok": True, "camera": _sanitized_camera(camera)}


@router.put("/{camera_id}")
def update_camera(camera_id: str, payload: CameraPayload, request: Request) -> dict[str, object]:
    state = request.app.state.sentinel
    try:
        camera_id = validate_camera_id(camera_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    existing = state.repo.get_camera(camera_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Camera not found")

    camera = _to_camera_dict(payload, request, existing_id=camera_id, existing=existing)
    state.repo.upsert_camera(camera)
    _auto_arm_if_detector_enabled(camera, request)
    _sync_after_camera_change(request)
    return {"ok": True, "camera": _sanitized_camera(camera)}


@router.delete("/{camera_id}")
def delete_camera(camera_id: str, request: Request) -> dict[str, object]:
    state = request.app.state.sentinel
    try:
        camera_id = validate_camera_id(camera_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    state.repo.delete_camera(camera_id)
    _sync_after_camera_change(request)
    return {"ok": True, "camera_id": camera_id}


@router.post("/test")
def test_camera(payload: CameraPayload, request: Request) -> dict[str, object]:
    state = request.app.state.sentinel
    camera = _to_camera_dict(payload, request, existing_id=payload.id)
    ok, message = state.runtime.test_camera(camera)
    return {"ok": ok, "message": message}


@router.get("/discover/onvif")
def discover_onvif_cameras() -> dict[str, object]:
    endpoints = discover_onvif()
    items = []
    for endpoint in endpoints:
        items.append(
            {
                "id": endpoint.id,
                "xaddr": endpoint.xaddr,
                "host": endpoint.host,
                "rtsp_candidates": guess_rtsp_candidates(endpoint.host),
            }
        )
    return {"items": items}


@router.get("/discover/webcams")
def discover_webcams(max_index: int = 10) -> dict[str, object]:
    discovered = discover_webcam_indices(max_index=max_index)
    default_index = next((item["index"] for item in discovered if item["status"] == "online"), None)
    items = [
        {
            "index": item["index"],
            "label": f"Webcam {item['index']}",
            "status": item["status"],
            "is_default": item["index"] == default_index,
        }
        for item in discovered
    ]
    return {"items": items}


@router.get("/default-webcam")
def detect_default_webcam(max_index: int = 5) -> dict[str, object]:
    """Best-effort default webcam detection by probing low indices."""
    index = detect_default_webcam_index(max_index=max_index)
    if index is not None:
        return {"ok": True, "index": str(index)}
    raise HTTPException(
        status_code=404,
        detail="No accessible webcam detected. Plug in a camera and try again.",
    )


@router.get("/{camera_id}/stream.mjpeg")
def stream_camera(camera_id: str, request: Request) -> StreamingResponse:
    state = request.app.state.sentinel
    try:
        camera_id = validate_camera_id(camera_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    headers = {
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
        "Expires": "0",
        "X-Accel-Buffering": "no",
    }
    return SafeStreamingResponse(
        state.runtime.stream(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers=headers,
    )
