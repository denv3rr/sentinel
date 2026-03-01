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
    detector_profile: Literal["balanced", "fast_motion", "small_object", "capture_card_reforger"] = "balanced"
    detector_profile_auto: bool = True
    child_attach_mode: Literal["off", "strict", "balanced", "aggressive"] = "balanced"
    detector_confidence: float = Field(default=0.25, ge=0.01, le=1.0)
    detector_image_size: int = Field(default=960, ge=64, le=4096)
    detector_max_boxes: int = Field(default=80, ge=1, le=512)
    detector_motion_threshold: int = Field(default=28, ge=1, le=255)
    detector_min_motion_area: int = Field(default=900, ge=1, le=200000)
    detector_smoothing_alpha: float = Field(default=0.42, ge=0.0, le=1.0)
    detector_hysteresis_frames: int = Field(default=2, ge=0, le=60)
    tracker_profile: Literal["balanced", "fast_motion", "small_object", "capture_card_reforger", "memory_strong"] = "balanced"
    tracker_profile_auto: bool = True
    tracker_iou_threshold: float = Field(default=0.35, ge=0.01, le=0.95)
    tracker_max_age: int = Field(default=18, ge=1, le=300)
    tracker_memory_age: int = Field(default=45, ge=1, le=900)
    tracker_reid_distance_scale: float = Field(default=2.4, ge=0.5, le=8.0)
    tracker_min_size_similarity: float = Field(default=0.2, ge=0.0, le=1.0)
    tracker_use_appearance: bool = True
    tracker_appearance_weight: float = Field(default=0.32, ge=0.0, le=2.0)
    tracker_min_appearance_similarity: float = Field(default=0.14, ge=0.0, le=1.0)
    tracker_appearance_ema_alpha: float = Field(default=0.55, ge=0.0, le=1.0)
    overlay_show_children: bool = True
    overlay_parent_thickness: int = Field(default=2, ge=1, le=8)
    overlay_child_thickness: int = Field(default=1, ge=1, le=8)
    overlay_child_shade_delta: float = Field(default=0.30, ge=-1.0, le=1.0)
    hierarchy_debug_export: bool = False
    range_mode: Literal["relative_depth", "relative_depth_first_observation", "calibrated_metric"] = "relative_depth"
    range_calibration: dict[str, float] | None = None
    range_calibration_preset: str | None = None
    range_calibration_presets: dict[str, dict[str, float]] = Field(default_factory=dict)
    zones: list[dict[str, object]] = Field(default_factory=list)


def _sanitized_camera(camera: dict[str, object]) -> dict[str, object]:
    out = dict(camera)
    out.setdefault("detection_enabled", False)
    out.setdefault("recording_mode", "event_only")
    out.setdefault("inference_max_side", 960)
    out.setdefault("detector_profile", "balanced")
    out.setdefault("detector_profile_auto", True)
    out.setdefault("child_attach_mode", "balanced")
    out.setdefault("detector_confidence", 0.25)
    out.setdefault("detector_image_size", 960)
    out.setdefault("detector_max_boxes", 80)
    out.setdefault("detector_motion_threshold", 28)
    out.setdefault("detector_min_motion_area", 900)
    out.setdefault("detector_smoothing_alpha", 0.42)
    out.setdefault("detector_hysteresis_frames", 2)
    out.setdefault("tracker_profile", "balanced")
    out.setdefault("tracker_profile_auto", True)
    out.setdefault("tracker_iou_threshold", 0.35)
    out.setdefault("tracker_max_age", 18)
    out.setdefault("tracker_memory_age", 45)
    out.setdefault("tracker_reid_distance_scale", 2.4)
    out.setdefault("tracker_min_size_similarity", 0.2)
    out.setdefault("tracker_use_appearance", True)
    out.setdefault("tracker_appearance_weight", 0.32)
    out.setdefault("tracker_min_appearance_similarity", 0.14)
    out.setdefault("tracker_appearance_ema_alpha", 0.55)
    out.setdefault("overlay_show_children", True)
    out.setdefault("overlay_parent_thickness", 2)
    out.setdefault("overlay_child_thickness", 1)
    out.setdefault("overlay_child_shade_delta", 0.30)
    out.setdefault("hierarchy_debug_export", False)
    out.setdefault("range_mode", "relative_depth")
    out.setdefault("range_calibration", None)
    out.setdefault("range_calibration_preset", None)
    out.setdefault("range_calibration_presets", {})
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
