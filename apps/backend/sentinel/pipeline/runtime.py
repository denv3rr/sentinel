from __future__ import annotations

import json
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
import numpy as np

from sentinel.camera.base import CameraSource
from sentinel.camera.health import ReconnectState
from sentinel.camera.opencv_cam import OpenCVCameraSource
from sentinel.camera.rtsp import RTSPCameraSource
from sentinel.pipeline.events import EventEngine
from sentinel.pipeline.recorder import MediaRecorder
from sentinel.storage.repo import StorageRepo
from sentinel.util.logging import get_logger
from sentinel.util.security import SecretStore, sanitize_rtsp_url, validate_camera_id
from sentinel.util.time import monotonic_ns as monotonic_ns_now
from sentinel.util.time import now_local_iso, now_utc_iso
from sentinel.vision.detect_base import Detection
from sentinel.vision.tracker_default import DefaultIoUTracker
from sentinel.vision.yolo_ultralytics import create_default_detector

logger = get_logger(__name__)


def _normalize_webcam_source(value: object) -> str:
    source = str(value).strip()
    if source.isdigit():
        return str(int(source))
    return source


@dataclass
class WorkerStatus:
    camera_id: str
    name: str
    online: bool = False
    fps: float = 0.0
    frame_drops: int = 0
    reconnects: int = 0
    inference_ms: float = 0.0
    encode_ms: float = 0.0
    persist_ms: float = 0.0
    persisted_events: int = 0
    last_error: str | None = None
    last_frame_wall_time: str | None = None
    safe_source: str = ""


@dataclass
class CameraWorker:
    camera_cfg: dict[str, Any]
    repo: StorageRepo
    data_dir: Path
    secret_store: SecretStore
    global_thresholds: dict[str, float]
    is_armed: Callable[[], bool]
    model_name: str = "yolov8n.pt"
    _thread: threading.Thread | None = field(default=None, init=False)
    _stop_event: threading.Event = field(default_factory=threading.Event, init=False)
    _status_lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _jpeg_lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _stream_lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _source_lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _active_source: CameraSource | None = field(default=None, init=False)
    _status: WorkerStatus = field(init=False)
    _latest_jpeg: bytes | None = field(default=None, init=False)
    _stream_subscribers: int = field(default=0, init=False)
    _config_hash: str = field(init=False)

    def __post_init__(self) -> None:
        source = str(self.camera_cfg.get("source", ""))
        safe_source = sanitize_rtsp_url(source) if self.camera_cfg.get("kind") != "webcam" else _normalize_webcam_source(source)
        self._status = WorkerStatus(
            camera_id=str(self.camera_cfg["id"]),
            name=str(self.camera_cfg.get("name", self.camera_cfg["id"])),
            safe_source=safe_source,
        )
        self._config_hash = json.dumps(self.camera_cfg, sort_keys=True)

    @property
    def config_hash(self) -> str:
        return self._config_hash

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        # Keep worker cooperative-stop first; daemon mode is only a final fail-safe
        # against OpenCV backend calls that can block thread exit on shutdown.
        self._thread = threading.Thread(
            target=self._run_loop,
            name=f"camera-{self.camera_cfg['id']}",
            daemon=True,
        )
        self._thread.start()

    def request_stop(self) -> None:
        self._stop_event.set()
        with self._source_lock:
            source = self._active_source
        if source is not None:
            try:
                source.close()
            except Exception:
                logger.debug("camera source close failed during stop: %s", self.camera_cfg.get("id"), exc_info=True)

    def wait_stopped(self, timeout: float = 3.0) -> None:
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=timeout)
        if thread and not thread.is_alive():
            self._thread = None

    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def stop(self, timeout: float = 3.0) -> None:
        self.request_stop()
        self.wait_stopped(timeout=timeout)

    def snapshot(self) -> dict[str, Any]:
        with self._status_lock:
            return {
                "camera_id": self._status.camera_id,
                "name": self._status.name,
                "online": self._status.online,
                "fps": round(self._status.fps, 2),
                "frame_drops": self._status.frame_drops,
                "reconnects": self._status.reconnects,
                "inference_ms": round(self._status.inference_ms, 2),
                "encode_ms": round(self._status.encode_ms, 2),
                "persist_ms": round(self._status.persist_ms, 2),
                "persisted_events": self._status.persisted_events,
                "last_error": self._status.last_error,
                "last_frame_wall_time": self._status.last_frame_wall_time,
                "source": self._status.safe_source,
            }

    def latest_jpeg(self) -> bytes | None:
        with self._jpeg_lock:
            return self._latest_jpeg

    def _register_stream_subscriber(self) -> None:
        with self._stream_lock:
            self._stream_subscribers += 1

    def _unregister_stream_subscriber(self) -> None:
        with self._stream_lock:
            if self._stream_subscribers > 0:
                self._stream_subscribers -= 1
            no_subscribers = self._stream_subscribers == 0
        if no_subscribers:
            with self._jpeg_lock:
                self._latest_jpeg = None

    def _has_stream_subscribers(self) -> bool:
        with self._stream_lock:
            return self._stream_subscribers > 0

    def stream(self) -> Iterator[bytes]:
        boundary = b"--frame\r\n"
        self._register_stream_subscriber()
        try:
            while not self._stop_event.is_set():
                payload = self.latest_jpeg()
                if payload is None:
                    if self._stop_event.wait(0.1):
                        break
                    continue
                yield boundary + b"Content-Type: image/jpeg\r\n\r\n" + payload + b"\r\n"
                if self._stop_event.wait(0.04):
                    break
        finally:
            self._unregister_stream_subscriber()

    def _set_error(self, message: str) -> None:
        with self._status_lock:
            self._status.last_error = message

    def _get_error(self) -> str | None:
        with self._status_lock:
            return self._status.last_error

    def _set_online(self, online: bool) -> None:
        with self._status_lock:
            self._status.online = online

    def _set_frame_meta(self, wall_time: str, fps: float) -> None:
        with self._status_lock:
            self._status.last_frame_wall_time = wall_time
            self._status.fps = fps

    def _increment_drop(self) -> None:
        with self._status_lock:
            self._status.frame_drops += 1

    def _increment_reconnect(self) -> None:
        with self._status_lock:
            self._status.reconnects += 1

    def _update_perf_metrics(
        self,
        inference_ms: float | None = None,
        encode_ms: float | None = None,
        persist_ms: float | None = None,
        persisted_events_delta: int = 0,
    ) -> None:
        with self._status_lock:
            if inference_ms is not None:
                self._status.inference_ms = inference_ms
            if encode_ms is not None:
                self._status.encode_ms = encode_ms
            if persist_ms is not None:
                self._status.persist_ms = persist_ms
            if persisted_events_delta:
                self._status.persisted_events += persisted_events_delta

    def _set_active_source(self, source: CameraSource | None) -> None:
        with self._source_lock:
            self._active_source = source

    def _build_source(self) -> CameraSource:
        kind = str(self.camera_cfg.get("kind", "webcam"))
        source = str(self.camera_cfg.get("source", "0"))

        if kind in {"rtsp", "onvif"}:
            secret_ref = self.camera_cfg.get("secret_ref")
            if isinstance(secret_ref, dict):
                resolved = self.secret_store.get(secret_ref)
                if resolved:
                    source = resolved
            return RTSPCameraSource(source, name=str(self.camera_cfg.get("name", "RTSP")))

        normalized_source = _normalize_webcam_source(source)
        webcam_source: int | str = normalized_source
        if normalized_source.isdigit():
            webcam_source = int(normalized_source)
        return OpenCVCameraSource(webcam_source, name=str(self.camera_cfg.get("name", "Webcam")))

    @staticmethod
    def _motion_score(prev_gray: np.ndarray | None, frame: np.ndarray) -> tuple[np.ndarray, float]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        if prev_gray is None:
            return gray, 0.0

        diff = cv2.absdiff(prev_gray, gray)
        _, mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        motion_ratio = float(np.count_nonzero(mask)) / float(mask.size)
        return gray, motion_ratio

    @staticmethod
    def _prepare_inference_frame(frame: np.ndarray, max_side: int) -> tuple[np.ndarray, float, float]:
        if max_side <= 0:
            return frame, 1.0, 1.0

        frame_height, frame_width = frame.shape[:2]
        longest_side = max(frame_height, frame_width)
        if longest_side <= max_side:
            return frame, 1.0, 1.0

        resize_ratio = float(max_side) / float(longest_side)
        resized_width = max(1, int(round(frame_width * resize_ratio)))
        resized_height = max(1, int(round(frame_height * resize_ratio)))
        resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
        scale_x = float(frame_width) / float(resized_width)
        scale_y = float(frame_height) / float(resized_height)
        return resized, scale_x, scale_y

    @staticmethod
    def _rescale_detections(
        detections: list[Detection], scale_x: float, scale_y: float, frame_width: int, frame_height: int
    ) -> list[Detection]:
        if scale_x == 1.0 and scale_y == 1.0:
            return detections

        rescaled: list[Detection] = []
        max_x = max(0, frame_width - 1)
        max_y = max(0, frame_height - 1)

        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            rx1 = int(round(x1 * scale_x))
            ry1 = int(round(y1 * scale_y))
            rx2 = int(round(x2 * scale_x))
            ry2 = int(round(y2 * scale_y))

            rx1 = max(0, min(max_x, rx1))
            ry1 = max(0, min(max_y, ry1))
            rx2 = max(rx1 + 1, min(frame_width, rx2))
            ry2 = max(ry1 + 1, min(frame_height, ry2))

            rescaled.append(
                Detection(
                    bbox=(rx1, ry1, rx2, ry2),
                    confidence=detection.confidence,
                    label=detection.label,
                    raw_label=detection.raw_label,
                )
            )
        return rescaled

    @staticmethod
    def _annotate(frame: np.ndarray, tracks: list[Any]) -> np.ndarray:
        out = frame.copy()
        for track in tracks:
            x1, y1, x2, y2 = track.bbox
            cv2.rectangle(out, (x1, y1), (x2, y2), (24, 193, 145), 2)
            text = f"#{track.track_id} {track.label} {track.confidence:.2f}"
            cv2.putText(out, text, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (24, 193, 145), 1)
        return out

    def _run_loop(self) -> None:
        source: CameraSource | None = None
        camera_id = str(self.camera_cfg.get("id", "unknown"))
        detector = None
        tracker: DefaultIoUTracker | None = None
        events = EventEngine(global_thresholds=self.global_thresholds)
        recorder = MediaRecorder(self.data_dir, clip_seconds=6)
        reconnect = ReconnectState()

        target_fps = float(self.camera_cfg.get("target_fps", 10.0))
        target_interval = 1.0 / max(1.0, target_fps)
        clip_frames_max = int(max(8, target_fps * 6))
        clip_buffer: deque[np.ndarray] = deque(maxlen=clip_frames_max)
        default_inference_max_side = 960
        try:
            inference_max_side = int(self.camera_cfg.get("inference_max_side", default_inference_max_side))
        except (TypeError, ValueError):
            inference_max_side = default_inference_max_side
        inference_max_side = max(0, inference_max_side)
        detector_enabled = bool(self.camera_cfg.get("detection_enabled", False))
        recording_mode = str(self.camera_cfg.get("recording_mode", "event_only")).lower()
        continuous_enabled = recording_mode in {"full", "live"}
        continuous_segment_seconds = 300 if recording_mode == "full" else 60
        continuous_writer: cv2.VideoWriter | None = None
        continuous_rel_path: str | None = None
        continuous_started_perf = 0.0
        continuous_start_wall_iso: str | None = None

        prev_gray: np.ndarray | None = None
        frame_counter = 0
        fps_window_start = time.perf_counter()
        current_fps = 0.0
        last_process = 0.0

        def close_continuous_writer(end_wall_iso: str | None) -> None:
            nonlocal continuous_writer, continuous_rel_path, continuous_started_perf, continuous_start_wall_iso
            if continuous_writer is None:
                return
            try:
                continuous_writer.release()
            except Exception:
                logger.debug("continuous writer release failed: %s", camera_id, exc_info=True)
            if continuous_rel_path:
                event_wall = end_wall_iso or now_utc_iso()
                event_local = now_local_iso()
                try:
                    self.repo.insert_event(
                        {
                            "id": f"evt-{uuid4().hex}",
                            "created_at": event_wall,
                            "local_time": event_local,
                            "monotonic_ns": monotonic_ns_now(),
                            "camera_id": str(self.camera_cfg["id"]),
                            "event_type": "continuous_recording",
                            "label": "unknown",
                            "confidence": 1.0,
                            "track_id": None,
                            "zone": None,
                            "motion": None,
                            "thumbnail_path": None,
                            "clip_path": continuous_rel_path,
                            "reviewed": False,
                            "exported": False,
                            "search_text": f"{self.camera_cfg.get('name', self.camera_cfg['id'])} {recording_mode} recording",
                            "metadata": {
                                "recording_mode": recording_mode,
                                "segment_seconds": continuous_segment_seconds,
                                "start_time": continuous_start_wall_iso,
                                "end_time": event_wall,
                            },
                        }
                    )
                except Exception:
                    logger.exception("failed to persist continuous recording event: %s", camera_id)
            continuous_writer = None
            continuous_rel_path = None
            continuous_started_perf = 0.0
            continuous_start_wall_iso = None

        try:
            source = self._build_source()
            self._set_active_source(source)

            while not self._stop_event.is_set():
                connect_ok = False
                try:
                    connect_ok = source.connect()
                except Exception:
                    logger.exception("camera source connect failed: %s", camera_id)
                    self._set_error("connection_exception")

                if not connect_ok:
                    self._set_online(False)
                    if self._get_error() != "connection_exception":
                        self._set_error("connection_failed")
                    wait_for = reconnect.register_failure()
                    if self._stop_event.wait(wait_for):
                        break
                    continue

                self._set_online(True)
                reconnect.reset()
                self._set_error("")

                while not self._stop_event.is_set():
                    packet = None
                    try:
                        packet = source.read_frame()
                    except Exception as exc:
                        self._set_error(f"read_exception:{type(exc).__name__}")
                        logger.exception("camera source read failed: %s", camera_id)

                    if packet is None:
                        self._set_online(False)
                        current_error = self._get_error() or ""
                        if not current_error.startswith("read_exception:"):
                            self._set_error("read_failed")
                        close_continuous_writer(None)
                        wait_for = reconnect.register_failure()
                        if reconnect.should_reconnect():
                            try:
                                source.reconnect()
                                reconnect.reset()
                                self._increment_reconnect()
                            except Exception:
                                logger.exception("camera source reconnect failed: %s", camera_id)
                        if self._stop_event.wait(wait_for):
                            break
                        continue

                    self._set_online(True)
                    reconnect.reset()

                    try:
                        now_perf = time.perf_counter()
                        if (now_perf - last_process) < target_interval:
                            self._increment_drop()
                            continue
                        last_process = now_perf

                        frame = packet.frame
                        clip_buffer.append(frame.copy())
                        is_armed = bool(self.is_armed())
                        motion = 0.0
                        tracked: list[Any] = []
                        emitted: list[dict[str, object]] = []
                        inference_ms: float | None = None
                        encode_ms: float | None = None
                        persist_ms: float | None = None

                        if detector_enabled and is_armed:
                            if detector is None:
                                detector = create_default_detector(model_name=self.model_name)
                            if tracker is None:
                                tracker = DefaultIoUTracker()
                            inference_started = time.perf_counter()
                            inference_frame, scale_x, scale_y = self._prepare_inference_frame(frame, inference_max_side)
                            prev_gray, motion = self._motion_score(prev_gray, inference_frame)
                            detections = detector.detect(inference_frame)
                            if scale_x != 1.0 or scale_y != 1.0:
                                frame_height, frame_width = frame.shape[:2]
                                detections = self._rescale_detections(detections, scale_x, scale_y, frame_width, frame_height)
                            tracked = tracker.update(detections)
                            emitted = events.evaluate(
                                self.camera_cfg,
                                tracked,
                                motion,
                                packet.wall_time_iso,
                                packet.local_time_iso,
                                packet.monotonic_ns,
                            )
                            inference_ms = (time.perf_counter() - inference_started) * 1000.0
                        else:
                            prev_gray = None

                        if emitted:
                            clip_frames = list(clip_buffer)
                            for event in emitted:
                                thumb_rel = recorder.write_thumbnail(
                                    camera_id=str(self.camera_cfg["id"]),
                                    event_id=str(event["id"]),
                                    frame=frame,
                                    wall_time_iso=packet.wall_time_iso,
                                )
                                clip_rel = recorder.write_clip(
                                    camera_id=str(self.camera_cfg["id"]),
                                    event_id=str(event["id"]),
                                    frames=clip_frames,
                                    wall_time_iso=packet.wall_time_iso,
                                    fps=target_fps,
                                )
                                event["thumbnail_path"] = thumb_rel
                                event["clip_path"] = clip_rel
                            persist_started = time.perf_counter()
                            persisted_count = len(emitted)
                            try:
                                self.repo.insert_events(emitted)
                                persist_ms = (time.perf_counter() - persist_started) * 1000.0
                            except Exception:
                                logger.exception("failed to batch persist emitted events: %s", camera_id)
                                for event in emitted:
                                    try:
                                        self.repo.insert_event(event)
                                    except Exception:
                                        logger.exception("failed to persist emitted event after batch fallback: %s", camera_id)
                                persist_ms = (time.perf_counter() - persist_started) * 1000.0
                            self._update_perf_metrics(
                                persist_ms=persist_ms,
                                persisted_events_delta=persisted_count,
                            )

                        if continuous_enabled:
                            if continuous_writer is None:
                                prepared = recorder.prepare_continuous_clip_path(
                                    camera_id=str(self.camera_cfg["id"]),
                                    wall_time_iso=packet.wall_time_iso,
                                    mode=recording_mode,
                                )
                                if prepared is not None:
                                    output_path, rel_path = prepared
                                    height, width = frame.shape[:2]
                                    writer = cv2.VideoWriter(
                                        str(output_path),
                                        cv2.VideoWriter_fourcc(*"mp4v"),
                                        max(1.0, target_fps),
                                        (width, height),
                                    )
                                    if writer.isOpened():
                                        continuous_writer = writer
                                        continuous_rel_path = rel_path
                                        continuous_started_perf = now_perf
                                        continuous_start_wall_iso = packet.wall_time_iso
                                    else:
                                        writer.release()
                                        self._set_error("recording_writer_failed")
                            if continuous_writer is not None:
                                continuous_writer.write(frame)
                                if (now_perf - continuous_started_perf) >= continuous_segment_seconds:
                                    close_continuous_writer(packet.wall_time_iso)

                        if self._has_stream_subscribers():
                            encode_started = time.perf_counter()
                            annotated = self._annotate(frame, tracked)
                            if not is_armed:
                                cv2.putText(
                                    annotated,
                                    "SYSTEM DISARMED",
                                    (16, 28),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.65,
                                    (0, 215, 255),
                                    2,
                                )
                            elif not detector_enabled:
                                cv2.putText(
                                    annotated,
                                    "DETECTOR OFF",
                                    (16, 28),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.65,
                                    (0, 215, 255),
                                    2,
                                )
                            ok, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
                            if ok:
                                with self._jpeg_lock:
                                    self._latest_jpeg = encoded.tobytes()
                            encode_ms = (time.perf_counter() - encode_started) * 1000.0

                        frame_counter += 1
                        elapsed = now_perf - fps_window_start
                        if elapsed >= 1.0:
                            current_fps = frame_counter / elapsed
                            frame_counter = 0
                            fps_window_start = now_perf
                        self._set_frame_meta(packet.wall_time_iso, current_fps)
                        self._update_perf_metrics(
                            inference_ms=inference_ms,
                            encode_ms=encode_ms,
                            persist_ms=persist_ms,
                        )
                    except Exception as exc:
                        close_continuous_writer(None)
                        self._set_error(f"pipeline_error:{type(exc).__name__}")
                        logger.exception("camera frame pipeline failed: %s", camera_id)
                        wait_for = reconnect.register_failure()
                        if self._stop_event.wait(min(wait_for, 0.5)):
                            break
                        continue

                close_continuous_writer(None)
                try:
                    source.close()
                except Exception:
                    logger.debug("camera source close failed after loop: %s", camera_id, exc_info=True)
        except Exception as exc:
            self._set_error(f"pipeline_crash:{type(exc).__name__}")
            logger.exception("camera runtime loop crashed: %s", camera_id)
        finally:
            close_continuous_writer(None)
            if source is not None:
                try:
                    source.close()
                except Exception:
                    logger.debug("camera source close failed at runtime shutdown: %s", camera_id, exc_info=True)
            self._set_active_source(None)
            self._set_online(False)


class PipelineRuntime:
    def __init__(
        self,
        repo: StorageRepo,
        data_dir: Path,
        secret_store: SecretStore,
        global_thresholds: dict[str, float],
        global_armed: bool,
        model_name: str = "yolov8n.pt",
    ) -> None:
        self.repo = repo
        self.data_dir = data_dir
        self.secret_store = secret_store
        self.global_thresholds = global_thresholds
        self.global_armed = global_armed
        self.model_name = model_name
        self._workers: dict[str, CameraWorker] = {}
        self._suppressed_statuses: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._stop_event = threading.Event()

    def is_armed(self) -> bool:
        return bool(self.global_armed)

    @staticmethod
    def _resolve_enabled_cameras(
        cameras: list[dict[str, Any]],
    ) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
        desired: dict[str, dict[str, Any]] = {}
        suppressed: dict[str, dict[str, Any]] = {}
        webcam_sources_in_use: dict[str, str] = {}

        for cfg in cameras:
            if not bool(cfg.get("enabled", True)):
                continue
            raw_camera_id = str(cfg.get("id", ""))
            if not raw_camera_id:
                continue
            try:
                camera_id = validate_camera_id(raw_camera_id)
            except ValueError:
                suppressed[raw_camera_id or f"invalid-{len(suppressed)+1}"] = {
                    "camera_id": raw_camera_id or "",
                    "name": str(cfg.get("name", raw_camera_id or "invalid")),
                    "online": False,
                    "fps": 0.0,
                    "frame_drops": 0,
                    "reconnects": 0,
                    "last_error": "invalid_camera_id",
                    "last_frame_wall_time": None,
                    "source": str(cfg.get("source", "")).strip(),
                }
                logger.warning("camera suppressed due to invalid camera id: %s", raw_camera_id)
                continue

            kind = str(cfg.get("kind", "webcam")).lower()
            source = str(cfg.get("source", "")).strip()
            if kind == "webcam":
                source = _normalize_webcam_source(source)
                owner = webcam_sources_in_use.get(source)
                if owner is not None:
                    error = f"duplicate_webcam_source:{source} (already used by {owner})"
                    suppressed[camera_id] = {
                        "camera_id": camera_id,
                        "name": str(cfg.get("name", camera_id)),
                        "online": False,
                        "fps": 0.0,
                        "frame_drops": 0,
                        "reconnects": 0,
                        "last_error": error,
                        "last_frame_wall_time": None,
                        "source": source,
                    }
                    logger.warning("camera suppressed due to duplicate webcam source %s: %s", source, camera_id)
                    continue
                webcam_sources_in_use[source] = camera_id

            desired[camera_id] = cfg

        return desired, suppressed

    def sync_cameras(self, cameras: list[dict[str, Any]]) -> None:
        desired, suppressed = self._resolve_enabled_cameras(cameras)

        with self._lock:
            self._stop_event.clear()
            self._suppressed_statuses = suppressed
            existing_ids = set(self._workers.keys())
            desired_ids = set(desired.keys())

            for camera_id in sorted(existing_ids - desired_ids):
                worker = self._workers.pop(camera_id)
                worker.stop()

            for camera_id, cfg in desired.items():
                serialized = json.dumps(cfg, sort_keys=True)
                current = self._workers.get(camera_id)
                if current is None:
                    self._start_worker(cfg)
                    continue
                if current.config_hash != serialized:
                    current.stop()
                    self._workers.pop(camera_id, None)
                    self._start_worker(cfg)

    def _start_worker(self, camera_cfg: dict[str, Any]) -> None:
        worker = CameraWorker(
            camera_cfg=camera_cfg,
            repo=self.repo,
            data_dir=self.data_dir,
            secret_store=self.secret_store,
            global_thresholds=self.global_thresholds,
            is_armed=self.is_armed,
            model_name=self.model_name,
        )
        worker.start()
        self._workers[str(camera_cfg["id"])] = worker
        logger.info("camera worker started: %s", camera_cfg.get("id"))

    def stop_all(self) -> None:
        with self._lock:
            self._stop_event.set()
            self._suppressed_statuses.clear()
            workers = list(self._workers.values())
            self._workers.clear()

        for worker in workers:
            worker.request_stop()

        deadline = time.perf_counter() + 6.0
        for worker in workers:
            remaining = max(0.0, deadline - time.perf_counter())
            worker.wait_stopped(timeout=remaining)
            if worker.is_running():
                logger.warning("camera runtime thread did not stop before timeout: %s", worker.camera_cfg.get("id"))

    def statuses(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            merged = {camera_id: worker.snapshot() for camera_id, worker in self._workers.items()}
            merged.update(self._suppressed_statuses)
            return merged

    def stream(self, camera_id: str) -> Iterator[bytes]:
        with self._lock:
            worker = self._workers.get(camera_id)
            suppressed = self._suppressed_statuses.get(camera_id)
        if not worker:
            message = str(suppressed.get("last_error", "")) if suppressed else None
            yield from self._placeholder_stream(camera_id, message=message)
            return
        yield from worker.stream()

    def _placeholder_stream(self, camera_id: str, message: str | None = None) -> Iterator[bytes]:
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.putText(frame, f"Camera {camera_id} offline", (70, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        if message:
            cv2.putText(frame, message[:80], (30, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)
        ok, encoded = cv2.imencode(".jpg", frame)
        payload = encoded.tobytes() if ok else b""
        while not self._stop_event.is_set():
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + payload + b"\r\n"
            if self._stop_event.wait(0.5):
                break

    def test_camera(self, camera_cfg: dict[str, Any]) -> tuple[bool, str]:
        kind = str(camera_cfg.get("kind", "webcam"))
        source_value = str(camera_cfg.get("source", "0"))

        if kind in {"rtsp", "onvif"}:
            secret_ref = camera_cfg.get("secret_ref")
            if isinstance(secret_ref, dict):
                resolved = self.secret_store.get(secret_ref)
                if resolved:
                    source_value = resolved
            source: CameraSource = RTSPCameraSource(source_value, name=str(camera_cfg.get("name", "RTSP")))
        else:
            webcam_value: int | str = int(source_value) if source_value.isdigit() else source_value
            source = OpenCVCameraSource(webcam_value, name=str(camera_cfg.get("name", "Webcam")))

        try:
            if not source.connect():
                return False, "Unable to connect"
            packet = source.read_frame()
            if packet is None:
                return False, "Connected but no frames"
            return True, "Camera stream looks healthy"
        finally:
            source.close()
