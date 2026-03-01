from __future__ import annotations

import time

import numpy as np

from sentinel.camera.base import CameraSource, FramePacket
from sentinel.pipeline.runtime import CameraWorker, PipelineRuntime
from sentinel.storage.db import Database
from sentinel.storage.repo import EventQuery, StorageRepo
from sentinel.util.security import SecretStore
from sentinel.util.time import monotonic_ns, now_local_iso, now_utc_iso
from sentinel.vision.detect_base import Detection, DetectionChild


class _NeverConnectSource(CameraSource):
    def __init__(self) -> None:
        self.close_calls = 0

    def connect(self) -> bool:
        return False

    def read_frame(self) -> FramePacket | None:
        return None

    def health(self) -> dict[str, object]:
        return {"connected": False}

    def reconnect(self) -> bool:
        return False

    def close(self) -> None:
        self.close_calls += 1


class _ExplodingFrameSource(CameraSource):
    def __init__(self) -> None:
        self.close_calls = 0

    def connect(self) -> bool:
        return True

    def read_frame(self) -> FramePacket | None:
        raise RuntimeError("frame read exploded")

    def health(self) -> dict[str, object]:
        return {"connected": True}

    def reconnect(self) -> bool:
        return True

    def close(self) -> None:
        self.close_calls += 1


class _SingleFrameSource(CameraSource):
    def __init__(self) -> None:
        self._sent = False
        self._connected = False

    def connect(self) -> bool:
        self._connected = True
        return True

    def read_frame(self) -> FramePacket | None:
        if not self._connected or self._sent:
            return None
        self._sent = True
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        return FramePacket(frame=frame, wall_time_iso=now_utc_iso(), local_time_iso=now_local_iso(), monotonic_ns=monotonic_ns())

    def health(self) -> dict[str, object]:
        return {"connected": self._connected}

    def reconnect(self) -> bool:
        return False

    def close(self) -> None:
        self._connected = False


def test_worker_stop_interrupts_long_backoff_wait(tmp_path, monkeypatch) -> None:
    db = Database(tmp_path / "db" / "sentinel.db")
    repo = StorageRepo(db)
    secret_store = SecretStore(tmp_path)
    source = _NeverConnectSource()

    monkeypatch.setattr("sentinel.pipeline.runtime.ReconnectState.register_failure", lambda self: 8.0)

    worker = CameraWorker(
        camera_cfg={
            "id": "cam-test",
            "name": "Test Cam",
            "kind": "webcam",
            "source": "0",
            "enabled": True,
            "detection_enabled": False,
            "recording_mode": "event_only",
        },
        repo=repo,
        data_dir=tmp_path,
        secret_store=secret_store,
        global_thresholds={"person": 0.35, "animal": 0.3, "vehicle": 0.35, "unknown": 0.45},
        is_armed=lambda: False,
    )
    monkeypatch.setattr(worker, "_build_source", lambda: source)

    worker.start()
    assert worker._thread is not None
    assert worker._thread.daemon is True
    time.sleep(0.15)
    worker.stop(timeout=1.0)

    deadline = time.perf_counter() + 1.0
    while worker._thread and worker._thread.is_alive() and time.perf_counter() < deadline:
        time.sleep(0.02)

    assert not worker._thread or not worker._thread.is_alive()
    assert source.close_calls >= 1

    db.close()


def test_worker_handles_frame_exceptions_without_thread_crash(tmp_path, monkeypatch) -> None:
    db = Database(tmp_path / "db" / "sentinel.db")
    repo = StorageRepo(db)
    secret_store = SecretStore(tmp_path)
    source = _ExplodingFrameSource()

    monkeypatch.setattr("sentinel.pipeline.runtime.ReconnectState.register_failure", lambda self: 0.05)

    worker = CameraWorker(
        camera_cfg={
            "id": "cam-exploding",
            "name": "Exploding Cam",
            "kind": "webcam",
            "source": "0",
            "enabled": True,
            "detection_enabled": False,
            "recording_mode": "event_only",
        },
        repo=repo,
        data_dir=tmp_path,
        secret_store=secret_store,
        global_thresholds={"person": 0.35, "animal": 0.3, "vehicle": 0.35, "unknown": 0.45},
        is_armed=lambda: False,
    )
    monkeypatch.setattr(worker, "_build_source", lambda: source)

    worker.start()
    time.sleep(0.2)
    assert worker._thread is not None
    assert worker._thread.is_alive()
    assert str(worker.snapshot().get("last_error", "")).startswith("read_exception:")

    worker.stop(timeout=1.0)
    assert not worker.is_running()
    assert source.close_calls >= 1

    db.close()


def test_duplicate_webcam_sources_are_suppressed() -> None:
    desired, suppressed = PipelineRuntime._resolve_enabled_cameras(
        [
            {"id": "cam-a", "name": "A", "kind": "webcam", "source": "0", "enabled": True},
            {"id": "cam-b", "name": "B", "kind": "webcam", "source": "00", "enabled": True},
            {"id": "cam-c", "name": "C", "kind": "rtsp", "source": "rtsp://example", "enabled": True},
            {"id": "cam-d", "name": "D", "kind": "webcam", "source": "1", "enabled": False},
        ]
    )

    assert set(desired.keys()) == {"cam-a", "cam-c"}
    assert set(suppressed.keys()) == {"cam-b"}
    assert str(suppressed["cam-b"]["last_error"]).startswith("duplicate_webcam_source:0")


def test_invalid_camera_ids_are_suppressed() -> None:
    desired, suppressed = PipelineRuntime._resolve_enabled_cameras(
        [
            {"id": "cam-valid", "name": "Valid", "kind": "webcam", "source": "0", "enabled": True},
            {"id": "../escape", "name": "Bad", "kind": "webcam", "source": "1", "enabled": True},
        ]
    )
    assert set(desired.keys()) == {"cam-valid"}
    assert "../escape" in suppressed
    assert suppressed["../escape"]["last_error"] == "invalid_camera_id"


def test_prepare_inference_frame_downscales_and_rescales_bboxes() -> None:
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    inference_frame, scale_x, scale_y = CameraWorker._prepare_inference_frame(frame, max_side=960)

    assert inference_frame.shape[:2] == (540, 960)
    assert scale_x == 2.0
    assert scale_y == 2.0

    detections = [Detection(bbox=(10, 20, 110, 220), confidence=0.9, label="person")]
    rescaled = CameraWorker._rescale_detections(detections, scale_x, scale_y, frame_width=1920, frame_height=1080)
    assert rescaled[0].bbox == (20, 40, 220, 440)


def test_rescale_detections_scales_children_with_parent() -> None:
    detections = [
        Detection(
            bbox=(50, 60, 150, 260),
            confidence=0.9,
            label="person",
            children=[DetectionChild(bbox=(70, 110, 110, 170), confidence=0.6, label="limb", raw_label="limb")],
        )
    ]
    rescaled = CameraWorker._rescale_detections(detections, 2.0, 2.0, frame_width=1920, frame_height=1080)
    assert rescaled[0].bbox == (100, 120, 300, 520)
    assert rescaled[0].children[0].bbox == (140, 220, 220, 340)


def test_annotate_uses_shaded_child_color() -> None:
    parent_color = CameraWorker._base_color_for_label("person")
    child_color = CameraWorker._shade_color(parent_color, factor=0.30)
    assert parent_color != child_color
    assert sum(child_color) > sum(parent_color)


def test_hierarchy_debug_writer_serializes_tracks(tmp_path) -> None:
    db = Database(tmp_path / "db" / "sentinel.db")
    repo = StorageRepo(db)
    secret_store = SecretStore(tmp_path)
    worker = CameraWorker(
        camera_cfg={
            "id": "cam-debug",
            "name": "Debug Cam",
            "kind": "webcam",
            "source": "0",
            "enabled": True,
            "detection_enabled": True,
            "recording_mode": "event_only",
        },
        repo=repo,
        data_dir=tmp_path,
        secret_store=secret_store,
        global_thresholds={"person": 0.35, "animal": 0.3, "vehicle": 0.35, "unknown": 0.45},
        is_armed=lambda: True,
    )
    path = tmp_path / "debug.jsonl"
    worker._write_hierarchy_debug(
        path=path,
        wall_time_iso="2026-01-01T00:00:00+00:00",
        motion=0.2,
        tracks=[
            type(
                "T",
                (),
                {
                    "track_id": 1,
                    "label": "person",
                    "confidence": 0.9,
                    "bbox": (10, 20, 100, 200),
                    "children": [DetectionChild(bbox=(30, 60, 60, 100), confidence=0.6, label="limb", child_id="1-c1")],
                },
            )()
        ],
        frame_shape=(240, 320, 3),
    )
    content = path.read_text(encoding="utf-8")
    assert "\"track_id\": 1" in content
    assert "\"child_id\": \"1-c1\"" in content
    db.close()


def test_detector_profile_auto_selection_and_manual_override(tmp_path) -> None:
    db = Database(tmp_path / "db" / "sentinel.db")
    repo = StorageRepo(db)
    secret_store = SecretStore(tmp_path)

    auto_worker = CameraWorker(
        camera_cfg={
            "id": "cam-auto",
            "name": "Capture Input",
            "kind": "webcam",
            "source": "elgato-capture",
            "enabled": True,
            "detector_profile": "balanced",
            "detector_profile_auto": True,
        },
        repo=repo,
        data_dir=tmp_path,
        secret_store=secret_store,
        global_thresholds={"person": 0.35, "animal": 0.3, "vehicle": 0.35, "unknown": 0.45},
        is_armed=lambda: True,
    )
    assert auto_worker._resolve_detector_profile() == "capture_card_reforger"

    manual_worker = CameraWorker(
        camera_cfg={
            "id": "cam-manual",
            "name": "Manual",
            "kind": "webcam",
            "source": "0",
            "enabled": True,
            "detector_profile": "small_object",
            "detector_profile_auto": False,
        },
        repo=repo,
        data_dir=tmp_path,
        secret_store=secret_store,
        global_thresholds={"person": 0.35, "animal": 0.3, "vehicle": 0.35, "unknown": 0.45},
        is_armed=lambda: True,
    )
    assert manual_worker._resolve_detector_profile() == "small_object"
    db.close()


def test_tracker_build_uses_camera_memory_config(tmp_path) -> None:
    db = Database(tmp_path / "db" / "sentinel.db")
    repo = StorageRepo(db)
    secret_store = SecretStore(tmp_path)
    worker = CameraWorker(
        camera_cfg={
            "id": "cam-track",
            "name": "Track Tuned",
            "kind": "webcam",
            "source": "0",
            "enabled": True,
            "tracker_profile": "memory_strong",
            "tracker_profile_auto": False,
            "tracker_iou_threshold": 0.22,
            "tracker_max_age": 32,
            "tracker_memory_age": 90,
            "tracker_reid_distance_scale": 3.0,
            "tracker_min_size_similarity": 0.1,
            "tracker_use_appearance": True,
            "tracker_appearance_weight": 0.6,
            "tracker_min_appearance_similarity": 0.2,
            "tracker_appearance_ema_alpha": 0.7,
        },
        repo=repo,
        data_dir=tmp_path,
        secret_store=secret_store,
        global_thresholds={"person": 0.35, "animal": 0.3, "vehicle": 0.35, "unknown": 0.45},
        is_armed=lambda: True,
    )
    tracker = worker._build_tracker()
    assert tracker.iou_threshold == 0.22
    assert tracker.max_age == 32
    assert tracker.memory_max_age == 90
    assert tracker.reid_distance_scale == 3.0
    assert tracker.min_size_similarity == 0.1
    assert tracker.use_appearance is True
    assert tracker.appearance_weight == 0.6
    assert tracker.min_appearance_similarity == 0.2
    assert tracker.appearance_ema_alpha == 0.7
    db.close()


def test_tracker_profile_auto_applies_capture_card_defaults(tmp_path) -> None:
    db = Database(tmp_path / "db" / "sentinel.db")
    repo = StorageRepo(db)
    secret_store = SecretStore(tmp_path)
    worker = CameraWorker(
        camera_cfg={
            "id": "cam-track-auto",
            "name": "Elgato Capture",
            "kind": "webcam",
            "source": "elgato-capture",
            "enabled": True,
            "detector_profile": "balanced",
            "detector_profile_auto": True,
            "tracker_profile": "balanced",
            "tracker_profile_auto": True,
            # Keep balanced defaults so profile auto-tuning can take effect.
            "tracker_iou_threshold": 0.35,
            "tracker_max_age": 18,
            "tracker_memory_age": 45,
            "tracker_reid_distance_scale": 2.4,
            "tracker_min_size_similarity": 0.2,
            "tracker_use_appearance": True,
            "tracker_appearance_weight": 0.32,
            "tracker_min_appearance_similarity": 0.14,
            "tracker_appearance_ema_alpha": 0.55,
        },
        repo=repo,
        data_dir=tmp_path,
        secret_store=secret_store,
        global_thresholds={"person": 0.35, "animal": 0.3, "vehicle": 0.35, "unknown": 0.45},
        is_armed=lambda: True,
    )
    tracker = worker._build_tracker()
    assert tracker.max_age == 30
    assert tracker.memory_max_age == 96
    assert tracker.reid_distance_scale == 3.7
    assert tracker.appearance_weight == 0.44
    db.close()


def test_attach_detection_appearance_signature(tmp_path) -> None:
    db = Database(tmp_path / "db" / "sentinel.db")
    repo = StorageRepo(db)
    secret_store = SecretStore(tmp_path)
    worker = CameraWorker(
        camera_cfg={"id": "cam-appearance", "name": "Appearance", "kind": "webcam", "source": "0", "enabled": True},
        repo=repo,
        data_dir=tmp_path,
        secret_store=secret_store,
        global_thresholds={"person": 0.35, "animal": 0.3, "vehicle": 0.35, "unknown": 0.45},
        is_armed=lambda: True,
    )
    frame = np.zeros((100, 120, 3), dtype=np.uint8)
    frame[20:80, 30:90, 2] = 255
    detections = [Detection(bbox=(30, 20, 90, 80), confidence=0.9, label="person")]
    enriched = worker._attach_detection_appearance(frame, detections)
    assert enriched[0].appearance_signature is not None
    assert len(enriched[0].appearance_signature or ()) == 24
    db.close()


def test_stream_subscriber_count_cleans_up_when_stream_stops(tmp_path) -> None:
    db = Database(tmp_path / "db" / "sentinel.db")
    repo = StorageRepo(db)
    secret_store = SecretStore(tmp_path)

    worker = CameraWorker(
        camera_cfg={
            "id": "cam-stream",
            "name": "Stream Cam",
            "kind": "webcam",
            "source": "0",
            "enabled": True,
            "detection_enabled": False,
            "recording_mode": "event_only",
        },
        repo=repo,
        data_dir=tmp_path,
        secret_store=secret_store,
        global_thresholds={"person": 0.35, "animal": 0.3, "vehicle": 0.35, "unknown": 0.45},
        is_armed=lambda: False,
    )

    worker._stop_event.set()
    stream_iter = worker.stream()
    stream_exhausted = False
    try:
        next(stream_iter)
    except StopIteration:
        stream_exhausted = True

    assert stream_exhausted is True
    assert worker._stream_subscribers == 0
    assert worker.latest_jpeg() is None
    db.close()


def test_worker_batches_emitted_events_into_single_repo_call(tmp_path, monkeypatch) -> None:
    db = Database(tmp_path / "db" / "sentinel.db")
    repo = StorageRepo(db)
    secret_store = SecretStore(tmp_path)
    repo.upsert_camera({"id": "cam-batch-w", "name": "Batch Worker", "kind": "webcam", "source": "0", "enabled": True})

    source = _SingleFrameSource()
    calls = {"batch": 0, "single": 0}
    original_insert_events = repo.insert_events
    original_insert_event = repo.insert_event

    def _evaluate(_self, camera_cfg, tracks, motion_score, wall_time_iso, local_time_iso, monotonic_ns_value):
        _ = tracks, motion_score
        return [
            {
                "id": "evt-worker-1",
                "created_at": wall_time_iso,
                "local_time": local_time_iso,
                "monotonic_ns": monotonic_ns_value,
                "camera_id": str(camera_cfg["id"]),
                "event_type": "motion_detection",
                "label": "person",
                "confidence": 0.9,
                "track_id": 1,
                "zone": None,
                "motion": 0.2,
                "reviewed": False,
                "exported": False,
                "search_text": "worker person",
                "metadata": {},
            },
            {
                "id": "evt-worker-2",
                "created_at": wall_time_iso,
                "local_time": local_time_iso,
                "monotonic_ns": monotonic_ns_value,
                "camera_id": str(camera_cfg["id"]),
                "event_type": "motion_detection",
                "label": "animal",
                "confidence": 0.88,
                "track_id": 2,
                "zone": None,
                "motion": 0.2,
                "reviewed": False,
                "exported": False,
                "search_text": "worker animal",
                "metadata": {},
            },
        ]

    def _insert_events(events):
        calls["batch"] += 1
        return original_insert_events(events)

    def _insert_event(event):
        calls["single"] += 1
        return original_insert_event(event)

    class _Detector:
        def detect(self, frame):
            _ = frame
            return []

    worker = CameraWorker(
        camera_cfg={
            "id": "cam-batch-w",
            "name": "Batch Worker",
            "kind": "webcam",
            "source": "0",
            "enabled": True,
            "detection_enabled": True,
            "recording_mode": "event_only",
            "target_fps": 30,
        },
        repo=repo,
        data_dir=tmp_path,
        secret_store=secret_store,
        global_thresholds={"person": 0.35, "animal": 0.3, "vehicle": 0.35, "unknown": 0.45},
        is_armed=lambda: True,
    )
    monkeypatch.setattr(worker, "_build_source", lambda: source)
    monkeypatch.setattr(repo, "insert_events", _insert_events)
    monkeypatch.setattr(repo, "insert_event", _insert_event)
    monkeypatch.setattr("sentinel.pipeline.runtime.EventEngine.evaluate", _evaluate)
    monkeypatch.setattr("sentinel.pipeline.runtime.create_default_detector", lambda model_name="yolov8n.pt": _Detector())
    monkeypatch.setattr("sentinel.pipeline.runtime.ReconnectState.register_failure", lambda self: 0.02)
    monkeypatch.setattr("sentinel.pipeline.runtime.MediaRecorder.write_thumbnail", lambda self, camera_id, event_id, frame, wall_time_iso: None)
    monkeypatch.setattr("sentinel.pipeline.runtime.MediaRecorder.write_clip", lambda self, camera_id, event_id, frames, wall_time_iso, fps: None)

    worker.start()
    time.sleep(0.2)
    worker.stop(timeout=1.0)

    assert calls["batch"] >= 1
    assert calls["single"] == 0
    persisted = repo.query_events(EventQuery(camera_id="cam-batch-w", limit=10))
    assert len(persisted) == 2
    db.close()


def test_worker_falls_back_to_single_event_inserts_when_batch_fails(tmp_path, monkeypatch) -> None:
    db = Database(tmp_path / "db" / "sentinel.db")
    repo = StorageRepo(db)
    secret_store = SecretStore(tmp_path)
    repo.upsert_camera({"id": "cam-fallback", "name": "Fallback Worker", "kind": "webcam", "source": "0", "enabled": True})

    source = _SingleFrameSource()
    calls = {"batch": 0, "single": 0}
    original_insert_event = repo.insert_event

    def _evaluate(_self, camera_cfg, tracks, motion_score, wall_time_iso, local_time_iso, monotonic_ns_value):
        _ = tracks, motion_score
        return [
            {
                "id": "evt-fallback-1",
                "created_at": wall_time_iso,
                "local_time": local_time_iso,
                "monotonic_ns": monotonic_ns_value,
                "camera_id": str(camera_cfg["id"]),
                "event_type": "motion_detection",
                "label": "person",
                "confidence": 0.9,
                "track_id": 1,
                "zone": None,
                "motion": 0.2,
                "reviewed": False,
                "exported": False,
                "search_text": "fallback person",
                "metadata": {},
            },
            {
                "id": "evt-fallback-2",
                "created_at": wall_time_iso,
                "local_time": local_time_iso,
                "monotonic_ns": monotonic_ns_value,
                "camera_id": str(camera_cfg["id"]),
                "event_type": "motion_detection",
                "label": "vehicle",
                "confidence": 0.9,
                "track_id": 2,
                "zone": None,
                "motion": 0.2,
                "reviewed": False,
                "exported": False,
                "search_text": "fallback vehicle",
                "metadata": {},
            },
        ]

    def _insert_events_fail(events):
        _ = events
        calls["batch"] += 1
        raise RuntimeError("batch write failed")

    def _insert_event(event):
        calls["single"] += 1
        return original_insert_event(event)

    class _Detector:
        def detect(self, frame):
            _ = frame
            return []

    worker = CameraWorker(
        camera_cfg={
            "id": "cam-fallback",
            "name": "Fallback Worker",
            "kind": "webcam",
            "source": "0",
            "enabled": True,
            "detection_enabled": True,
            "recording_mode": "event_only",
            "target_fps": 30,
        },
        repo=repo,
        data_dir=tmp_path,
        secret_store=secret_store,
        global_thresholds={"person": 0.35, "animal": 0.3, "vehicle": 0.35, "unknown": 0.45},
        is_armed=lambda: True,
    )
    monkeypatch.setattr(worker, "_build_source", lambda: source)
    monkeypatch.setattr(repo, "insert_events", _insert_events_fail)
    monkeypatch.setattr(repo, "insert_event", _insert_event)
    monkeypatch.setattr("sentinel.pipeline.runtime.EventEngine.evaluate", _evaluate)
    monkeypatch.setattr("sentinel.pipeline.runtime.create_default_detector", lambda model_name="yolov8n.pt": _Detector())
    monkeypatch.setattr("sentinel.pipeline.runtime.ReconnectState.register_failure", lambda self: 0.02)
    monkeypatch.setattr("sentinel.pipeline.runtime.MediaRecorder.write_thumbnail", lambda self, camera_id, event_id, frame, wall_time_iso: None)
    monkeypatch.setattr("sentinel.pipeline.runtime.MediaRecorder.write_clip", lambda self, camera_id, event_id, frames, wall_time_iso, fps: None)

    worker.start()
    time.sleep(0.2)
    worker.stop(timeout=1.0)

    assert calls["batch"] >= 1
    assert calls["single"] >= 2
    persisted = repo.query_events(EventQuery(camera_id="cam-fallback", limit=10))
    assert len(persisted) == 2
    db.close()


def test_worker_persists_range_metadata_for_tracks(tmp_path, monkeypatch) -> None:
    db = Database(tmp_path / "db" / "sentinel.db")
    repo = StorageRepo(db)
    secret_store = SecretStore(tmp_path)
    repo.upsert_camera({"id": "cam-range", "name": "Range Worker", "kind": "webcam", "source": "0", "enabled": True})

    source = _SingleFrameSource()

    def _evaluate(_self, camera_cfg, tracks, motion_score, wall_time_iso, local_time_iso, monotonic_ns_value):
        _ = motion_score
        if not tracks:
            return []
        first = tracks[0]
        return [
            {
                "id": "evt-range-1",
                "created_at": wall_time_iso,
                "local_time": local_time_iso,
                "monotonic_ns": monotonic_ns_value,
                "camera_id": str(camera_cfg["id"]),
                "event_type": "motion_detection",
                "label": first.label,
                "confidence": first.confidence,
                "track_id": first.track_id,
                "zone": None,
                "motion": 0.2,
                "reviewed": False,
                "exported": False,
                "search_text": "range test",
                "metadata": {},
            }
        ]

    class _Detector:
        def detect(self, frame):
            _ = frame
            return [Detection(bbox=(10, 20, 110, 220), confidence=0.9, label="person")]

    worker = CameraWorker(
        camera_cfg={
            "id": "cam-range",
            "name": "Range Worker",
            "kind": "webcam",
            "source": "0",
            "enabled": True,
            "detection_enabled": True,
            "recording_mode": "event_only",
            "range_mode": "relative_depth",
            "target_fps": 30,
        },
        repo=repo,
        data_dir=tmp_path,
        secret_store=secret_store,
        global_thresholds={"person": 0.35, "animal": 0.3, "vehicle": 0.35, "unknown": 0.45},
        is_armed=lambda: True,
    )
    monkeypatch.setattr(worker, "_build_source", lambda: source)
    monkeypatch.setattr("sentinel.pipeline.runtime.EventEngine.evaluate", _evaluate)
    monkeypatch.setattr("sentinel.pipeline.runtime.create_default_detector", lambda model_name="yolov8n.pt": _Detector())
    monkeypatch.setattr("sentinel.pipeline.runtime.ReconnectState.register_failure", lambda self: 0.02)
    monkeypatch.setattr("sentinel.pipeline.runtime.MediaRecorder.write_thumbnail", lambda self, camera_id, event_id, frame, wall_time_iso: None)
    monkeypatch.setattr("sentinel.pipeline.runtime.MediaRecorder.write_clip", lambda self, camera_id, event_id, frames, wall_time_iso, fps: None)

    worker.start()
    time.sleep(0.2)
    worker.stop(timeout=1.0)

    persisted = repo.query_events(EventQuery(camera_id="cam-range", limit=10))
    assert len(persisted) == 1
    metadata = persisted[0]["metadata"]
    assert "range" in metadata
    assert metadata["range"]["depth_rel"] is not None
    db.close()
