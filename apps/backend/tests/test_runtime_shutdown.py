from __future__ import annotations

import time

from sentinel.camera.base import CameraSource, FramePacket
from sentinel.pipeline.runtime import CameraWorker, PipelineRuntime
from sentinel.storage.db import Database
from sentinel.storage.repo import StorageRepo
from sentinel.util.security import SecretStore


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
            {"id": "cam-b", "name": "B", "kind": "webcam", "source": "0", "enabled": True},
            {"id": "cam-c", "name": "C", "kind": "rtsp", "source": "rtsp://example", "enabled": True},
            {"id": "cam-d", "name": "D", "kind": "webcam", "source": "1", "enabled": False},
        ]
    )

    assert set(desired.keys()) == {"cam-a", "cam-c"}
    assert set(suppressed.keys()) == {"cam-b"}
    assert str(suppressed["cam-b"]["last_error"]).startswith("duplicate_webcam_source:0")
