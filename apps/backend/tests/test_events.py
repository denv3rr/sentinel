from __future__ import annotations

import datetime as dt

from sentinel.pipeline.events import EventEngine
from sentinel.storage.db import Database
from sentinel.storage.repo import EventQuery, StorageRepo
from sentinel.vision.track_base import TrackedObject


def test_event_dedupe_and_cooldown() -> None:
    engine = EventEngine(global_thresholds={"person": 0.35, "animal": 0.30, "vehicle": 0.35, "unknown": 0.45})
    camera_cfg = {
        "id": "cam-1",
        "name": "Front Door",
        "labels": ["person", "vehicle", "animal", "unknown"],
        "min_confidence": {"person": 0.4},
        "cooldown_seconds": 5,
        "motion_threshold": 0.01,
        "zones": [],
    }
    tracks = [
        TrackedObject(track_id=11, bbox=(10, 10, 120, 220), confidence=0.92, label="person"),
    ]

    first = engine.evaluate(camera_cfg, tracks, motion_score=0.2, wall_time_iso="2026-01-01T00:00:00+00:00", local_time_iso="2025-12-31T19:00:00-05:00", monotonic_ns=10_000_000_000)
    second = engine.evaluate(camera_cfg, tracks, motion_score=0.2, wall_time_iso="2026-01-01T00:00:01+00:00", local_time_iso="2025-12-31T19:00:01-05:00", monotonic_ns=12_000_000_000)
    third = engine.evaluate(camera_cfg, tracks, motion_score=0.2, wall_time_iso="2026-01-01T00:00:07+00:00", local_time_iso="2025-12-31T19:00:07-05:00", monotonic_ns=16_500_000_000)

    assert len(first) == 1
    assert len(second) == 0
    assert len(third) == 1


def test_event_query_filters(tmp_path) -> None:
    db = Database(tmp_path / "db" / "sentinel.db")
    repo = StorageRepo(db)

    repo.upsert_camera({"id": "cam-a", "name": "A", "kind": "webcam", "source": "0", "enabled": True})
    repo.upsert_camera({"id": "cam-b", "name": "B", "kind": "webcam", "source": "1", "enabled": True})

    base = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)

    def insert(eid: str, camera_id: str, label: str, conf: float, offset_sec: int) -> None:
        when = (base + dt.timedelta(seconds=offset_sec)).isoformat()
        repo.insert_event(
            {
                "id": eid,
                "created_at": when,
                "local_time": when,
                "monotonic_ns": 1,
                "camera_id": camera_id,
                "event_type": "motion_detection",
                "label": label,
                "confidence": conf,
                "track_id": 1,
                "zone": None,
                "motion": 0.12,
                "thumbnail_path": None,
                "clip_path": None,
                "reviewed": False,
                "exported": False,
                "search_text": f"{camera_id} {label}",
                "metadata": {},
            }
        )

    insert("e1", "cam-a", "person", 0.95, 0)
    insert("e2", "cam-a", "vehicle", 0.60, 10)
    insert("e3", "cam-b", "person", 0.45, 20)

    q1 = EventQuery(camera_id="cam-a", limit=50)
    r1 = repo.query_events(q1)
    assert len(r1) == 2

    q2 = EventQuery(label="person", min_confidence=0.5, limit=50)
    r2 = repo.query_events(q2)
    assert len(r2) == 1
    assert r2[0]["id"] == "e1"

    q3 = EventQuery(start=(base + dt.timedelta(seconds=15)).isoformat(), limit=50)
    r3 = repo.query_events(q3)
    assert len(r3) == 1
    assert r3[0]["id"] == "e3"