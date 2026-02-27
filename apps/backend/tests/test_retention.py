from __future__ import annotations

import datetime as dt

from sentinel.storage.db import Database
from sentinel.storage.repo import EventQuery, StorageRepo
from sentinel.storage.retention import RetentionService


def test_retention_deletes_old_records_and_media(tmp_path) -> None:
    data_dir = tmp_path / "data"
    (data_dir / "db").mkdir(parents=True, exist_ok=True)
    (data_dir / "media" / "cam-1").mkdir(parents=True, exist_ok=True)

    db = Database(data_dir / "db" / "sentinel.db")
    repo = StorageRepo(db)
    repo.upsert_camera({"id": "cam-1", "name": "A", "kind": "webcam", "source": "0", "enabled": True})

    old_thumb = data_dir / "media" / "cam-1" / "old.jpg"
    new_thumb = data_dir / "media" / "cam-1" / "new.jpg"
    old_thumb.write_bytes(b"old")
    new_thumb.write_bytes(b"new")

    now = dt.datetime.now(dt.timezone.utc)
    old_time = (now - dt.timedelta(days=10)).isoformat()
    new_time = now.isoformat()

    repo.insert_event(
        {
            "id": "old-event",
            "created_at": old_time,
            "local_time": old_time,
            "monotonic_ns": 1,
            "camera_id": "cam-1",
            "event_type": "motion_detection",
            "label": "person",
            "confidence": 0.8,
            "track_id": 1,
            "zone": None,
            "motion": 0.2,
            "thumbnail_path": str(old_thumb.relative_to(data_dir)),
            "clip_path": None,
            "reviewed": False,
            "exported": False,
            "search_text": "old",
            "metadata": {},
        }
    )
    repo.insert_event(
        {
            "id": "new-event",
            "created_at": new_time,
            "local_time": new_time,
            "monotonic_ns": 2,
            "camera_id": "cam-1",
            "event_type": "motion_detection",
            "label": "person",
            "confidence": 0.8,
            "track_id": 2,
            "zone": None,
            "motion": 0.2,
            "thumbnail_path": str(new_thumb.relative_to(data_dir)),
            "clip_path": None,
            "reviewed": False,
            "exported": False,
            "search_text": "new",
            "metadata": {},
        }
    )

    retention = RetentionService(repo, data_dir)
    summary = retention.enforce(days=5, max_gb=None)

    assert summary["deleted_by_age"] == 1
    assert not old_thumb.exists()
    assert new_thumb.exists()

    remaining = repo.query_events(EventQuery(limit=100))
    assert len(remaining) == 1
    assert remaining[0]["id"] == "new-event"
