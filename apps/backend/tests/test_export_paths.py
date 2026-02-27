from __future__ import annotations

from pathlib import Path

from sentinel.storage.db import Database
from sentinel.storage.export import ExportService
from sentinel.storage.repo import EventQuery, StorageRepo


def _seed_event(repo: StorageRepo) -> None:
    repo.upsert_camera({"id": "cam-1", "name": "Cam 1", "kind": "webcam", "source": "0", "enabled": True})
    repo.insert_event(
        {
            "id": "evt-1",
            "created_at": "2026-01-01T00:00:00+00:00",
            "local_time": "2025-12-31T19:00:00-05:00",
            "monotonic_ns": 1,
            "camera_id": "cam-1",
            "event_type": "motion_detection",
            "label": "person",
            "confidence": 0.98,
            "track_id": 1,
            "zone": None,
            "motion": 0.2,
            "thumbnail_path": None,
            "clip_path": None,
            "reviewed": False,
            "exported": False,
            "search_text": "cam-1 person",
            "metadata": {},
        }
    )


def test_export_defaults_to_data_dir(tmp_path) -> None:
    data_dir = tmp_path / "data"
    db = Database(data_dir / "db" / "sentinel.db")
    repo = StorageRepo(db)
    _seed_event(repo)

    service = ExportService(repo, data_dir)
    result = service.export("jsonl", EventQuery(limit=100))

    assert result["inside_data_dir"] is True
    target = data_dir / str(result["path"])
    assert target.exists()
    assert target.is_file()
    assert result["count"] == 1


def test_export_can_target_external_dir(tmp_path) -> None:
    data_dir = tmp_path / "data"
    external_dir = tmp_path / "external_exports"
    db = Database(data_dir / "db" / "sentinel.db")
    repo = StorageRepo(db)
    _seed_event(repo)

    service = ExportService(repo, data_dir)
    result = service.export("jsonl", EventQuery(limit=100), export_dir=external_dir)

    assert result["inside_data_dir"] is False
    output = Path(str(result["path"]))
    assert output.exists()
    assert output.is_file()
    assert external_dir in output.parents
    assert result["count"] == 1
