from __future__ import annotations

import datetime as dt
from pathlib import Path

from sentinel.util.time import now_utc

from .repo import StorageRepo


def _dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


class RetentionService:
    def __init__(self, repo: StorageRepo, data_dir: Path) -> None:
        self.repo = repo
        self.data_dir = data_dir

    def enforce(self, days: int, max_gb: float | None) -> dict[str, int | float]:
        deleted_age = self._enforce_age(days)
        deleted_size = self._enforce_size(max_gb)
        media_gb = _dir_size_bytes(self.data_dir / "media") / (1024**3)
        return {
            "deleted_by_age": deleted_age,
            "deleted_by_size": deleted_size,
            "media_gb": round(media_gb, 3),
        }

    def _enforce_age(self, days: int) -> int:
        cutoff = now_utc() - dt.timedelta(days=max(1, days))
        rows = self.repo.list_events_before(cutoff.isoformat())
        if not rows:
            return 0
        event_ids: list[str] = []
        for row in rows:
            self._delete_media_path(row.get("thumbnail_path"))
            self._delete_media_path(row.get("clip_path"))
            event_ids.append(str(row["id"]))
        self.repo.delete_events(event_ids)
        return len(event_ids)

    def _enforce_size(self, max_gb: float | None) -> int:
        if max_gb is None or max_gb <= 0:
            return 0

        target_bytes = int(max_gb * (1024**3))
        media_dir = self.data_dir / "media"
        current = _dir_size_bytes(media_dir)
        if current <= target_bytes:
            return 0

        deleted = 0
        batch = self.repo.list_oldest_events(limit=20_000)
        event_ids: list[str] = []
        for row in batch:
            if current <= target_bytes:
                break
            for key in ("thumbnail_path", "clip_path"):
                rel = row.get(key)
                if rel:
                    path = self.data_dir / str(rel)
                    if path.exists() and path.is_file():
                        size = path.stat().st_size
                        path.unlink(missing_ok=True)
                        current = max(0, current - size)
            event_ids.append(str(row["id"]))
            deleted += 1

        self.repo.delete_events(event_ids)
        return deleted

    def _delete_media_path(self, rel_path: str | None) -> None:
        if not rel_path:
            return
        path = self.data_dir / rel_path
        if path.exists() and path.is_file():
            path.unlink(missing_ok=True)