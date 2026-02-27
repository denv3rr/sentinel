from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from sentinel.util.time import now_utc_iso

from .db import Database


@dataclass
class EventQuery:
    start: str | None = None
    end: str | None = None
    camera_id: str | None = None
    label: str | None = None
    min_confidence: float | None = None
    zone: str | None = None
    reviewed: bool | None = None
    exported: bool | None = None
    search: str | None = None
    limit: int = 200
    offset: int = 0
    sort: str = "desc"


class StorageRepo:
    def __init__(self, db: Database) -> None:
        self.db = db
        self._event_insert_sql = """
            INSERT INTO events (
                id, created_at, local_time, monotonic_ns, camera_id, event_type, label,
                confidence, track_id, zone, motion, thumbnail_path, clip_path,
                reviewed, exported, search_text, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

    def upsert_camera(self, camera: dict[str, Any]) -> None:
        now_iso = now_utc_iso()
        payload = (
            camera["id"],
            camera["name"],
            camera["kind"],
            camera["source"],
            int(camera.get("enabled", True)),
            json.dumps(camera, ensure_ascii=True),
            now_iso,
            now_iso,
        )
        self.db.execute(
            """
            INSERT INTO cameras (id, name, kind, source, enabled, settings_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
              name=excluded.name,
              kind=excluded.kind,
              source=excluded.source,
              enabled=excluded.enabled,
              settings_json=excluded.settings_json,
              updated_at=excluded.updated_at
            """,
            payload,
        )

    def list_cameras(self) -> list[dict[str, Any]]:
        rows = self.db.query("SELECT settings_json FROM cameras ORDER BY created_at ASC")
        return [json.loads(row["settings_json"]) for row in rows]

    def get_camera(self, camera_id: str) -> dict[str, Any] | None:
        row = self.db.query_one("SELECT settings_json FROM cameras WHERE id = ?", (camera_id,))
        if not row:
            return None
        return json.loads(row["settings_json"])

    def delete_camera(self, camera_id: str) -> None:
        self.db.execute("DELETE FROM cameras WHERE id = ?", (camera_id,))

    def insert_event(self, event: dict[str, Any]) -> None:
        self.db.execute(self._event_insert_sql, self._event_params(event))

    def _event_params(self, event: dict[str, Any]) -> tuple[Any, ...]:
        return (
            event["id"],
            event["created_at"],
            event["local_time"],
            event["monotonic_ns"],
            event["camera_id"],
            event.get("event_type", "motion_detection"),
            event["label"],
            float(event["confidence"]),
            event.get("track_id"),
            event.get("zone"),
            event.get("motion"),
            event.get("thumbnail_path"),
            event.get("clip_path"),
            int(event.get("reviewed", False)),
            int(event.get("exported", False)),
            event.get("search_text", ""),
            json.dumps(event.get("metadata", {}), ensure_ascii=True),
        )

    def insert_events(self, events: list[dict[str, Any]]) -> None:
        if not events:
            return
        rows = [self._event_params(event) for event in events]
        self.db.executemany(self._event_insert_sql, rows)

    def query_events(self, query: EventQuery) -> list[dict[str, Any]]:
        clauses: list[str] = ["1=1"]
        params: list[Any] = []

        if query.start:
            clauses.append("created_at >= ?")
            params.append(query.start)
        if query.end:
            clauses.append("created_at <= ?")
            params.append(query.end)
        if query.camera_id:
            clauses.append("camera_id = ?")
            params.append(query.camera_id)
        if query.label:
            clauses.append("label = ?")
            params.append(query.label)
        if query.min_confidence is not None:
            clauses.append("confidence >= ?")
            params.append(query.min_confidence)
        if query.zone:
            clauses.append("zone = ?")
            params.append(query.zone)
        if query.reviewed is not None:
            clauses.append("reviewed = ?")
            params.append(int(query.reviewed))
        if query.exported is not None:
            clauses.append("exported = ?")
            params.append(int(query.exported))
        if query.search:
            clauses.append("search_text LIKE ?")
            params.append(f"%{query.search.strip()}%")

        order = "DESC" if query.sort.lower() == "desc" else "ASC"
        limit = max(1, min(query.limit, 2000))
        offset = max(0, query.offset)
        params.extend([limit, offset])

        sql = f"""
            SELECT * FROM events
            WHERE {' AND '.join(clauses)}
            ORDER BY created_at {order}
            LIMIT ? OFFSET ?
        """

        rows = self.db.query(sql, tuple(params))
        out: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["reviewed"] = bool(item["reviewed"])
            item["exported"] = bool(item["exported"])
            item["metadata"] = json.loads(item["metadata_json"]) if item.get("metadata_json") else {}
            item.pop("metadata_json", None)
            out.append(item)
        return out

    def count_events(self, query: EventQuery) -> int:
        clauses: list[str] = ["1=1"]
        params: list[Any] = []

        if query.start:
            clauses.append("created_at >= ?")
            params.append(query.start)
        if query.end:
            clauses.append("created_at <= ?")
            params.append(query.end)
        if query.camera_id:
            clauses.append("camera_id = ?")
            params.append(query.camera_id)
        if query.label:
            clauses.append("label = ?")
            params.append(query.label)
        if query.min_confidence is not None:
            clauses.append("confidence >= ?")
            params.append(query.min_confidence)
        if query.zone:
            clauses.append("zone = ?")
            params.append(query.zone)
        if query.reviewed is not None:
            clauses.append("reviewed = ?")
            params.append(int(query.reviewed))
        if query.exported is not None:
            clauses.append("exported = ?")
            params.append(int(query.exported))
        if query.search:
            clauses.append("search_text LIKE ?")
            params.append(f"%{query.search.strip()}%")

        row = self.db.query_one(
            f"SELECT COUNT(*) AS count FROM events WHERE {' AND '.join(clauses)}",
            tuple(params),
        )
        return int(row["count"]) if row else 0

    def mark_event_reviewed(self, event_id: str, reviewed: bool = True) -> None:
        self.db.execute("UPDATE events SET reviewed = ? WHERE id = ?", (int(reviewed), event_id))

    def mark_events_exported(self, event_ids: list[str], exported: bool = True) -> None:
        if not event_ids:
            return
        placeholders = ",".join(["?"] * len(event_ids))
        values: list[Any] = [int(exported), *event_ids]
        self.db.execute(f"UPDATE events SET exported = ? WHERE id IN ({placeholders})", tuple(values))

    def list_events_before(self, cutoff_iso: str, limit: int = 10_000) -> list[dict[str, Any]]:
        rows = self.db.query(
            """
            SELECT id, thumbnail_path, clip_path, created_at
            FROM events
            WHERE created_at < ?
            ORDER BY created_at ASC
            LIMIT ?
            """,
            (cutoff_iso, limit),
        )
        return [dict(r) for r in rows]

    def list_oldest_events(self, limit: int = 10_000) -> list[dict[str, Any]]:
        rows = self.db.query(
            """
            SELECT id, thumbnail_path, clip_path, created_at
            FROM events
            ORDER BY created_at ASC
            LIMIT ?
            """,
            (limit,),
        )
        return [dict(r) for r in rows]

    def delete_events(self, event_ids: list[str]) -> None:
        if not event_ids:
            return
        placeholders = ",".join(["?"] * len(event_ids))
        self.db.execute(f"DELETE FROM events WHERE id IN ({placeholders})", tuple(event_ids))

    def append_settings_history(self, key: str, value: dict[str, Any]) -> None:
        self.db.execute(
            "INSERT INTO settings_history (created_at, key, value_json) VALUES (?, ?, ?)",
            (now_utc_iso(), key, json.dumps(value, ensure_ascii=True)),
        )
