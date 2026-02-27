from __future__ import annotations

import csv
import json
from pathlib import Path

from sentinel.util.time import now_utc

from .repo import EventQuery, StorageRepo


class ExportService:
    def __init__(self, repo: StorageRepo, data_dir: Path) -> None:
        self.repo = repo
        self.data_dir = data_dir
        self.exports_dir = data_dir / "exports"
        self.exports_dir.mkdir(parents=True, exist_ok=True)

    def export(self, export_format: str, query: EventQuery, export_dir: Path | None = None) -> dict[str, object]:
        normalized = export_format.lower()
        if normalized not in {"jsonl", "csv"}:
            msg = f"Unsupported format: {export_format}"
            raise ValueError(msg)

        rows = self.repo.query_events(query)
        timestamp = now_utc().strftime("%Y%m%dT%H%M%SZ")
        output_root = export_dir or self.exports_dir
        output_root.mkdir(parents=True, exist_ok=True)
        output_path = output_root / f"events_{timestamp}.{normalized}"

        if normalized == "jsonl":
            with output_path.open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=True) + "\n")
        else:
            with output_path.open("w", encoding="utf-8", newline="") as f:
                if rows:
                    fieldnames = list(rows[0].keys())
                else:
                    fieldnames = [
                        "id",
                        "created_at",
                        "camera_id",
                        "label",
                        "confidence",
                        "track_id",
                        "zone",
                    ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)

        self.repo.mark_events_exported([row["id"] for row in rows], exported=True)
        try:
            rel = output_path.relative_to(self.data_dir)
            return {"path": str(rel), "count": len(rows), "format": normalized, "inside_data_dir": True}
        except ValueError:
            return {
                "path": str(output_path),
                "count": len(rows),
                "format": normalized,
                "inside_data_dir": False,
            }
