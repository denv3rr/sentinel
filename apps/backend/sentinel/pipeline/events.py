from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from sentinel.vision.track_base import TrackedObject
from sentinel.vision.zones import detect_zone_hits


@dataclass
class EventEngine:
    global_thresholds: dict[str, float]

    def __post_init__(self) -> None:
        self._last_event_ns: dict[tuple[str, int, str, str], int] = {}

    def evaluate(
        self,
        camera_cfg: dict[str, object],
        tracks: list[TrackedObject],
        motion_score: float,
        wall_time_iso: str,
        local_time_iso: str,
        monotonic_ns: int,
    ) -> list[dict[str, object]]:
        motion_threshold = float(camera_cfg.get("motion_threshold", 0.012))
        if motion_score < motion_threshold:
            return []

        camera_id = str(camera_cfg["id"])
        allowed_labels = set(camera_cfg.get("labels", ["person", "animal", "vehicle", "unknown"]))
        camera_thresholds = dict(camera_cfg.get("min_confidence", {}))
        cooldown_seconds = int(camera_cfg.get("cooldown_seconds", 8))
        cooldown_ns = max(1, cooldown_seconds) * 1_000_000_000
        zones = camera_cfg.get("zones", [])
        person_only = bool(camera_cfg.get("person_only_alerts", False))

        events: list[dict[str, object]] = []

        for track in tracks:
            if person_only and track.label != "person":
                continue
            if track.label not in allowed_labels:
                continue

            threshold = float(camera_thresholds.get(track.label, self.global_thresholds.get(track.label, 0.35)))
            if track.confidence < threshold:
                continue

            zone_hits = detect_zone_hits(track.bbox, zones if isinstance(zones, list) else [])
            if zone_hits.ignore_zones:
                continue

            include_zone = ""
            include_present = any(str(z.get("mode", "include")) == "include" for z in zones) if isinstance(zones, list) else False
            if include_present:
                if not zone_hits.include_zones:
                    continue
                include_zone = zone_hits.include_zones[0]
            elif zone_hits.include_zones:
                include_zone = zone_hits.include_zones[0]

            dedupe_key = (camera_id, track.track_id, track.label, include_zone)
            last_ns = self._last_event_ns.get(dedupe_key)
            if last_ns is not None and (monotonic_ns - last_ns) < cooldown_ns:
                continue
            self._last_event_ns[dedupe_key] = monotonic_ns

            raw_label = track.raw_label or track.label
            events.append(
                {
                    "id": f"evt-{uuid4().hex}",
                    "created_at": wall_time_iso,
                    "local_time": local_time_iso,
                    "monotonic_ns": monotonic_ns,
                    "camera_id": camera_id,
                    "event_type": "motion_detection",
                    "label": track.label,
                    "confidence": round(float(track.confidence), 4),
                    "track_id": int(track.track_id),
                    "zone": include_zone or None,
                    "motion": round(float(motion_score), 6),
                    "reviewed": False,
                    "exported": False,
                    "search_text": f"{camera_cfg.get('name', camera_id)} {track.label} {raw_label} {include_zone}".strip(),
                    "metadata": {
                        "bbox": list(track.bbox),
                        "raw_label": raw_label,
                        "threshold": threshold,
                    },
                }
            )

        return events