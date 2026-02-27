from __future__ import annotations

from dataclasses import dataclass

from .detect_base import Detection
from .track_base import TrackedObject, Tracker


@dataclass
class _TrackState:
    track_id: int
    bbox: tuple[int, int, int, int]
    confidence: float
    label: str
    raw_label: str | None
    age: int


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class DefaultIoUTracker(Tracker):
    """Simple ByteTrack-inspired IoU matcher for stable IDs."""

    def __init__(self, iou_threshold: float = 0.35, max_age: int = 18) -> None:
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self._next_id = 1
        self._tracks: dict[int, _TrackState] = {}

    def update(self, detections: list[Detection]) -> list[TrackedObject]:
        assigned_tracks: set[int] = set()
        output: list[TrackedObject] = []

        for detection in detections:
            best_track_id: int | None = None
            best_score = 0.0
            for track_id, track in self._tracks.items():
                if track.label != detection.label:
                    continue
                score = _iou(track.bbox, detection.bbox)
                if score > best_score and score >= self.iou_threshold:
                    best_score = score
                    best_track_id = track_id

            if best_track_id is None:
                best_track_id = self._next_id
                self._next_id += 1

            self._tracks[best_track_id] = _TrackState(
                track_id=best_track_id,
                bbox=detection.bbox,
                confidence=detection.confidence,
                label=detection.label,
                raw_label=detection.raw_label,
                age=0,
            )
            assigned_tracks.add(best_track_id)

        stale: list[int] = []
        for track_id, track in self._tracks.items():
            if track_id in assigned_tracks:
                continue
            track.age += 1
            if track.age > self.max_age:
                stale.append(track_id)

        for track_id in stale:
            self._tracks.pop(track_id, None)

        for track_id in sorted(assigned_tracks):
            track = self._tracks[track_id]
            output.append(
                TrackedObject(
                    track_id=track.track_id,
                    bbox=track.bbox,
                    confidence=track.confidence,
                    label=track.label,
                    raw_label=track.raw_label,
                )
            )

        return output