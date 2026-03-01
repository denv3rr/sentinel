from __future__ import annotations

from dataclasses import dataclass, field
import math

from .detect_base import Detection, DetectionChild
from .track_base import TrackedObject, Tracker


@dataclass
class _ChildTrackState:
    child_id: str
    bbox: tuple[int, int, int, int]
    confidence: float
    label: str
    raw_label: str | None
    age: int = 0


@dataclass
class _TrackState:
    track_id: int
    bbox: tuple[int, int, int, int]
    confidence: float
    label: str
    raw_label: str | None
    age: int = 0
    misses: int = 0
    hits: int = 1
    velocity: tuple[float, float] = (0.0, 0.0)
    memory_age: int = 0
    appearance_signature: tuple[float, ...] | None = None
    child_seq: int = 1
    child_states: list[_ChildTrackState] = field(default_factory=list)


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


def _center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _diag(bbox: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = bbox
    w = max(1.0, float(x2 - x1))
    h = max(1.0, float(y2 - y1))
    return math.sqrt((w * w) + (h * h))


def _area(bbox: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = bbox
    return float(max(1, x2 - x1) * max(1, y2 - y1))


def _size_similarity(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    area_a = _area(a)
    area_b = _area(b)
    smaller = min(area_a, area_b)
    larger = max(area_a, area_b)
    if larger <= 0:
        return 0.0
    return max(0.0, min(1.0, smaller / larger))


def _appearance_similarity(a: tuple[float, ...] | None, b: tuple[float, ...] | None) -> float:
    if not a or not b:
        return 0.0
    if len(a) != len(b):
        return 0.0
    dot = sum((x * y) for x, y in zip(a, b, strict=True))
    return max(0.0, min(1.0, dot))


def _blend_appearance(
    prior: tuple[float, ...] | None,
    current: tuple[float, ...] | None,
    ema_alpha: float,
) -> tuple[float, ...] | None:
    if current is None:
        return prior
    if prior is None or len(prior) != len(current):
        return current

    alpha = max(0.0, min(1.0, float(ema_alpha)))
    blended = tuple(((1.0 - alpha) * prev) + (alpha * cur) for prev, cur in zip(prior, current, strict=True))
    norm = math.sqrt(sum((value * value) for value in blended))
    if norm <= 1e-9:
        return current
    return tuple(value / norm for value in blended)


class DefaultIoUTracker(Tracker):
    """IoU + center-distance tracker with revival-oriented object memory."""

    def __init__(
        self,
        iou_threshold: float = 0.35,
        max_age: int = 18,
        memory_max_age: int = 45,
        reid_distance_scale: float = 2.4,
        min_size_similarity: float = 0.2,
        use_appearance: bool = True,
        appearance_weight: float = 0.32,
        min_appearance_similarity: float = 0.14,
        appearance_ema_alpha: float = 0.55,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.memory_max_age = memory_max_age
        self.reid_distance_scale = reid_distance_scale
        self.min_size_similarity = min_size_similarity
        self.use_appearance = use_appearance
        self.appearance_weight = appearance_weight
        self.min_appearance_similarity = min_appearance_similarity
        self.appearance_ema_alpha = appearance_ema_alpha
        self._next_id = 1
        self._tracks: dict[int, _TrackState] = {}
        self._memory: dict[int, _TrackState] = {}

    def _predict_bbox(self, track: _TrackState) -> tuple[int, int, int, int]:
        if track.misses <= 0 and track.memory_age <= 0:
            return track.bbox
        misses = track.misses + track.memory_age
        vx, vy = track.velocity
        dt = min(8, misses)
        x1, y1, x2, y2 = track.bbox
        return (
            int(round(x1 + (vx * dt))),
            int(round(y1 + (vy * dt))),
            int(round(x2 + (vx * dt))),
            int(round(y2 + (vy * dt))),
        )

    def _assignment_score(self, track: _TrackState, detection: Detection) -> float:
        if track.label != detection.label:
            return -1.0
        predicted_bbox = self._predict_bbox(track)
        iou_pred = _iou(predicted_bbox, detection.bbox)
        iou_last = _iou(track.bbox, detection.bbox)
        combined_misses = track.misses + track.memory_age
        dynamic_iou_threshold = max(0.03, self.iou_threshold * (0.80 ** min(8, combined_misses)))

        center_track = _center(predicted_bbox)
        center_det = _center(detection.bbox)
        dist = math.sqrt((center_track[0] - center_det[0]) ** 2 + (center_track[1] - center_det[1]) ** 2)
        norm_dist = dist / max(1.0, max(_diag(track.bbox), _diag(detection.bbox)))
        dist_gate = self.reid_distance_scale + (0.45 * min(8, combined_misses))
        size_sim = _size_similarity(track.bbox, detection.bbox)
        min_size_sim = max(0.05, self.min_size_similarity - (0.03 * min(8, combined_misses)))
        best_iou = max(iou_pred, iou_last * 0.92)
        appearance_sim = _appearance_similarity(track.appearance_signature, detection.appearance_signature)
        has_appearance = track.appearance_signature is not None and detection.appearance_signature is not None

        if best_iou < dynamic_iou_threshold and not (norm_dist <= dist_gate and size_sim >= min_size_sim):
            return -1.0
        if self.use_appearance and has_appearance:
            # Gate weak spatial matches when appearance strongly disagrees.
            if best_iou < 0.20 and appearance_sim < self.min_appearance_similarity:
                return -1.0

        proximity_bonus = max(0.0, 1.0 - (norm_dist / max(1.0, dist_gate)))
        memory_penalty = 0.01 * track.memory_age
        appearance_score = 0.0
        if self.use_appearance and has_appearance:
            appearance_score = self.appearance_weight * appearance_sim
        return (
            best_iou
            + (0.22 * proximity_bonus)
            + (0.14 * size_sim)
            + appearance_score
            - (combined_misses * 0.012)
            - memory_penalty
        )

    def _assign_child_ids(self, track: _TrackState, children: list[DetectionChild]) -> list[DetectionChild]:
        if not children:
            for child_state in track.child_states:
                child_state.age += 1
            track.child_states = [child_state for child_state in track.child_states if child_state.age <= 2]
            return []

        used_state_indexes: set[int] = set()
        resolved_children: list[DetectionChild] = []
        next_child_states: list[_ChildTrackState] = []

        for child in children:
            best_index: int | None = None
            best_iou = 0.0
            for index, child_state in enumerate(track.child_states):
                if index in used_state_indexes:
                    continue
                if child_state.label != child.label:
                    continue
                score = _iou(child_state.bbox, child.bbox)
                if score >= 0.20 and score > best_iou:
                    best_iou = score
                    best_index = index

            if best_index is not None:
                state = track.child_states[best_index]
                used_state_indexes.add(best_index)
                child_id = state.child_id
            else:
                child_id = f"{track.track_id}-c{track.child_seq}"
                track.child_seq += 1

            resolved = DetectionChild(
                bbox=child.bbox,
                confidence=child.confidence,
                label=child.label,
                raw_label=child.raw_label,
                child_id=child_id,
            )
            resolved_children.append(resolved)
            next_child_states.append(
                _ChildTrackState(
                    child_id=child_id,
                    bbox=resolved.bbox,
                    confidence=resolved.confidence,
                    label=resolved.label,
                    raw_label=resolved.raw_label,
                    age=0,
                )
            )

        for index, child_state in enumerate(track.child_states):
            if index in used_state_indexes:
                continue
            aged = _ChildTrackState(
                child_id=child_state.child_id,
                bbox=child_state.bbox,
                confidence=child_state.confidence,
                label=child_state.label,
                raw_label=child_state.raw_label,
                age=child_state.age + 1,
            )
            if aged.age <= 2:
                next_child_states.append(aged)

        track.child_states = next_child_states
        return resolved_children

    def _create_track(self, detection: Detection) -> _TrackState:
        self._purge_memory_by_label(detection.label)
        track_id = self._next_id
        self._next_id += 1
        track = _TrackState(
            track_id=track_id,
            bbox=detection.bbox,
            confidence=detection.confidence,
            label=detection.label,
            raw_label=detection.raw_label,
            appearance_signature=detection.appearance_signature,
        )
        track.child_states = []
        resolved_children = self._assign_child_ids(track, detection.children)
        detection.children = resolved_children
        self._tracks[track_id] = track
        return track

    def _purge_memory_by_label(self, label: str) -> None:
        stale_ids = [track_id for track_id, track in self._memory.items() if track.label == label and track.memory_age > self.memory_max_age]
        for track_id in stale_ids:
            self._memory.pop(track_id, None)

    def _update_track(self, track: _TrackState, detection: Detection) -> None:
        prev_center = _center(track.bbox)
        new_center = _center(detection.bbox)
        track.velocity = (new_center[0] - prev_center[0], new_center[1] - prev_center[1])
        track.bbox = detection.bbox
        track.confidence = detection.confidence
        track.label = detection.label
        track.raw_label = detection.raw_label
        track.appearance_signature = _blend_appearance(
            track.appearance_signature,
            detection.appearance_signature,
            ema_alpha=self.appearance_ema_alpha,
        )
        track.age = 0
        track.misses = 0
        track.memory_age = 0
        track.hits += 1
        detection.children = self._assign_child_ids(track, detection.children)

    def _match_active_track(self, detection: Detection, assigned_tracks: set[int]) -> tuple[int | None, float]:
        best_track_id: int | None = None
        best_score = -1.0
        for track_id, track in self._tracks.items():
            if track_id in assigned_tracks:
                continue
            score = self._assignment_score(track, detection)
            if score > best_score:
                best_score = score
                best_track_id = track_id
        return best_track_id, best_score

    def _revive_from_memory(self, detection: Detection, revived_track_ids: set[int]) -> _TrackState | None:
        best_track_id: int | None = None
        best_score = -1.0
        for track_id, track in self._memory.items():
            if track_id in revived_track_ids:
                continue
            score = self._assignment_score(track, detection)
            if score > best_score:
                best_score = score
                best_track_id = track_id

        if best_track_id is None or best_score < 0.0:
            return None

        track = self._memory.pop(best_track_id)
        track.age = 0
        track.misses = 0
        track.memory_age = 0
        self._tracks[best_track_id] = track
        self._update_track(track, detection)
        revived_track_ids.add(best_track_id)
        return track

    def _age_unmatched_tracks(self, assigned_tracks: set[int]) -> None:
        stale_track_ids: list[int] = []
        for track_id, track in self._tracks.items():
            if track_id in assigned_tracks:
                continue
            track.age += 1
            track.misses += 1
            if track.misses > self.max_age:
                stale_track_ids.append(track_id)

        for track_id in stale_track_ids:
            track = self._tracks.pop(track_id, None)
            if track is None:
                continue
            track.memory_age = 0
            self._memory[track_id] = track

    def _age_memory(self) -> None:
        stale_memory_ids: list[int] = []
        for track_id, track in self._memory.items():
            track.memory_age += 1
            if track.memory_age > self.memory_max_age:
                stale_memory_ids.append(track_id)
        for track_id in stale_memory_ids:
            self._memory.pop(track_id, None)

    def update(self, detections: list[Detection]) -> list[TrackedObject]:
        output: list[TrackedObject] = []
        assigned_tracks: set[int] = set()
        revived_track_ids: set[int] = set()
        unmatched: list[Detection] = []

        for detection in sorted(detections, key=lambda det: det.confidence, reverse=True):
            best_track_id, best_score = self._match_active_track(detection, assigned_tracks)
            if best_track_id is None or best_score < 0.0:
                unmatched.append(detection)
                continue
            track = self._tracks[best_track_id]
            self._update_track(track, detection)
            assigned_tracks.add(track.track_id)
            output.append(
                TrackedObject(
                    track_id=track.track_id,
                    bbox=track.bbox,
                    confidence=track.confidence,
                    label=track.label,
                    raw_label=track.raw_label,
                    children=list(detection.children),
                )
            )

        for detection in unmatched:
            track = self._revive_from_memory(detection, revived_track_ids)
            if track is None:
                track = self._create_track(detection)
            assigned_tracks.add(track.track_id)
            output.append(
                TrackedObject(
                    track_id=track.track_id,
                    bbox=track.bbox,
                    confidence=track.confidence,
                    label=track.label,
                    raw_label=track.raw_label,
                    children=list(detection.children),
                )
            )

        self._age_unmatched_tracks(assigned_tracks)
        self._age_memory()

        output.sort(key=lambda item: item.track_id)
        return output
