from __future__ import annotations

from sentinel.vision.detect_base import Detection, DetectionChild
from sentinel.vision.tracker_default import DefaultIoUTracker


def test_tracker_reuses_id_after_brief_detection_gap() -> None:
    tracker = DefaultIoUTracker(iou_threshold=0.35, max_age=10)

    first = tracker.update([Detection(bbox=(100, 100, 180, 240), confidence=0.9, label="person")])
    assert len(first) == 1
    first_id = first[0].track_id

    # Two missed frames.
    assert tracker.update([]) == []
    assert tracker.update([]) == []

    # Return with shifted box that would normally have weak IoU.
    returned = tracker.update([Detection(bbox=(135, 110, 215, 250), confidence=0.88, label="person")])
    assert len(returned) == 1
    assert returned[0].track_id == first_id


def test_tracker_reuses_id_after_gap_with_low_overlap_and_fast_shift() -> None:
    tracker = DefaultIoUTracker(iou_threshold=0.35, max_age=8, memory_max_age=20)

    initial = tracker.update([Detection(bbox=(40, 40, 120, 220), confidence=0.94, label="person")])
    track_id = initial[0].track_id

    # Simulate detector misses while the target moves rapidly.
    for _ in range(4):
        assert tracker.update([]) == []

    # Return appears with near-zero overlap; should still re-associate.
    returned = tracker.update([Detection(bbox=(150, 60, 230, 240), confidence=0.9, label="person")])
    assert len(returned) == 1
    assert returned[0].track_id == track_id


def test_tracker_revives_id_from_inactive_memory_pool() -> None:
    tracker = DefaultIoUTracker(
        iou_threshold=0.35,
        max_age=2,
        memory_max_age=12,
        reid_distance_scale=2.8,
    )
    first = tracker.update([Detection(bbox=(100, 100, 180, 220), confidence=0.91, label="person")])
    first_id = first[0].track_id

    # Force active-track expiration into inactive memory.
    assert tracker.update([]) == []
    assert tracker.update([]) == []
    assert tracker.update([]) == []

    revived = tracker.update([Detection(bbox=(122, 108, 202, 228), confidence=0.89, label="person")])
    assert len(revived) == 1
    assert revived[0].track_id == first_id


def test_tracker_memory_window_expires_old_identity() -> None:
    tracker = DefaultIoUTracker(iou_threshold=0.35, max_age=1, memory_max_age=2)
    first = tracker.update([Detection(bbox=(100, 100, 180, 220), confidence=0.91, label="person")])
    first_id = first[0].track_id

    # Move the track to inactive memory, then age it out.
    assert tracker.update([]) == []
    assert tracker.update([]) == []
    assert tracker.update([]) == []
    assert tracker.update([]) == []

    returned = tracker.update([Detection(bbox=(108, 102, 188, 222), confidence=0.9, label="person")])
    assert len(returned) == 1
    assert returned[0].track_id != first_id


def test_tracker_appearance_prevents_wrong_revival() -> None:
    appearance_a = tuple([1.0] + [0.0] * 23)
    appearance_b = tuple([0.0, 1.0] + [0.0] * 22)
    tracker = DefaultIoUTracker(
        iou_threshold=0.35,
        max_age=1,
        memory_max_age=20,
        reid_distance_scale=3.0,
        use_appearance=True,
        min_appearance_similarity=0.6,
    )
    first = tracker.update(
        [
            Detection(
                bbox=(50, 50, 130, 230),
                confidence=0.92,
                label="person",
                appearance_signature=appearance_a,
            )
        ]
    )
    first_id = first[0].track_id

    assert tracker.update([]) == []
    assert tracker.update([]) == []
    candidate = tracker.update(
        [
            Detection(
                bbox=(158, 60, 238, 240),
                confidence=0.9,
                label="person",
                appearance_signature=appearance_b,
            )
        ]
    )
    assert len(candidate) == 1
    assert candidate[0].track_id != first_id


def test_tracker_keeps_child_ids_stable_for_parent_track() -> None:
    tracker = DefaultIoUTracker(iou_threshold=0.35, max_age=10)

    det1 = Detection(
        bbox=(40, 40, 180, 240),
        confidence=0.93,
        label="person",
        children=[DetectionChild(bbox=(80, 120, 120, 170), confidence=0.6, label="limb")],
    )
    out1 = tracker.update([det1])
    child_id_1 = out1[0].children[0].child_id
    assert child_id_1 is not None

    det2 = Detection(
        bbox=(45, 45, 185, 245),
        confidence=0.92,
        label="person",
        children=[DetectionChild(bbox=(84, 122, 124, 174), confidence=0.62, label="limb")],
    )
    out2 = tracker.update([det2])
    assert out2[0].children[0].child_id == child_id_1

    det3 = Detection(
        bbox=(50, 50, 190, 250),
        confidence=0.91,
        label="person",
        children=[
            DetectionChild(bbox=(88, 124, 128, 176), confidence=0.61, label="limb"),
            DetectionChild(bbox=(130, 150, 165, 205), confidence=0.58, label="limb"),
        ],
    )
    out3 = tracker.update([det3])
    child_ids = {child.child_id for child in out3[0].children}
    assert child_id_1 in child_ids
    assert len(child_ids) == 2
