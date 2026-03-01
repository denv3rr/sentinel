from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .detect_base import Detection, DetectionChild, Detector

CATEGORY_MAP = {
    "person": "person",
    "bicycle": "vehicle",
    "car": "vehicle",
    "motorbike": "vehicle",
    "motorcycle": "vehicle",
    "bus": "vehicle",
    "truck": "vehicle",
    "train": "vehicle",
    "boat": "vehicle",
    "cat": "animal",
    "dog": "animal",
    "bird": "animal",
    "horse": "animal",
    "sheep": "animal",
    "cow": "animal",
    "bear": "animal",
    "zebra": "animal",
    "giraffe": "animal",
    "hand": "limb",
    "arm": "limb",
    "forearm": "limb",
    "wrist": "limb",
    "elbow": "limb",
    "leg": "limb",
    "foot": "limb",
}

_MOTION_RAW_LABELS = {"motion", "limb"}


@dataclass(frozen=True)
class DetectorProfile:
    name: str
    confidence: float
    image_size: int
    max_boxes: int
    motion_threshold: int
    min_motion_area: int
    smoothing_alpha: float
    hysteresis_frames: int
    merge_iou_threshold: float
    smoothing_iou_threshold: float
    emit_motion_labels: bool


DETECTOR_PROFILES: dict[str, DetectorProfile] = {
    "balanced": DetectorProfile(
        name="balanced",
        confidence=0.25,
        image_size=960,
        max_boxes=80,
        motion_threshold=28,
        min_motion_area=900,
        smoothing_alpha=0.42,
        hysteresis_frames=2,
        merge_iou_threshold=0.26,
        smoothing_iou_threshold=0.25,
        emit_motion_labels=False,
    ),
    "fast_motion": DetectorProfile(
        name="fast_motion",
        confidence=0.16,
        image_size=800,
        max_boxes=96,
        motion_threshold=19,
        min_motion_area=260,
        smoothing_alpha=0.52,
        hysteresis_frames=4,
        merge_iou_threshold=0.20,
        smoothing_iou_threshold=0.18,
        emit_motion_labels=True,
    ),
    "small_object": DetectorProfile(
        name="small_object",
        confidence=0.14,
        image_size=1280,
        max_boxes=140,
        motion_threshold=24,
        min_motion_area=150,
        smoothing_alpha=0.36,
        hysteresis_frames=2,
        merge_iou_threshold=0.24,
        smoothing_iou_threshold=0.20,
        emit_motion_labels=True,
    ),
    "capture_card_reforger": DetectorProfile(
        name="capture_card_reforger",
        confidence=0.18,
        image_size=960,
        max_boxes=120,
        motion_threshold=30,
        min_motion_area=220,
        smoothing_alpha=0.50,
        hysteresis_frames=5,
        merge_iou_threshold=0.18,
        smoothing_iou_threshold=0.16,
        emit_motion_labels=True,
    ),
}


def _resolve_profile(name: str) -> DetectorProfile:
    normalized = (name or "balanced").strip().lower()
    return DETECTOR_PROFILES.get(normalized, DETECTOR_PROFILES["balanced"])


@dataclass
class DetectorSettings:
    model_name: str = "yolov8n.pt"
    profile_name: str = "balanced"
    confidence: float = 0.25
    image_size: int = 960
    max_boxes: int = 80
    motion_threshold: int = 28
    min_motion_area: int = 900
    smoothing_alpha: float = 0.42
    hysteresis_frames: int = 2
    merge_iou_threshold: float = 0.26
    smoothing_iou_threshold: float = 0.25
    emit_motion_labels: bool = False
    child_attach_mode: str = "balanced"

    @classmethod
    def from_profile(
        cls,
        model_name: str,
        confidence: float,
        profile_name: str = "balanced",
        image_size: int | None = None,
        max_boxes: int | None = None,
        motion_threshold: int | None = None,
        min_motion_area: int | None = None,
        smoothing_alpha: float | None = None,
        hysteresis_frames: int | None = None,
        child_attach_mode: str | None = None,
    ) -> "DetectorSettings":
        profile = _resolve_profile(profile_name)
        return cls(
            model_name=model_name,
            profile_name=profile.name,
            confidence=float(confidence if confidence is not None else profile.confidence),
            image_size=int(image_size if image_size is not None else profile.image_size),
            max_boxes=max(1, int(max_boxes if max_boxes is not None else profile.max_boxes)),
            motion_threshold=max(1, int(motion_threshold if motion_threshold is not None else profile.motion_threshold)),
            min_motion_area=max(1, int(min_motion_area if min_motion_area is not None else profile.min_motion_area)),
            smoothing_alpha=float(smoothing_alpha if smoothing_alpha is not None else profile.smoothing_alpha),
            hysteresis_frames=max(0, int(hysteresis_frames if hysteresis_frames is not None else profile.hysteresis_frames)),
            merge_iou_threshold=float(profile.merge_iou_threshold),
            smoothing_iou_threshold=float(profile.smoothing_iou_threshold),
            emit_motion_labels=bool(profile.emit_motion_labels),
            child_attach_mode=str(child_attach_mode or "balanced").strip().lower(),
        )


def _sanitize_bbox(bbox: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    max_x = max(0, width - 1)
    max_y = max(0, height - 1)
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(max_x, int(x1)))
    y1 = max(0, min(max_y, int(y1)))
    x2 = max(0, min(width, int(x2)))
    y2 = max(0, min(height, int(y2)))
    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)
    return x1, y1, x2, y2


def _sanitize_children(children: list[DetectionChild], width: int, height: int) -> list[DetectionChild]:
    sanitized: list[DetectionChild] = []
    for child in children:
        sanitized.append(
            DetectionChild(
                bbox=_sanitize_bbox(child.bbox, width, height),
                confidence=float(child.confidence),
                label=str(child.label or "unknown"),
                raw_label=child.raw_label,
                child_id=child.child_id,
            )
        )
    return sanitized


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter_area = float((inter_x2 - inter_x1) * (inter_y2 - inter_y1))
    area_a = float(max(1, (ax2 - ax1) * (ay2 - ay1)))
    area_b = float(max(1, (bx2 - bx1) * (by2 - by1)))
    return inter_area / (area_a + area_b - inter_area)


def _containment_ratio(inner: tuple[int, int, int, int], outer: tuple[int, int, int, int]) -> float:
    ix1, iy1, ix2, iy2 = inner
    ox1, oy1, ox2, oy2 = outer
    inter_x1 = max(ix1, ox1)
    inter_y1 = max(iy1, oy1)
    inter_x2 = min(ix2, ox2)
    inter_y2 = min(iy2, oy2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter_area = float((inter_x2 - inter_x1) * (inter_y2 - inter_y1))
    inner_area = float(max(1, (ix2 - ix1) * (iy2 - iy1)))
    return inter_area / inner_area


@dataclass
class _TemporalState:
    bbox: tuple[int, int, int, int]
    confidence: float
    label: str
    raw_label: str | None
    ttl: int
    children: list[DetectionChild] = field(default_factory=list)


class BaseDetector(Detector, ABC):
    def __init__(self, settings: DetectorSettings) -> None:
        self.settings = settings
        self._states: list[_TemporalState] = []

    @abstractmethod
    def detect_raw(self, frame: np.ndarray) -> list[Detection]:
        raise NotImplementedError

    def detect(self, frame: np.ndarray) -> list[Detection]:
        raw = self.detect_raw(frame)
        return self._finalize(raw, frame)

    def _finalize(self, detections: list[Detection], frame: np.ndarray) -> list[Detection]:
        height, width = frame.shape[:2]
        normalized: list[Detection] = []
        for detection in detections:
            bbox = _sanitize_bbox(detection.bbox, width, height)
            if detection.confidence < self.settings.confidence:
                continue
            normalized.append(
                Detection(
                    bbox=bbox,
                    confidence=float(detection.confidence),
                    label=str(detection.label or "unknown"),
                    raw_label=detection.raw_label,
                    children=_sanitize_children(detection.children, width, height),
                )
            )

        normalized.sort(key=lambda item: (item.confidence, item.bbox), reverse=True)
        if not normalized and not self._states:
            return []

        alpha = float(max(0.0, min(1.0, self.settings.smoothing_alpha)))
        use_temporal = alpha > 0.0 or self.settings.hysteresis_frames > 0
        if not use_temporal:
            return normalized[: self.settings.max_boxes]

        previous_states = self._states
        used_previous: set[int] = set()
        next_states: list[_TemporalState] = []

        for detection in normalized:
            best_index: int | None = None
            best_iou = 0.0
            for index, state in enumerate(previous_states):
                if index in used_previous:
                    continue
                if state.label != detection.label:
                    continue
                iou_score = _iou(detection.bbox, state.bbox)
                if iou_score >= self.settings.smoothing_iou_threshold and iou_score > best_iou:
                    best_iou = iou_score
                    best_index = index

            if best_index is None:
                next_states.append(
                    _TemporalState(
                        bbox=detection.bbox,
                        confidence=detection.confidence,
                        label=detection.label,
                        raw_label=detection.raw_label,
                        ttl=self.settings.hysteresis_frames,
                        children=list(detection.children),
                    )
                )
                continue

            used_previous.add(best_index)
            prior = previous_states[best_index]
            if alpha <= 0.0:
                merged_bbox = detection.bbox
                merged_conf = detection.confidence
            else:
                merged_bbox = (
                    int(round((1.0 - alpha) * prior.bbox[0] + alpha * detection.bbox[0])),
                    int(round((1.0 - alpha) * prior.bbox[1] + alpha * detection.bbox[1])),
                    int(round((1.0 - alpha) * prior.bbox[2] + alpha * detection.bbox[2])),
                    int(round((1.0 - alpha) * prior.bbox[3] + alpha * detection.bbox[3])),
                )
                merged_conf = (1.0 - alpha) * prior.confidence + alpha * detection.confidence
            merged_children = list(detection.children) if detection.children else list(prior.children)
            next_states.append(
                _TemporalState(
                    bbox=merged_bbox,
                    confidence=float(merged_conf),
                    label=detection.label,
                    raw_label=detection.raw_label or prior.raw_label,
                    ttl=self.settings.hysteresis_frames,
                    children=merged_children,
                )
            )

        if self.settings.hysteresis_frames > 0:
            for index, state in enumerate(previous_states):
                if index in used_previous:
                    continue
                ttl = state.ttl - 1
                if ttl < 0:
                    continue
                carry_confidence = state.confidence * 0.92
                if carry_confidence < max(0.05, self.settings.confidence * 0.5):
                    continue
                next_states.append(
                    _TemporalState(
                        bbox=state.bbox,
                        confidence=float(carry_confidence),
                        label=state.label,
                        raw_label=state.raw_label,
                        ttl=ttl,
                        children=list(state.children),
                    )
                )

        next_states.sort(key=lambda state: (state.confidence, state.bbox), reverse=True)
        self._states = next_states[: self.settings.max_boxes]
        return [
            Detection(
                bbox=state.bbox,
                confidence=round(float(state.confidence), 4),
                label=state.label,
                raw_label=state.raw_label,
                children=list(state.children),
            )
            for state in self._states
        ]


def _load_yolo_model(model_name: str) -> Any | None:
    try:
        from ultralytics import YOLO  # type: ignore

        return YOLO(model_name)
    except Exception:
        return None


def _load_manifest_payload(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception:
            return None
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    return None


def _resolve_model_name(model_name: str, model_registry_path: str | None = None) -> str:
    candidate = Path(model_name).expanduser()
    if candidate.exists() and candidate.suffix.lower() in {".json", ".yaml", ".yml"}:
        payload = _load_manifest_payload(candidate.resolve())
        if payload and isinstance(payload.get("weights_path"), str):
            return str(payload["weights_path"])

    registry_hint = model_registry_path or os.environ.get("SENTINEL_MODEL_REGISTRY", "artifacts/ml/registry.json")
    registry_path = Path(registry_hint).expanduser().resolve()
    if not registry_path.exists():
        return model_name
    try:
        registry_payload = json.loads(registry_path.read_text(encoding="utf-8"))
    except Exception:
        return model_name
    if not isinstance(registry_payload, dict):
        return model_name
    aliases = registry_payload.get("aliases")
    if not isinstance(aliases, dict):
        return model_name
    manifest_path_raw = aliases.get(model_name)
    if not isinstance(manifest_path_raw, str):
        return model_name
    manifest_path = Path(manifest_path_raw).expanduser().resolve()
    payload = _load_manifest_payload(manifest_path)
    if payload and isinstance(payload.get("weights_path"), str):
        return str(payload["weights_path"])
    return model_name


class UltralyticsDetector(BaseDetector):
    def __init__(self, settings: DetectorSettings, model: Any | None = None) -> None:
        super().__init__(settings)
        self._model = model if model is not None else _load_yolo_model(settings.model_name)

    @property
    def model_loaded(self) -> bool:
        return self._model is not None

    def detect_raw(self, frame: np.ndarray) -> list[Detection]:
        if self._model is None:
            return []

        results = self._model.predict(
            frame,
            conf=self.settings.confidence,
            imgsz=self.settings.image_size,
            max_det=self.settings.max_boxes,
            verbose=False,
        )
        detections: list[Detection] = []
        for result in results:
            names = getattr(result, "names", {})
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls.item())
                raw_label = str(names.get(cls_id, "unknown")).lower()
                mapped = CATEGORY_MAP.get(raw_label, "unknown")
                if mapped in _MOTION_RAW_LABELS and not self.settings.emit_motion_labels:
                    mapped = "unknown"
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(
                    Detection(
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        confidence=float(box.conf.item()),
                        label=mapped,
                        raw_label=raw_label,
                    )
                )
        return detections


class MotionAssistDetector(BaseDetector):
    def __init__(self, settings: DetectorSettings) -> None:
        super().__init__(settings)
        self._prev_gray: np.ndarray | None = None

    def detect_raw(self, frame: np.ndarray) -> list[Detection]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        if self._prev_gray is None:
            self._prev_gray = gray
            return []

        delta = cv2.absdiff(self._prev_gray, gray)
        _, threshold = cv2.threshold(delta, self.settings.motion_threshold, 255, cv2.THRESH_BINARY)
        threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8), iterations=2)
        threshold = cv2.dilate(threshold, None, iterations=2)
        contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self._prev_gray = gray

        frame_area = float(max(1, frame.shape[0] * frame.shape[1]))
        mean_motion = float(np.mean(delta)) / 255.0
        detections: list[Detection] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < float(self.settings.min_motion_area):
                continue
            x, y, w, h = cv2.boundingRect(contour)
            raw_label = "limb" if self._looks_like_limb(w, h, area) else "motion"
            label = raw_label if self.settings.emit_motion_labels else "unknown"
            area_ratio = min(1.0, area / frame_area)
            confidence = min(0.95, max(self.settings.confidence, 0.22 + (area_ratio * 2.5) + (mean_motion * 0.75)))
            detections.append(
                Detection(
                    bbox=(x, y, x + w, y + h),
                    confidence=confidence,
                    label=label,
                    raw_label=raw_label,
                )
            )
        return detections

    @staticmethod
    def _looks_like_limb(width: int, height: int, area: float) -> bool:
        if width <= 0 or height <= 0:
            return False
        ratio = float(width) / float(height)
        elongated = ratio > 1.8 or ratio < 0.55
        compact = area < 4500.0
        return elongated and compact


class HybridDetector(BaseDetector):
    def __init__(
        self,
        settings: DetectorSettings,
        yolo_detector: UltralyticsDetector | None = None,
        motion_detector: MotionAssistDetector | None = None,
    ) -> None:
        super().__init__(settings)
        self._yolo = yolo_detector if yolo_detector is not None else UltralyticsDetector(settings=settings)
        self._motion = motion_detector if motion_detector is not None else MotionAssistDetector(settings=settings)

    @property
    def model_loaded(self) -> bool:
        return self._yolo.model_loaded

    def detect_raw(self, frame: np.ndarray) -> list[Detection]:
        yolo_detections = self._yolo.detect_raw(frame)
        motion_detections = self._motion.detect_raw(frame)
        if not yolo_detections:
            return motion_detections

        merged: list[Detection] = [
            Detection(
                bbox=detection.bbox,
                confidence=detection.confidence,
                label=detection.label,
                raw_label=detection.raw_label,
                children=list(detection.children),
            )
            for detection in yolo_detections
        ]
        for motion_detection in motion_detections:
            best_overlap = 0.0
            best_containment = 0.0
            best_parent_index: int | None = None
            for index, parent in enumerate(merged):
                if parent.label in _MOTION_RAW_LABELS:
                    continue
                overlap = _iou(motion_detection.bbox, parent.bbox)
                containment = _containment_ratio(motion_detection.bbox, parent.bbox)
                if overlap > best_overlap or (overlap == best_overlap and containment > best_containment):
                    best_overlap = overlap
                    best_containment = containment
                    best_parent_index = index

            should_attach = best_parent_index is not None and self._should_attach(best_overlap, best_containment)
            if not should_attach:
                merged.append(motion_detection)
                continue

            parent = merged[best_parent_index]
            parent.children.append(
                DetectionChild(
                    bbox=motion_detection.bbox,
                    confidence=motion_detection.confidence,
                    label=motion_detection.label,
                    raw_label=motion_detection.raw_label,
                )
            )
        return merged

    def _should_attach(self, best_overlap: float, best_containment: float) -> bool:
        mode = self.settings.child_attach_mode
        if mode == "off":
            return False
        if mode == "strict":
            return best_overlap >= max(0.35, self.settings.merge_iou_threshold) and best_containment >= 0.70
        if mode == "aggressive":
            return best_overlap >= max(0.10, self.settings.merge_iou_threshold * 0.6) or best_containment >= 0.40
        return best_overlap >= self.settings.merge_iou_threshold or best_containment >= 0.55


def create_default_detector(
    model_name: str = "yolov8n.pt",
    confidence: float = 0.25,
    *,
    profile: str = "balanced",
    child_attach_mode: str | None = None,
    model_registry_path: str | None = None,
    image_size: int | None = None,
    max_boxes: int | None = None,
    motion_threshold: int | None = None,
    min_motion_area: int | None = None,
    smoothing_alpha: float | None = None,
    hysteresis_frames: int | None = None,
) -> Detector:
    resolved_model_name = _resolve_model_name(model_name, model_registry_path=model_registry_path)
    settings = DetectorSettings.from_profile(
        model_name=resolved_model_name,
        confidence=confidence,
        profile_name=profile,
        image_size=image_size,
        max_boxes=max_boxes,
        motion_threshold=motion_threshold,
        min_motion_area=min_motion_area,
        smoothing_alpha=smoothing_alpha,
        hysteresis_frames=hysteresis_frames,
        child_attach_mode=child_attach_mode,
    )
    return HybridDetector(settings=settings)
