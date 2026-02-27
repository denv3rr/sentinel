from __future__ import annotations

import cv2
import numpy as np

from .detect_base import Detection, Detector

CATEGORY_MAP = {
    "person": "person",
    "bicycle": "vehicle",
    "car": "vehicle",
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
}


class _FallbackUnknownDetector(Detector):
    def __init__(self) -> None:
        self._prev_gray: np.ndarray | None = None

    def detect(self, frame: np.ndarray) -> list[Detection]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        if self._prev_gray is None:
            self._prev_gray = gray
            return []

        delta = cv2.absdiff(self._prev_gray, gray)
        _, threshold = cv2.threshold(delta, 28, 255, cv2.THRESH_BINARY)
        threshold = cv2.dilate(threshold, None, iterations=2)
        contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self._prev_gray = gray

        detections: list[Detection] = []
        for contour in contours:
            if cv2.contourArea(contour) < 1500:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            detections.append(Detection(bbox=(x, y, x + w, y + h), confidence=0.35, label="unknown"))
        return detections


class UltralyticsDetector(Detector):
    def __init__(self, model_name: str = "yolov8n.pt", confidence: float = 0.25) -> None:
        self.confidence = confidence
        self.model_name = model_name
        self._model = None
        self._fallback = _FallbackUnknownDetector()

        try:
            from ultralytics import YOLO  # type: ignore

            self._model = YOLO(model_name)
        except Exception:
            self._model = None

    def detect(self, frame: np.ndarray) -> list[Detection]:
        if self._model is None:
            return self._fallback.detect(frame)

        results = self._model.predict(frame, conf=self.confidence, verbose=False)
        detections: list[Detection] = []

        for result in results:
            names = result.names
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls.item())
                raw_label = str(names.get(cls_id, "unknown"))
                mapped_label = CATEGORY_MAP.get(raw_label, "unknown")
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(
                    Detection(
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        confidence=float(box.conf.item()),
                        label=mapped_label,
                        raw_label=raw_label,
                    )
                )

        return detections


def create_default_detector(model_name: str = "yolov8n.pt", confidence: float = 0.25) -> Detector:
    return UltralyticsDetector(model_name=model_name, confidence=confidence)