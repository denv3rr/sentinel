from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CalibrationPriors:
    focal_length_px: float
    target_height_m: float
    camera_tilt_deg: float = 0.0


@dataclass(frozen=True)
class RangeEstimate:
    depth_rel: float | None
    range_est_m: float | None
    range_sigma_m: float
    method: str
    calibrated: bool
    provenance: dict[str, object]


class RangeEstimator:
    """Range/depth helper for generic 2D object detections.

    This module intentionally provides relative depth by default. Metric range is only
    emitted when calibration priors are available.
    """

    def __init__(
        self,
        mode: str = "relative_depth",
        ema_alpha: float = 0.45,
        calibration: CalibrationPriors | None = None,
    ) -> None:
        self.mode = mode
        self.ema_alpha = max(0.0, min(1.0, float(ema_alpha)))
        self.calibration = calibration
        self._baseline_heights: dict[str, float] = {}
        self._smoothed_depth_rel: dict[str, float] = {}
        self._prev_height: dict[str, float] = {}
        self._prev_center: dict[str, tuple[float, float]] = {}

    def estimate(
        self,
        *,
        track_id: str | int,
        bbox: tuple[int, int, int, int],
        frame_shape: tuple[int, int] | tuple[int, int, int],
    ) -> RangeEstimate:
        frame_h = max(1, int(frame_shape[0]))
        x1, y1, x2, y2 = bbox
        bbox_h = max(1.0, float(y2 - y1))
        bbox_w = max(1.0, float(x2 - x1))
        area = bbox_h * bbox_w
        center = (float(x1 + x2) / 2.0, float(y1 + y2) / 2.0)
        key = str(track_id)

        if self.mode == "relative_depth_first_observation":
            baseline = self._baseline_heights.get(key)
            if baseline is None:
                baseline = bbox_h
                self._baseline_heights[key] = bbox_h
            depth_rel = baseline / bbox_h
        else:
            depth_rel = float(frame_h) / bbox_h

        prior_depth = self._smoothed_depth_rel.get(key)
        if prior_depth is not None:
            depth_rel = ((1.0 - self.ema_alpha) * prior_depth) + (self.ema_alpha * depth_rel)
        self._smoothed_depth_rel[key] = depth_rel

        jitter = self._bbox_jitter(key, bbox_h)
        blur_proxy = self._motion_blur_proxy(key, center, area)
        tilt_penalty = self._tilt_penalty(center_y=center[1], frame_h=frame_h)

        base_uncertainty = 0.08 + (jitter * 0.45) + (blur_proxy * 0.35) + (tilt_penalty * 0.22)
        base_uncertainty = max(0.05, min(base_uncertainty, 0.95))

        calibrated = self.mode == "calibrated_metric" and self.calibration is not None
        range_est_m: float | None = None
        range_sigma_m: float
        if calibrated and self.calibration:
            raw_range = (self.calibration.focal_length_px * self.calibration.target_height_m) / bbox_h
            tilt_rad = math.radians(self.calibration.camera_tilt_deg)
            cos_tilt = max(0.15, math.cos(tilt_rad))
            range_est_m = raw_range / cos_tilt
            range_sigma_m = range_est_m * base_uncertainty
        else:
            range_sigma_m = max(0.02, depth_rel * base_uncertainty)

        provenance = {
            "method": self.mode,
            "calibrated": calibrated,
            "priors": (
                {
                    "focal_length_px": self.calibration.focal_length_px,
                    "target_height_m": self.calibration.target_height_m,
                    "camera_tilt_deg": self.calibration.camera_tilt_deg,
                }
                if self.calibration
                else None
            ),
            "factors": {
                "bbox_jitter": round(jitter, 6),
                "motion_blur_proxy": round(blur_proxy, 6),
                "tilt_penalty": round(tilt_penalty, 6),
            },
        }
        return RangeEstimate(
            depth_rel=round(depth_rel, 6),
            range_est_m=(round(range_est_m, 6) if range_est_m is not None else None),
            range_sigma_m=round(range_sigma_m, 6),
            method=self.mode,
            calibrated=calibrated,
            provenance=provenance,
        )

    def _bbox_jitter(self, key: str, height: float) -> float:
        previous = self._prev_height.get(key)
        self._prev_height[key] = height
        if previous is None:
            return 0.0
        return min(1.0, abs(height - previous) / max(1.0, previous))

    def _motion_blur_proxy(self, key: str, center: tuple[float, float], area: float) -> float:
        previous = self._prev_center.get(key)
        self._prev_center[key] = center
        if previous is None:
            return 0.0
        dx = center[0] - previous[0]
        dy = center[1] - previous[1]
        speed = math.sqrt(dx * dx + dy * dy)
        # Lower area generally means less stable detector geometry under motion blur.
        area_factor = min(1.0, 5000.0 / max(500.0, area))
        return min(1.0, (speed / 60.0) * area_factor)

    @staticmethod
    def _tilt_penalty(center_y: float, frame_h: int) -> float:
        vertical_offset = abs((center_y / max(1.0, float(frame_h))) - 0.5) * 2.0
        return min(1.0, vertical_offset * 0.8)
