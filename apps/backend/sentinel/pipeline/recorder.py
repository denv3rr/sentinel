from __future__ import annotations

import datetime as dt
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np


class MediaRecorder:
    def __init__(self, data_dir: Path, clip_seconds: int = 6) -> None:
        self.data_dir = data_dir
        self.clip_seconds = max(0, clip_seconds)

    def _media_dir_for(self, camera_id: str, event_time_utc: dt.datetime) -> Path:
        media_dir = (
            self.data_dir
            / "media"
            / camera_id
            / event_time_utc.strftime("%Y")
            / event_time_utc.strftime("%m")
            / event_time_utc.strftime("%d")
        )
        media_dir.mkdir(parents=True, exist_ok=True)
        return media_dir

    def write_thumbnail(self, camera_id: str, event_id: str, frame: np.ndarray, wall_time_iso: str) -> str | None:
        try:
            event_time = dt.datetime.fromisoformat(wall_time_iso)
            if event_time.tzinfo is None:
                event_time = event_time.replace(tzinfo=dt.timezone.utc)
            media_dir = self._media_dir_for(camera_id, event_time.astimezone(dt.timezone.utc))
            output = media_dir / f"{event_id}_thumb.jpg"
            ok = cv2.imwrite(str(output), frame)
            if not ok:
                return None
            return str(output.relative_to(self.data_dir))
        except Exception:
            return None

    def write_clip(
        self,
        camera_id: str,
        event_id: str,
        frames: list[np.ndarray],
        wall_time_iso: str,
        fps: float,
    ) -> str | None:
        if self.clip_seconds <= 0 or not frames:
            return None

        try:
            event_time = dt.datetime.fromisoformat(wall_time_iso)
            if event_time.tzinfo is None:
                event_time = event_time.replace(tzinfo=dt.timezone.utc)
            media_dir = self._media_dir_for(camera_id, event_time.astimezone(dt.timezone.utc))
            output = media_dir / f"{event_id}_clip.mp4"

            height, width = frames[0].shape[:2]
            writer = cv2.VideoWriter(
                str(output),
                cv2.VideoWriter_fourcc(*"mp4v"),
                max(1.0, fps),
                (width, height),
            )
            for frame in frames:
                if frame.shape[0] != height or frame.shape[1] != width:
                    frame = cv2.resize(frame, (width, height))
                writer.write(frame)
            writer.release()
            if output.exists():
                return str(output.relative_to(self.data_dir))
        except Exception:
            return None
        return None

    def prepare_continuous_clip_path(
        self,
        camera_id: str,
        wall_time_iso: str,
        mode: str,
    ) -> tuple[Path, str] | None:
        try:
            event_time = dt.datetime.fromisoformat(wall_time_iso)
            if event_time.tzinfo is None:
                event_time = event_time.replace(tzinfo=dt.timezone.utc)
            media_dir = self._media_dir_for(camera_id, event_time.astimezone(dt.timezone.utc))
            stamp = event_time.strftime("%H%M%S")
            output = media_dir / f"{mode}_{stamp}_{uuid4().hex[:8]}.mp4"
            return output, str(output.relative_to(self.data_dir))
        except Exception:
            return None
