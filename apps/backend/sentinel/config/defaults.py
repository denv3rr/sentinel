from __future__ import annotations

from pathlib import Path

from sentinel.util.paths import platform_default_data_dir

APP_VERSION = 1
DEFAULT_BIND = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_LOG_LEVEL = "info"
DEFAULT_MODEL = "yolov8n.pt"
DEFAULT_RETENTION_DAYS = 30
DEFAULT_RETENTION_MAX_GB = 50.0
DEFAULT_EVENT_COOLDOWN_SECONDS = 8
DEFAULT_MOTION_THRESHOLD = 0.012
DEFAULT_CLIP_SECONDS = 6
DEFAULT_LABEL_THRESHOLDS = {
    "person": 0.35,
    "animal": 0.30,
    "vehicle": 0.35,
    "unknown": 0.45,
}


def default_data_dir() -> Path:
    return platform_default_data_dir()