from __future__ import annotations

from sentinel.config.schema import AppSettings, CameraConfig


def test_safe_defaults_for_detection_and_recording() -> None:
    camera = CameraConfig()
    assert camera.detection_enabled is False
    assert camera.recording_mode == "event_only"


def test_app_settings_default_disarmed() -> None:
    settings = AppSettings(data_dir="/tmp/sentinel-data")
    assert settings.armed is False