from __future__ import annotations

from sentinel.config.schema import AppSettings, CameraConfig


def test_safe_defaults_for_detection_and_recording() -> None:
    camera = CameraConfig()
    assert camera.detection_enabled is False
    assert camera.recording_mode == "event_only"
    assert camera.detector_profile == "balanced"
    assert camera.child_attach_mode == "balanced"
    assert camera.tracker_profile == "balanced"
    assert camera.tracker_profile_auto is True
    assert camera.tracker_iou_threshold == 0.35
    assert camera.tracker_max_age == 18
    assert camera.tracker_memory_age == 45
    assert camera.tracker_use_appearance is True
    assert camera.overlay_show_children is True
    assert camera.range_mode == "relative_depth"


def test_app_settings_default_disarmed() -> None:
    settings = AppSettings(data_dir="/tmp/sentinel-data")
    assert settings.operating_mode == "home"
    assert settings.armed is False
