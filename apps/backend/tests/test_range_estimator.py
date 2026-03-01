from __future__ import annotations

from sentinel.modules.range_estimator import CalibrationPriors, RangeEstimator


def test_relative_depth_mode_outputs_depth_and_uncertainty() -> None:
    estimator = RangeEstimator(mode="relative_depth")
    estimate = estimator.estimate(track_id="t1", bbox=(10, 20, 60, 140), frame_shape=(480, 640))

    assert estimate.method == "relative_depth"
    assert estimate.depth_rel is not None
    assert estimate.depth_rel > 0.0
    assert estimate.range_est_m is None
    assert estimate.range_sigma_m > 0.0
    assert estimate.calibrated is False


def test_first_observation_mode_normalizes_against_initial_bbox_height() -> None:
    estimator = RangeEstimator(mode="relative_depth_first_observation")
    first = estimator.estimate(track_id="t1", bbox=(20, 30, 80, 130), frame_shape=(480, 640))
    second = estimator.estimate(track_id="t1", bbox=(20, 30, 80, 180), frame_shape=(480, 640))

    assert first.depth_rel is not None
    assert second.depth_rel is not None
    assert second.depth_rel < first.depth_rel
    assert second.range_est_m is None


def test_calibrated_metric_mode_outputs_meters_when_calibrated() -> None:
    estimator = RangeEstimator(
        mode="calibrated_metric",
        calibration=CalibrationPriors(focal_length_px=900.0, target_height_m=1.7, camera_tilt_deg=5.0),
    )
    estimate = estimator.estimate(track_id="t2", bbox=(100, 100, 180, 260), frame_shape=(720, 1280))

    assert estimate.calibrated is True
    assert estimate.range_est_m is not None
    assert estimate.range_est_m > 0.0
    assert estimate.range_sigma_m > 0.0
    assert estimate.provenance["priors"] is not None


def test_calibrated_metric_without_priors_keeps_metric_range_null() -> None:
    estimator = RangeEstimator(mode="calibrated_metric")
    estimate = estimator.estimate(track_id="t3", bbox=(50, 50, 120, 200), frame_shape=(600, 800))

    assert estimate.calibrated is False
    assert estimate.range_est_m is None
    assert estimate.depth_rel is not None

