from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import importlib
import json
from pathlib import Path
import time
from typing import Any
from uuid import uuid4

import numpy as np

from sentinel.vision.yolo_ultralytics import create_default_detector

from .dataset import DatasetConfig, dataset_fingerprint, load_dataset_config
from .manifest import ModelManifest, ModelRegistry, load_manifest, save_manifest
from .reproducibility import configure_reproducibility, default_run_metadata


def _load_yolo(model_name: str) -> Any:
    ultra = importlib.import_module("ultralytics")
    yolo_cls = ultra.YOLO
    return yolo_cls(model_name)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _extract_metric(value: Any, fallback: float = 0.0) -> float:
    if isinstance(value, (float, int)):
        return float(value)
    for attr in ("item",):
        obj = getattr(value, attr, None)
        if callable(obj):
            try:
                return float(obj())
            except Exception:
                continue
    return float(fallback)


def _best_weight_path(run_result: Any, output_dir: Path, run_name: str) -> Path:
    save_dir = getattr(run_result, "save_dir", None)
    if isinstance(save_dir, (str, Path)):
        candidate = Path(save_dir) / "weights" / "best.pt"
        if candidate.exists():
            return candidate.resolve()
    fallback = output_dir / run_name / "weights" / "best.pt"
    if fallback.exists():
        return fallback.resolve()
    return fallback.resolve()


def _manifest_from_train(
    *,
    model_id: str,
    weights_path: Path,
    dataset_cfg: DatasetConfig,
    image_size: int,
    confidence: float,
    metrics: dict[str, float],
    metadata: dict[str, Any],
) -> ModelManifest:
    return ModelManifest(
        model_id=model_id,
        version="1.0.0",
        task="detect",
        labels=list(dataset_cfg.labels),
        thresholds={label: float(confidence) for label in dataset_cfg.labels},
        input_size=int(image_size),
        training_data_tag=dataset_cfg.tag,
        created_at=_utc_now(),
        weights_path=str(weights_path),
        metrics=metrics,
        metadata=metadata,
    )


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


@dataclass
class TrainResult:
    ok: bool
    run_dir: str
    manifest_path: str
    metadata_path: str
    model_id: str
    weights_path: str
    metrics: dict[str, float]


def run_train(
    *,
    dataset_config_path: str,
    model_name: str = "yolov8n.pt",
    output_dir: str = "artifacts/ml/train",
    alias: str | None = None,
    profile: str = "balanced",
    image_size: int = 960,
    epochs: int = 10,
    batch_size: int = 8,
    confidence: float = 0.25,
    seed: int = 7,
    deterministic: bool = True,
    registry_path: str = "artifacts/ml/registry.json",
    dry_run: bool = False,
) -> TrainResult:
    output_root = Path(output_dir).expanduser().resolve()
    run_name = f"train-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:6]}"
    run_dir = output_root / run_name
    dataset_cfg = load_dataset_config(dataset_config_path)
    dataset_hash = dataset_fingerprint(dataset_config_path)

    reproducibility = configure_reproducibility(seed=seed, deterministic=deterministic)
    metrics: dict[str, float]
    weights_path: Path

    if dry_run:
        run_dir.mkdir(parents=True, exist_ok=True)
        weights_path = run_dir / "weights" / "best.pt"
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        weights_path.touch(exist_ok=True)
        metrics = {"map50": 0.0, "map50_95": 0.0, "precision": 0.0, "recall": 0.0}
    else:
        model = _load_yolo(model_name)
        result = model.train(
            data=str(Path(dataset_config_path).expanduser().resolve()),
            epochs=max(1, int(epochs)),
            imgsz=max(64, int(image_size)),
            project=str(output_root),
            name=run_name,
            batch=max(1, int(batch_size)),
            seed=int(seed),
            deterministic=bool(deterministic),
            workers=0,
            verbose=False,
        )
        weights_path = _best_weight_path(result, output_root, run_name)
        box = getattr(result, "box", None)
        metrics = {
            "map50": _extract_metric(getattr(box, "map50", 0.0), 0.0),
            "map50_95": _extract_metric(getattr(box, "map", 0.0), 0.0),
            "precision": _extract_metric(getattr(box, "mp", 0.0), 0.0),
            "recall": _extract_metric(getattr(box, "mr", 0.0), 0.0),
        }

    model_id = alias or f"{Path(model_name).stem}-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"
    metadata = default_run_metadata(
        {
            "dataset_config_path": str(Path(dataset_config_path).expanduser().resolve()),
            "model_name": model_name,
            "profile": profile,
            "image_size": image_size,
            "epochs": epochs,
            "batch_size": batch_size,
            "confidence": confidence,
            "seed": seed,
            "deterministic": deterministic,
            "dry_run": dry_run,
        },
        dataset_hash=dataset_hash,
    )
    metadata["reproducibility"] = reproducibility
    metadata["metrics"] = metrics
    metadata["dataset"] = {
        "name": dataset_cfg.name,
        "tag": dataset_cfg.tag,
        "labels": dataset_cfg.labels,
    }

    manifest = _manifest_from_train(
        model_id=model_id,
        weights_path=weights_path,
        dataset_cfg=dataset_cfg,
        image_size=image_size,
        confidence=confidence,
        metrics=metrics,
        metadata={"profile": profile},
    )
    manifest_path = save_manifest(manifest, run_dir / "manifest.json")
    metadata_path = _write_json(run_dir / "run_metadata.json", metadata)

    registry = ModelRegistry(registry_path)
    registry.register(alias or model_id, manifest_path)

    return TrainResult(
        ok=True,
        run_dir=str(run_dir),
        manifest_path=str(manifest_path),
        metadata_path=str(metadata_path),
        model_id=model_id,
        weights_path=str(weights_path),
        metrics=metrics,
    )


@dataclass
class EvalResult:
    ok: bool
    manifest_path: str
    metadata_path: str
    metrics: dict[str, float]


def run_eval(
    *,
    model: str,
    dataset_config_path: str,
    output_dir: str = "artifacts/ml/eval",
    image_size: int | None = None,
    seed: int = 7,
    deterministic: bool = True,
    registry_path: str = "artifacts/ml/registry.json",
    dry_run: bool = False,
) -> EvalResult:
    registry = ModelRegistry(registry_path)
    manifest = load_manifest(registry.resolve(model))
    dataset_cfg = load_dataset_config(dataset_config_path)

    reproducibility = configure_reproducibility(seed=seed, deterministic=deterministic)
    output_root = Path(output_dir).expanduser().resolve()
    run_dir = output_root / f"eval-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:6]}"
    run_dir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        metrics = {"map50": 0.0, "map50_95": 0.0, "precision": 0.0, "recall": 0.0}
    else:
        model_obj = _load_yolo(manifest.weights_path)
        result = model_obj.val(
            data=str(Path(dataset_config_path).expanduser().resolve()),
            imgsz=int(image_size or manifest.input_size),
            workers=0,
            verbose=False,
        )
        box = getattr(result, "box", None)
        metrics = {
            "map50": _extract_metric(getattr(box, "map50", 0.0), 0.0),
            "map50_95": _extract_metric(getattr(box, "map", 0.0), 0.0),
            "precision": _extract_metric(getattr(box, "mp", 0.0), 0.0),
            "recall": _extract_metric(getattr(box, "mr", 0.0), 0.0),
        }

    metadata = default_run_metadata(
        {
            "mode": "eval",
            "manifest_model_id": manifest.model_id,
            "dataset_config_path": str(Path(dataset_config_path).expanduser().resolve()),
            "image_size": int(image_size or manifest.input_size),
            "seed": seed,
            "deterministic": deterministic,
            "dry_run": dry_run,
        },
        dataset_hash=dataset_fingerprint(dataset_config_path),
    )
    metadata["metrics"] = metrics
    metadata["reproducibility"] = reproducibility
    metadata["dataset"] = {"name": dataset_cfg.name, "tag": dataset_cfg.tag}
    metadata_path = _write_json(run_dir / "eval_metadata.json", metadata)

    return EvalResult(
        ok=True,
        manifest_path=str(registry.resolve(model)),
        metadata_path=str(metadata_path),
        metrics=metrics,
    )


@dataclass
class ExportResult:
    ok: bool
    manifest_path: str
    metadata_path: str
    export_path: str


def run_export(
    *,
    model: str,
    output_dir: str = "artifacts/ml/export",
    export_format: str = "onnx",
    image_size: int | None = None,
    optimize: bool = False,
    registry_path: str = "artifacts/ml/registry.json",
    dry_run: bool = False,
) -> ExportResult:
    registry = ModelRegistry(registry_path)
    manifest_path = registry.resolve(model)
    manifest = load_manifest(manifest_path)
    run_dir = Path(output_dir).expanduser().resolve() / f"export-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:6]}"
    run_dir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        export_path = run_dir / f"{Path(manifest.weights_path).stem}.{export_format}"
        export_path.touch(exist_ok=True)
    else:
        model_obj = _load_yolo(manifest.weights_path)
        export_result = model_obj.export(
            format=export_format,
            imgsz=int(image_size or manifest.input_size),
            optimize=bool(optimize),
            simplify=bool(optimize),
        )
        export_path = Path(str(export_result)).expanduser().resolve()

    metadata = default_run_metadata(
        {
            "mode": "export",
            "manifest_model_id": manifest.model_id,
            "export_format": export_format,
            "image_size": int(image_size or manifest.input_size),
            "optimize": optimize,
            "dry_run": dry_run,
        }
    )
    metadata["export_path"] = str(export_path)
    metadata_path = _write_json(run_dir / "export_metadata.json", metadata)

    return ExportResult(
        ok=True,
        manifest_path=str(manifest_path),
        metadata_path=str(metadata_path),
        export_path=str(export_path),
    )


@dataclass
class BenchmarkResult:
    ok: bool
    profile: str
    model_name: str
    cold_start_ms: float
    warm_inference_ms: float
    fast_motion_fps: float


def run_benchmark(
    *,
    model_name: str = "yolov8n.pt",
    profile: str = "balanced",
    confidence: float = 0.25,
    frames: int = 120,
    frame_width: int = 640,
    frame_height: int = 360,
) -> BenchmarkResult:
    frame_static = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    frame_motion = frame_static.copy()
    cv_width = max(10, frame_width // 12)
    cv_height = max(10, frame_height // 7)

    cold_started = time.perf_counter()
    detector = create_default_detector(
        model_name=model_name,
        confidence=confidence,
        profile=profile,
    )
    _ = detector.detect(frame_static)
    cold_ms = (time.perf_counter() - cold_started) * 1000.0

    warm_loops = max(30, int(frames))
    warm_started = time.perf_counter()
    for _ in range(warm_loops):
        _ = detector.detect(frame_static)
    warm_ms = ((time.perf_counter() - warm_started) * 1000.0) / float(warm_loops)

    motion_started = time.perf_counter()
    for index in range(warm_loops):
        frame_motion.fill(0)
        x = int((index * 17) % max(1, frame_width - cv_width))
        y = int((index * 11) % max(1, frame_height - cv_height))
        frame_motion[y : y + cv_height, x : x + cv_width] = 255
        _ = detector.detect(frame_motion)
    motion_seconds = max(1e-6, (time.perf_counter() - motion_started))
    motion_fps = float(warm_loops) / motion_seconds

    return BenchmarkResult(
        ok=True,
        profile=profile,
        model_name=model_name,
        cold_start_ms=round(cold_ms, 3),
        warm_inference_ms=round(warm_ms, 3),
        fast_motion_fps=round(motion_fps, 3),
    )
