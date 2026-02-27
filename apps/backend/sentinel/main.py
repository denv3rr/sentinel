from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
import threading
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from sentinel.api import (
    routes_cameras,
    routes_events,
    routes_exports,
    routes_health,
    routes_settings,
)
from sentinel.camera.opencv_cam import detect_default_webcam_index
from sentinel.config.migrate import SettingsStore
from sentinel.pipeline.runtime import PipelineRuntime
from sentinel.storage.db import Database
from sentinel.storage.export import ExportService
from sentinel.storage.repo import StorageRepo
from sentinel.storage.retention import RetentionService
from sentinel.util.logging import get_logger, setup_logging
from sentinel.util.paths import ensure_data_tree
from sentinel.util.security import SecretStore, resolve_path_within_base

logger = get_logger(__name__)


@dataclass
class SentinelState:
    settings_store: SettingsStore
    log_level: str
    db: Database
    repo: StorageRepo
    runtime: PipelineRuntime
    export_service: ExportService
    retention_service: RetentionService
    secret_store: SecretStore
    data_dir: Path
    _shutdown_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _shutdown_started: bool = field(default=False, init=False, repr=False)
    _shutdown_complete: bool = field(default=False, init=False, repr=False)

    @classmethod
    def create(cls, data_dir: str | None = None, bind: str | None = None, port: int | None = None, log_level: str = "info") -> "SentinelState":
        settings_store = SettingsStore(cli_data_dir=data_dir)

        updates: dict[str, Any] = {}
        if bind:
            updates["bind"] = bind
            updates["allow_lan"] = bind == "0.0.0.0"
        if port:
            updates["port"] = port
        if updates:
            settings_store.update(**updates)

        data_path = Path(settings_store.settings.data_dir)
        ensure_data_tree(data_path)
        setup_logging(log_level, data_path)

        secret_store = SecretStore(data_path)
        db = Database(data_path / "db" / "sentinel.db")
        repo = StorageRepo(db)

        stored_cameras = repo.list_cameras()
        if not stored_cameras:
            for camera in settings_store.settings.cameras:
                repo.upsert_camera(camera.model_dump(mode="json"))
            stored_cameras = repo.list_cameras()
        if not stored_cameras:
            default_index = detect_default_webcam_index(max_index=5)
            if default_index is not None:
                default_camera = {
                    "id": f"cam-{uuid4().hex[:8]}",
                    "name": "Default Webcam",
                    "kind": "webcam",
                    "source": str(default_index),
                    "enabled": True,
                    "detection_enabled": False,
                    "recording_mode": "event_only",
                    "labels": ["person", "animal", "vehicle", "unknown"],
                    "min_confidence": dict(settings_store.settings.label_thresholds),
                    "cooldown_seconds": 8,
                    "motion_threshold": 0.012,
                    "inference_max_side": 960,
                    "zones": [],
                }
                repo.upsert_camera(default_camera)
                stored_cameras = repo.list_cameras()
                logger.info("Auto-added default webcam at index %s", default_index)
            else:
                logger.info("No default webcam detected during startup bootstrap")

        settings_store.update(cameras=stored_cameras, onboarding_completed=bool(stored_cameras))

        runtime = PipelineRuntime(
            repo=repo,
            data_dir=data_path,
            secret_store=secret_store,
            global_thresholds=settings_store.settings.label_thresholds,
            global_armed=settings_store.settings.armed,
        )
        runtime.sync_cameras(stored_cameras)

        export_service = ExportService(repo, data_path)
        retention_service = RetentionService(repo, data_path)

        return cls(
            settings_store=settings_store,
            log_level=log_level,
            db=db,
            repo=repo,
            runtime=runtime,
            export_service=export_service,
            retention_service=retention_service,
            secret_store=secret_store,
            data_dir=data_path,
        )

    def refresh_workers(self) -> None:
        self.runtime.sync_cameras(self.repo.list_cameras())

    def change_data_dir(self, new_data_dir: str) -> None:
        cameras = self.repo.list_cameras()
        self.settings_store.update(cameras=cameras, onboarding_completed=bool(cameras))
        self.runtime.stop_all()
        self.db.close()

        self.settings_store.set_data_dir(new_data_dir)
        self.data_dir = Path(self.settings_store.settings.data_dir)
        ensure_data_tree(self.data_dir)
        setup_logging(self.log_level, self.data_dir)

        self.secret_store = SecretStore(self.data_dir)
        self.db = Database(self.data_dir / "db" / "sentinel.db")
        self.repo = StorageRepo(self.db)
        for camera in cameras:
            self.repo.upsert_camera(camera)

        self.runtime = PipelineRuntime(
            repo=self.repo,
            data_dir=self.data_dir,
            secret_store=self.secret_store,
            global_thresholds=self.settings_store.settings.label_thresholds,
            global_armed=self.settings_store.settings.armed,
        )
        self.runtime.sync_cameras(cameras)
        self.export_service = ExportService(self.repo, self.data_dir)
        self.retention_service = RetentionService(self.repo, self.data_dir)

    def begin_shutdown(self) -> None:
        with self._shutdown_lock:
            if self._shutdown_started:
                return
            self._shutdown_started = True
        self.runtime.stop_all()

    def shutdown(self) -> None:
        self.begin_shutdown()
        with self._shutdown_lock:
            if self._shutdown_complete:
                return
            self._shutdown_complete = True
        self.db.close()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _frontend_dist_dir() -> Path:
    return _repo_root() / "apps" / "frontend" / "dist"


def _packaged_ui_dir() -> Path:
    return Path(__file__).resolve().parent / "web"


def create_app(
    data_dir: str | None = None,
    bind: str | None = None,
    port: int | None = None,
    log_level: str = "info",
) -> FastAPI:
    state = SentinelState.create(data_dir=data_dir, bind=bind, port=port, log_level=log_level)
    dist_dir = _frontend_dist_dir()
    packaged_ui = _packaged_ui_dir()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            yield
        finally:
            app.state.sentinel.shutdown()

    app = FastAPI(title="Sentinel", version="0.1.0", lifespan=lifespan)
    app.state.sentinel = state

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(routes_health.router, prefix="/api")
    app.include_router(routes_cameras.router, prefix="/api")
    app.include_router(routes_events.router, prefix="/api")
    app.include_router(routes_exports.router, prefix="/api")
    app.include_router(routes_settings.router, prefix="/api")

    @app.get("/api/media/{relative_path:path}")
    def get_media(relative_path: str, request: Request) -> FileResponse:
        settings = request.app.state.sentinel.settings_store.settings
        base = Path(settings.data_dir).resolve()
        target = resolve_path_within_base(base, relative_path)
        if target is None:
            raise HTTPException(status_code=400, detail="Invalid path")
        exports_base = Path(settings.export_dir).resolve() if settings.export_dir else (base / "exports").resolve()
        media_base = (base / "media").resolve()
        if not (target.is_relative_to(media_base) or target.is_relative_to(exports_base)):
            raise HTTPException(status_code=400, detail="Invalid path")
        if not target.exists() or not target.is_file():
            raise HTTPException(status_code=404, detail="Media not found")
        return FileResponse(target)

    ui_dir = dist_dir if (dist_dir / "index.html").exists() else packaged_ui
    assets = ui_dir / "assets"
    if assets.exists():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/", response_model=None)
    def root():
        index_path = ui_dir / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
        return HTMLResponse("<h1>Sentinel</h1><p>Frontend bundle missing.</p>")

    @app.get("/{full_path:path}", response_model=None)
    def spa(full_path: str):
        candidate = resolve_path_within_base(ui_dir, full_path)
        if candidate is not None and candidate.exists() and candidate.is_file():
            return FileResponse(candidate)
        index_path = ui_dir / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
        return HTMLResponse("<h1>Sentinel</h1><p>Frontend bundle missing.</p>")

    return app
