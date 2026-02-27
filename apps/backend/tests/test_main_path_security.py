from __future__ import annotations

from fastapi.testclient import TestClient

from sentinel.main import create_app
from sentinel.pipeline.runtime import PipelineRuntime


def _create_test_app(tmp_path, monkeypatch, ui_dir):
    bootstrap_path = tmp_path / "bootstrap.json"
    data_dir = tmp_path / "data"
    monkeypatch.setattr("sentinel.config.migrate.bootstrap_config_path", lambda: bootstrap_path)
    monkeypatch.setattr("sentinel.main.detect_default_webcam_index", lambda max_index=5: None)
    monkeypatch.setattr(PipelineRuntime, "sync_cameras", lambda self, cameras: None)
    monkeypatch.setattr("sentinel.main._frontend_dist_dir", lambda: ui_dir)
    monkeypatch.setattr("sentinel.main._packaged_ui_dir", lambda: ui_dir)
    app = create_app(data_dir=str(data_dir), log_level="warning")
    return app, data_dir


def test_media_route_blocks_directory_escape(tmp_path, monkeypatch) -> None:
    ui_dir = tmp_path / "ui"
    ui_dir.mkdir(parents=True, exist_ok=True)
    (ui_dir / "index.html").write_text("index", encoding="utf-8")
    app, data_dir = _create_test_app(tmp_path, monkeypatch, ui_dir)

    inside = data_dir / "media" / "ok.txt"
    inside.parent.mkdir(parents=True, exist_ok=True)
    inside.write_text("ok", encoding="utf-8")

    outside = data_dir.parent / "data-escape" / "secret.txt"
    outside.parent.mkdir(parents=True, exist_ok=True)
    outside.write_text("secret", encoding="utf-8")

    with TestClient(app) as client:
        allowed = client.get("/api/media/media/ok.txt")
        assert allowed.status_code == 200
        assert allowed.text == "ok"

        blocked = client.get("/api/media/..%2Fdata-escape%2Fsecret.txt")
        assert blocked.status_code == 400
        assert blocked.json()["detail"] == "Invalid path"

        blocked_non_media = client.get("/api/media/config/settings.json")
        assert blocked_non_media.status_code == 400
        assert blocked_non_media.json()["detail"] == "Invalid path"


def test_spa_route_does_not_serve_files_outside_ui_dir(tmp_path, monkeypatch) -> None:
    ui_dir = tmp_path / "ui"
    ui_dir.mkdir(parents=True, exist_ok=True)
    (ui_dir / "index.html").write_text("<h1>index</h1>", encoding="utf-8")
    (ui_dir / "inside.txt").write_text("inside", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("secret", encoding="utf-8")

    app, _ = _create_test_app(tmp_path, monkeypatch, ui_dir)

    with TestClient(app) as client:
        inside = client.get("/inside.txt")
        assert inside.status_code == 200
        assert inside.text == "inside"

        blocked = client.get("/..%2Fsecret.txt")
        assert blocked.status_code == 200
        assert blocked.text == "<h1>index</h1>"
        assert "secret" not in blocked.text
