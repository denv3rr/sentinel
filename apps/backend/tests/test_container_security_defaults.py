from __future__ import annotations

from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def test_dockerfile_uses_non_root_user_and_runtime_data_dir() -> None:
    dockerfile = (_repo_root() / "Dockerfile").read_text(encoding="utf-8")

    assert "USER sentinel" in dockerfile
    assert "--data-dir" in dockerfile
    assert "/runtime_data" in dockerfile


def test_compose_applies_container_hardening_defaults() -> None:
    compose = (_repo_root() / "docker-compose.yml").read_text(encoding="utf-8")

    assert "./runtime_data:/runtime_data" in compose
    assert "--data-dir" in compose
    assert "/runtime_data" in compose
    assert "read_only: true" in compose
    assert "tmpfs:" in compose
    assert "- /tmp" in compose
    assert "cap_drop:" in compose
    assert "- ALL" in compose
    assert "security_opt:" in compose
    assert "no-new-privileges:true" in compose
