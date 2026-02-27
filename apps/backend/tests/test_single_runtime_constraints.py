from __future__ import annotations

import ast
import tomllib
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def test_only_single_runtime_entrypoint_script() -> None:
    pyproject = _repo_root() / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    scripts = data.get("project", {}).get("scripts", {})
    assert set(scripts.keys()) == {"sentinel"}
    assert scripts["sentinel"] == "sentinel.cli:main"


def test_no_subprocess_or_multiprocessing_imports_in_runtime_code() -> None:
    banned_modules = {"subprocess", "multiprocessing", "celery", "redis"}
    package_root = _repo_root() / "apps" / "backend" / "sentinel"
    violations: list[str] = []

    for path in package_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name.split(".", 1)[0]
                    if module in banned_modules:
                        violations.append(f"{path}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module = node.module.split(".", 1)[0]
                    if module in banned_modules:
                        violations.append(f"{path}: from {node.module} import ...")

    assert violations == []
