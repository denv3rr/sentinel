from __future__ import annotations

import os
import sys
from pathlib import Path


def platform_default_data_dir() -> Path:
    home = Path.home()
    if sys.platform.startswith("win"):
        root = Path(os.environ.get("LOCALAPPDATA", home / "AppData" / "Local"))
        return root / "sentinel"
    if sys.platform == "darwin":
        return home / "Library" / "Application Support" / "sentinel"
    return Path(os.environ.get("XDG_DATA_HOME", home / ".local" / "share")) / "sentinel"


def bootstrap_config_path() -> Path:
    home = Path.home()
    return home / ".sentinel" / "bootstrap.json"


def ensure_data_tree(data_dir: Path) -> dict[str, Path]:
    tree = {
        "root": data_dir,
        "db": data_dir / "db",
        "media": data_dir / "media",
        "exports": data_dir / "exports",
        "logs": data_dir / "logs",
        "config": data_dir / "config",
    }
    for path in tree.values():
        path.mkdir(parents=True, exist_ok=True)
    return tree


def resolve_data_dir(cli_data_dir: str | None) -> Path:
    if cli_data_dir:
        return Path(cli_data_dir).expanduser().resolve()
    return platform_default_data_dir().resolve()