from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sentinel.util.paths import bootstrap_config_path, ensure_data_tree, resolve_data_dir

from .defaults import APP_VERSION, default_data_dir
from .schema import AppSettings


class SettingsStore:
    def __init__(self, cli_data_dir: str | None = None) -> None:
        self.bootstrap_path = bootstrap_config_path()
        self.bootstrap_path.parent.mkdir(parents=True, exist_ok=True)
        bootstrap = self._read_json(self.bootstrap_path, default={})

        configured = bootstrap.get("data_dir")
        chosen_dir = resolve_data_dir(cli_data_dir or configured or str(default_data_dir()))
        self._data_tree = ensure_data_tree(chosen_dir)

        self.settings_path = self._data_tree["config"] / "settings.json"
        raw_settings = self._read_json(self.settings_path, default={})
        migrated = migrate_settings(raw_settings, str(chosen_dir))
        self._settings = AppSettings.model_validate(migrated)
        self._settings.data_dir = str(chosen_dir)
        self.save()
        self._write_json(self.bootstrap_path, {"data_dir": str(chosen_dir)})

    @property
    def settings(self) -> AppSettings:
        return self._settings

    @property
    def data_tree(self) -> dict[str, Path]:
        return self._data_tree

    def update(self, **changes: Any) -> AppSettings:
        merged = self._settings.model_dump()
        merged.update(changes)
        self._settings = AppSettings.model_validate(merged)
        self.save()
        return self._settings

    def set_data_dir(self, new_data_dir: str) -> AppSettings:
        resolved = resolve_data_dir(new_data_dir)
        self._data_tree = ensure_data_tree(resolved)
        self.settings_path = self._data_tree["config"] / "settings.json"
        current = self._settings.model_dump()
        current["data_dir"] = str(resolved)
        self._settings = AppSettings.model_validate(current)
        self.save()
        self._write_json(self.bootstrap_path, {"data_dir": str(resolved)})
        return self._settings

    def save(self) -> None:
        payload = self._settings.model_dump(mode="json")
        self._write_json(self.settings_path, payload)

    @staticmethod
    def _read_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
        if not path.exists():
            return default
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return default

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def migrate_settings(raw: dict[str, Any], data_dir: str) -> dict[str, Any]:
    if not raw:
        return {
            "version": APP_VERSION,
            "data_dir": data_dir,
            "export_dir": None,
            "allow_lan": False,
            "operating_mode": "home",
            "armed": False,
            "telemetry_opt_in": False,
            "onboarding_completed": False,
            "retention": {"days": 30, "max_gb": 50.0},
            "label_thresholds": {
                "person": 0.35,
                "animal": 0.30,
                "vehicle": 0.35,
                "unknown": 0.45,
            },
            "cameras": [],
        }

    raw.setdefault("version", APP_VERSION)
    raw.setdefault("data_dir", data_dir)
    raw.setdefault("export_dir", None)
    raw.setdefault("allow_lan", False)
    raw.setdefault("operating_mode", "home")
    raw.setdefault("armed", False)
    raw.setdefault("telemetry_opt_in", False)
    raw.setdefault("onboarding_completed", False)
    raw.setdefault("retention", {"days": 30, "max_gb": 50.0})
    raw.setdefault(
        "label_thresholds",
        {
            "person": 0.35,
            "animal": 0.30,
            "vehicle": 0.35,
            "unknown": 0.45,
        },
    )
    raw.setdefault("cameras", [])
    return raw
