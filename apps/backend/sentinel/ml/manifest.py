from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any


@dataclass
class ModelManifest:
    model_id: str
    version: str
    task: str
    labels: list[str]
    thresholds: dict[str, float]
    input_size: int
    training_data_tag: str
    created_at: str
    weights_path: str
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ModelManifest":
        return cls(
            model_id=str(payload["model_id"]),
            version=str(payload.get("version", "1.0.0")),
            task=str(payload.get("task", "detect")),
            labels=[str(item) for item in payload.get("labels", [])],
            thresholds={str(k): float(v) for k, v in dict(payload.get("thresholds", {})).items()},
            input_size=int(payload.get("input_size", 960)),
            training_data_tag=str(payload.get("training_data_tag", "unknown")),
            created_at=str(payload.get("created_at", datetime.now(UTC).isoformat())),
            weights_path=str(payload.get("weights_path", "")),
            metrics={str(k): float(v) for k, v in dict(payload.get("metrics", {})).items()},
            metadata=dict(payload.get("metadata", {})),
        )


def _load_yaml_optional(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as exc:
        msg = "YAML manifests require PyYAML to be installed"
        raise RuntimeError(msg) from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        msg = f"Manifest payload must be an object: {path}"
        raise ValueError(msg)
    return data


def _load_dict(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        data = _load_yaml_optional(path)
    if not isinstance(data, dict):
        msg = f"Manifest payload must be an object: {path}"
        raise ValueError(msg)
    return data


def save_manifest(manifest: ModelManifest, path: str | Path) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    suffix = target.suffix.lower()
    payload = manifest.to_dict()
    if suffix in {"", ".json"}:
        if suffix == "":
            target = target.with_suffix(".json")
        target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return target

    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as exc:
            msg = "YAML manifests require PyYAML to be installed"
            raise RuntimeError(msg) from exc
        target.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
        return target

    msg = f"Unsupported manifest extension: {target.suffix}"
    raise ValueError(msg)


def load_manifest(path: str | Path) -> ModelManifest:
    source = Path(path).expanduser().resolve()
    if not source.exists():
        msg = f"Manifest not found: {source}"
        raise FileNotFoundError(msg)
    payload = _load_dict(source)
    return ModelManifest.from_dict(payload)


class ModelRegistry:
    def __init__(self, registry_path: str | Path) -> None:
        self.path = Path(registry_path).expanduser().resolve()

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"aliases": {}}
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {"aliases": {}}
        aliases = data.get("aliases")
        if not isinstance(aliases, dict):
            data["aliases"] = {}
        return data

    def _write(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def register(self, alias: str, manifest_path: str | Path) -> Path:
        normalized = alias.strip()
        if not normalized:
            msg = "Alias must not be empty"
            raise ValueError(msg)
        data = self._read()
        aliases = dict(data.get("aliases", {}))
        resolved = str(Path(manifest_path).expanduser().resolve())
        aliases[normalized] = resolved
        data["aliases"] = aliases
        self._write(data)
        return Path(resolved)

    def resolve(self, alias_or_path: str) -> Path:
        candidate = Path(alias_or_path).expanduser()
        if candidate.exists():
            return candidate.resolve()
        data = self._read()
        aliases = data.get("aliases", {})
        if isinstance(aliases, dict) and alias_or_path in aliases:
            return Path(str(aliases[alias_or_path])).expanduser().resolve()
        msg = f"Unknown model alias/path: {alias_or_path}"
        raise FileNotFoundError(msg)

