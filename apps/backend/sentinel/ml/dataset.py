from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    train: str
    val: str
    test: str | None
    labels: list[str]
    tag: str
    metadata: dict[str, Any]


def _load_yaml_optional(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as exc:
        msg = "YAML dataset configs require PyYAML to be installed"
        raise RuntimeError(msg) from exc

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        msg = f"Dataset config must contain an object: {path}"
        raise ValueError(msg)
    return data


def _load_config_dict(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix == ".json":
        data = json.loads(text)
    elif suffix in {".yaml", ".yml"}:
        return _load_yaml_optional(path)
    else:
        # Try JSON first, then YAML.
        try:
            data = json.loads(text)
        except Exception:
            return _load_yaml_optional(path)

    if not isinstance(data, dict):
        msg = f"Dataset config must contain an object: {path}"
        raise ValueError(msg)
    return data


def dataset_fingerprint(path: str | Path) -> str:
    config_path = Path(path).expanduser().resolve()
    h = sha256()
    h.update(str(config_path).encode("utf-8"))
    if config_path.exists() and config_path.is_file():
        h.update(config_path.read_bytes())
    return h.hexdigest()


def load_dataset_config(path: str | Path) -> DatasetConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        msg = f"Dataset config not found: {config_path}"
        raise FileNotFoundError(msg)

    payload = _load_config_dict(config_path)

    train = str(payload.get("train", "")).strip()
    val = str(payload.get("val", "")).strip()
    if not train or not val:
        msg = "Dataset config requires non-empty 'train' and 'val' fields"
        raise ValueError(msg)

    label_names: list[str] = []
    names = payload.get("names")
    if isinstance(names, dict):
        def _key_order(item: object) -> tuple[int, int | str]:
            try:
                return (0, int(str(item)))
            except Exception:
                return (1, str(item))

        label_names = [str(names[key]) for key in sorted(names.keys(), key=_key_order)]
    elif isinstance(names, list):
        label_names = [str(item) for item in names]
    elif isinstance(payload.get("labels"), list):
        label_names = [str(item) for item in payload.get("labels", [])]
    if not label_names:
        label_names = ["person", "animal", "vehicle", "unknown"]

    dataset_name = str(payload.get("name", config_path.stem)).strip() or config_path.stem
    tag = str(payload.get("tag", "")).strip() or f"{dataset_name}:{dataset_fingerprint(config_path)[:12]}"
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    test_field = payload.get("test")
    test_path = str(test_field).strip() if isinstance(test_field, str) and test_field.strip() else None

    return DatasetConfig(
        name=dataset_name,
        train=train,
        val=val,
        test=test_path,
        labels=label_names,
        tag=tag,
        metadata=metadata,
    )
