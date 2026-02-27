from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import SplitResult, urlsplit, urlunsplit

from cryptography.fernet import Fernet

RTSP_PASSWORD_RE = re.compile(r"(rtsp://[^:@/]+:)([^@/]+)(@)", re.IGNORECASE)
PASSWORD_PAIR_RE = re.compile(r"(password\s*[=:]\s*)([^\s,;]+)", re.IGNORECASE)
TOKEN_RE = re.compile(r"(token\s*[=:]\s*)([^\s,;]+)", re.IGNORECASE)
CAMERA_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def sanitize_rtsp_url(url: str) -> str:
    try:
        parts: SplitResult = urlsplit(url)
        if parts.scheme.lower() not in {"rtsp", "rtsps"}:
            return url
        hostname = parts.hostname or ""
        user = parts.username
        redacted_user = user if user else "user"
        port = f":{parts.port}" if parts.port else ""
        netloc = f"{redacted_user}:***@{hostname}{port}" if user or parts.password else f"{hostname}{port}"
        return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))
    except Exception:
        return RTSP_PASSWORD_RE.sub(r"\1***\3", url)


def resolve_path_within_base(base_dir: Path, untrusted_path: str | Path) -> Path | None:
    try:
        resolved_base = base_dir.resolve()
        target = (resolved_base / Path(untrusted_path)).resolve()
    except Exception:
        return None

    try:
        target.relative_to(resolved_base)
    except ValueError:
        return None
    return target


def validate_camera_id(camera_id: str) -> str:
    value = str(camera_id)
    if not CAMERA_ID_RE.fullmatch(value):
        raise ValueError("Invalid camera id")
    return value


def validate_rtsp_url(source: str, *, allow_redacted_password: bool = False) -> str:
    value = str(source)
    if not value:
        raise ValueError("Invalid RTSP source")
    if any(ch.isspace() for ch in value):
        raise ValueError("Invalid RTSP source")

    parts: SplitResult = urlsplit(value)
    if parts.scheme.lower() not in {"rtsp", "rtsps"}:
        raise ValueError("Invalid RTSP source")
    if not parts.hostname:
        raise ValueError("Invalid RTSP source")
    if parts.fragment:
        raise ValueError("Invalid RTSP source")
    if "\\" in parts.path:
        raise ValueError("Invalid RTSP source")

    try:
        _ = parts.port
    except ValueError as exc:
        raise ValueError("Invalid RTSP source") from exc

    password = parts.password or ""
    if password == "***" and not allow_redacted_password:
        raise ValueError("Invalid RTSP source")
    if any(ch.isspace() for ch in password):
        raise ValueError("Invalid RTSP source")

    username = parts.username or ""
    if any(ch.isspace() for ch in username):
        raise ValueError("Invalid RTSP source")

    return value


def redact_secrets(text: str) -> str:
    text = RTSP_PASSWORD_RE.sub(r"\1***\3", text)
    text = PASSWORD_PAIR_RE.sub(r"\1***", text)
    text = TOKEN_RE.sub(r"\1***", text)
    return text


def scrub_sensitive(obj: Any) -> Any:
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for key, value in obj.items():
            lowered = key.lower()
            if "password" in lowered or "token" in lowered or "secret" in lowered:
                out[key] = "***"
            elif lowered == "source" and isinstance(value, str):
                out[key] = sanitize_rtsp_url(value)
            else:
                out[key] = scrub_sensitive(value)
        return out
    if isinstance(obj, list):
        return [scrub_sensitive(v) for v in obj]
    if isinstance(obj, str):
        return redact_secrets(obj)
    return obj


@dataclass
class SecretReference:
    provider: str
    ref: str

    def as_dict(self) -> dict[str, str]:
        return {"provider": self.provider, "ref": self.ref}


class SecretStore:
    def __init__(self, data_dir: Path, service_name: str = "sentinel") -> None:
        self.service_name = service_name
        config_dir = data_dir / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        self._secrets_file = config_dir / "secrets.enc.json"
        self._key_file = config_dir / "secrets.key"

    def _load_key(self) -> bytes:
        if self._key_file.exists():
            return self._key_file.read_bytes()
        key = Fernet.generate_key()
        self._key_file.write_bytes(key)
        try:
            os.chmod(self._key_file, 0o600)
        except PermissionError:
            pass
        return key

    def _read_map(self) -> dict[str, str]:
        if not self._secrets_file.exists():
            return {}
        return json.loads(self._secrets_file.read_text(encoding="utf-8"))

    def _write_map(self, payload: dict[str, str]) -> None:
        self._secrets_file.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    def store(self, name: str, value: str) -> SecretReference:
        try:
            import keyring  # type: ignore

            keyring.set_password(self.service_name, name, value)
            return SecretReference(provider="keyring", ref=name)
        except Exception:
            pass

        key = self._load_key()
        fernet = Fernet(key)
        payload = self._read_map()
        payload[name] = fernet.encrypt(value.encode("utf-8")).decode("utf-8")
        self._write_map(payload)
        return SecretReference(provider="encrypted_file", ref=name)

    def get(self, reference: dict[str, str] | SecretReference | None) -> str | None:
        if reference is None:
            return None
        if isinstance(reference, SecretReference):
            provider = reference.provider
            ref = reference.ref
        else:
            provider = reference.get("provider", "")
            ref = reference.get("ref", "")

        if not ref:
            return None

        if provider == "keyring":
            try:
                import keyring  # type: ignore

                return keyring.get_password(self.service_name, ref)
            except Exception:
                return None

        if provider == "encrypted_file":
            payload = self._read_map()
            token = payload.get(ref)
            if not token:
                return None
            key = self._load_key()
            fernet = Fernet(key)
            try:
                return fernet.decrypt(token.encode("utf-8")).decode("utf-8")
            except Exception:
                return None

        return None
