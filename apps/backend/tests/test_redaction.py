from __future__ import annotations

from sentinel.util.security import redact_secrets, sanitize_rtsp_url, scrub_sensitive


def test_rtsp_password_redaction() -> None:
    raw = "rtsp://admin:superSecret@192.168.1.50:554/stream1"
    safe = sanitize_rtsp_url(raw)
    assert "superSecret" not in safe
    assert "***" in safe


def test_generic_secret_redaction() -> None:
    text = "password=myPass token=abc123 rtsp://u:p@cam"
    redacted = redact_secrets(text)
    assert "myPass" not in redacted
    assert "abc123" not in redacted
    assert "rtsp://u:***@cam" in redacted


def test_scrub_sensitive_recursive() -> None:
    payload = {
        "source": "rtsp://u:p@cam.local/live",
        "password": "abc",
        "nested": {"token": "123", "ok": "value"},
    }
    scrubbed = scrub_sensitive(payload)
    assert scrubbed["password"] == "***"
    assert scrubbed["nested"]["token"] == "***"
    assert "p@" not in scrubbed["source"]