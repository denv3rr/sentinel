# Security Policy

## Local-First Security Stance

Sentinel is designed for local-only operation by default.

- API binds to `127.0.0.1` unless explicitly changed.
- No cloud calls are required for core functionality.
- Telemetry is OFF by default.
- Camera credentials are sanitized in logs/API and stored via keychain/encrypted fallback.

## LAN Exposure Warning

Binding to `0.0.0.0` enables LAN access and increases risk.

If you opt in to LAN mode:
- Place Sentinel behind a trusted LAN/VPN.
- Use host firewall rules.
- Avoid exposing Sentinel directly to the public internet.
- Rotate camera credentials and use least privilege accounts.

The UI displays a warning banner when LAN exposure is active.

## Sensitive Data Handling

- RTSP passwords are redacted before log and UI output.
- Export files may include operational metadata and should be handled as sensitive.
- Event media can contain personal data; enforce retention and access controls.

## Vulnerability Reporting

Please report security issues privately to maintainers before public disclosure.