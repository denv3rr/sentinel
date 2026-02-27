# Camera Setup

## Webcam

- Kind: `webcam`
- Source: index (`0`, `1`, ...)

## RTSP

- Kind: `rtsp`
- Source: full RTSP URL
- Credentials are stored securely and redacted from logs/UI.

## ONVIF (Best Effort)

- Use Cameras page discovery.
- Sentinel probes WS-Discovery and lists endpoint hosts.
- Candidate RTSP URLs are suggested for quick testing.
- Vendor-specific ONVIF profile parsing is limited in v1.

## Multi-Camera

- Add multiple cameras from Cameras page.
- Each camera has independent thresholds, cooldown, zones, and enable/disable state.