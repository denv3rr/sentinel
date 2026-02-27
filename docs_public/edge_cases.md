# Edge Cases

Sentinel addresses key real-world failure modes:

- Camera disconnect/reconnect: reconnect backoff and worker health status.
- RTSP jitter/timeouts: read failure handling and reconnect attempts.
- Variable FPS/high load: adaptive throttling and frame drops with counters.
- Motion noise (rain/insects/exposure shifts): motion+detection gating and per-label thresholds.
- False positives: cooldown windows, per-label confidence thresholds, zones.
- Storage pressure: retention policies by age and size.
- Clock shifts/DST: both wall clock and monotonic timestamps stored.
- Safe shutdown/crash recovery: SQLite WAL + media path checks.