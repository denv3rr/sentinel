from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class ReconnectState:
    failures: int = 0
    max_failures_before_reconnect: int = 5
    min_backoff_seconds: float = 0.5
    max_backoff_seconds: float = 8.0

    def register_failure(self) -> float:
        self.failures += 1
        return min(self.max_backoff_seconds, self.min_backoff_seconds * (2 ** max(0, self.failures - 1)))

    def should_reconnect(self) -> bool:
        return self.failures >= self.max_failures_before_reconnect

    def sleep_backoff(self) -> None:
        time.sleep(self.register_failure())

    def reset(self) -> None:
        self.failures = 0