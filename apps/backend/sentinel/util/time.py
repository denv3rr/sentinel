from __future__ import annotations

import datetime as dt
import time
from zoneinfo import ZoneInfo


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def now_utc_iso() -> str:
    return now_utc().isoformat()


def now_local() -> dt.datetime:
    return dt.datetime.now().astimezone()


def now_local_iso() -> str:
    return now_local().isoformat()


def monotonic_ns() -> int:
    return time.monotonic_ns()


def parse_iso8601(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    parsed = dt.datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def as_local_timezone(utc_time: dt.datetime) -> dt.datetime:
    if utc_time.tzinfo is None:
        utc_time = utc_time.replace(tzinfo=dt.timezone.utc)
    local_tz = dt.datetime.now().astimezone().tzinfo or ZoneInfo("UTC")
    return utc_time.astimezone(local_tz)