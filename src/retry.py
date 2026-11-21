"""Shared retry helpers: full-jitter backoff and Retry-After parsing."""
from __future__ import annotations

import random
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Mapping


def full_jitter(attempt: int, base: float = 1.0, cap: float = 60.0, rng: random.Random | None = None) -> float:
    """AWS 'full jitter': uniform(0, min(cap, base * 2**attempt))."""
    r = rng or random
    return r.uniform(0, min(cap, base * (2**attempt)))


def parse_retry_after(headers: Mapping[str, str] | None) -> float | None:
    """Retry-After as seconds-int or HTTP-date. Returns seconds to wait, or None."""
    if not headers:
        return None
    val = None
    for k, v in headers.items():
        if k.lower() == "retry-after":
            val = str(v).strip()
            break
    if not val:
        return None
    try:
        return max(0.0, float(val))
    except ValueError:
        pass
    try:
        dt = parsedate_to_datetime(val)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return max(0.0, (dt - datetime.now(timezone.utc)).total_seconds())
    except (TypeError, ValueError):
        return None
