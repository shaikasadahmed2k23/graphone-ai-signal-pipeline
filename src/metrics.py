"""Process-wide run metrics. Swap for Prometheus/StatsD by replacing inc()."""
from __future__ import annotations

import threading
from collections import Counter


class Metrics:
    def __init__(self) -> None:
        self._c: Counter[str] = Counter()
        self._lock = threading.Lock()

    def inc(self, name: str, n: int = 1) -> None:
        with self._lock:
            self._c[name] += n

    def get(self, name: str) -> int:
        return self._c.get(name, 0)

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(sorted(self._c.items()))

    def reset(self) -> None:
        with self._lock:
            self._c.clear()

    def report(self) -> str:
        snap = self.snapshot()
        if not snap:
            return "(no metrics)"
        w = max(len(k) for k in snap)
        return "\n".join(f"  {k:<{w}}  {v}" for k, v in snap.items())


metrics = Metrics()
