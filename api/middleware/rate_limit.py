"""In-process fixed-window request limiter."""
from __future__ import annotations

import threading
import time
from collections import defaultdict


WINDOW = 60
LIMIT = 100
_requests: dict[str, list[float]] = defaultdict(list)
_lock = threading.Lock()


def check_rate_limit(client: str) -> bool:
    """Record one request and return whether the client remains allowed."""
    now = time.monotonic()
    with _lock:
        recent = [t for t in _requests[client] if now - t < WINDOW]
        if len(recent) >= LIMIT:
            _requests[client] = recent
            return False
        recent.append(now)
        _requests[client] = recent
        return True


def reset_rate_limits() -> None:
    """Clear limiter state, primarily for controlled tests and workers."""
    with _lock:
        _requests.clear()
