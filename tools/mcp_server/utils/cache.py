"""A minimal in-memory TTL cache, used to avoid re-fetching the
(effectively static) molecule catalog on every tool call that needs to
resolve a molecule name, while still eventually picking up a real catalog
change (e.g. after a kernel restart with a different build) instead of
caching forever like the pre-Phase-2 code did (a plain dict, reset only by
the test suite, never by production code). Also caches FAILURES for a
short TTL, so a tight loop of tool calls made while the kernel is down
doesn't each wait out their own full connection/timeout error -- see
`set_failure`/`get`'s re-raise behavior below.
"""
import time
from typing import Any, Optional


class TTLCache:
    def __init__(self, ttl_seconds: float = 300.0, failure_ttl_seconds: float = 10.0):
        self._ttl = ttl_seconds
        self._failure_ttl = failure_ttl_seconds
        # key -> (expires_at, value_or_exception, is_failure)
        self._store: dict[str, tuple[float, Any, bool]] = {}

    def get(self, key: str):
        """Returns the cached value, or None if missing/expired. Re-raises
        a cached failure (so callers see the same error shape they would
        from a fresh call) instead of returning it as if it were a value."""
        entry = self._store.get(key)
        if entry is None:
            return None
        expires_at, value, is_failure = entry
        if time.monotonic() >= expires_at:
            del self._store[key]
            return None
        if is_failure:
            raise value
        return value

    def set(self, key: str, value: Any) -> None:
        self._store[key] = (time.monotonic() + self._ttl, value, False)

    def set_failure(self, key: str, exc: Exception) -> None:
        self._store[key] = (time.monotonic() + self._failure_ttl, exc, True)

    def invalidate(self, key: Optional[str] = None) -> None:
        """Clears one key, or the whole cache if key is None."""
        if key is None:
            self._store.clear()
        else:
            self._store.pop(key, None)
