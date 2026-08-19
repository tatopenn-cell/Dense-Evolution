"""
Unit tests for tools/mcp_server/utils/cache.py's TTLCache -- pure logic,
no kernel dependency, unlike tests/integration/test_mcp_server.py.
"""
import time

import pytest

from mcp_server.utils.cache import TTLCache


def test_miss_returns_none():
    cache = TTLCache(ttl_seconds=60.0)
    assert cache.get("missing") is None


def test_set_then_get_round_trips():
    cache = TTLCache(ttl_seconds=60.0)
    cache.set("h2", {"n_qubits": 2})
    assert cache.get("h2") == {"n_qubits": 2}


def test_entry_expires_after_ttl():
    cache = TTLCache(ttl_seconds=0.05)
    cache.set("h2", "value")
    assert cache.get("h2") == "value"
    time.sleep(0.08)
    assert cache.get("h2") is None


def test_failure_is_cached_and_reraised():
    cache = TTLCache(ttl_seconds=60.0, failure_ttl_seconds=60.0)
    exc = RuntimeError("kernel unreachable")
    cache.set_failure("h2", exc)
    with pytest.raises(RuntimeError, match="kernel unreachable"):
        cache.get("h2")


def test_cached_failure_expires_independently_of_ttl():
    cache = TTLCache(ttl_seconds=60.0, failure_ttl_seconds=0.05)
    cache.set_failure("h2", RuntimeError("down"))
    with pytest.raises(RuntimeError):
        cache.get("h2")
    time.sleep(0.08)
    assert cache.get("h2") is None  # expired, not raised


def test_invalidate_one_key():
    cache = TTLCache(ttl_seconds=60.0)
    cache.set("h2", "a")
    cache.set("lih", "b")
    cache.invalidate("h2")
    assert cache.get("h2") is None
    assert cache.get("lih") == "b"


def test_invalidate_all_keys():
    cache = TTLCache(ttl_seconds=60.0)
    cache.set("h2", "a")
    cache.set("lih", "b")
    cache.invalidate()
    assert cache.get("h2") is None
    assert cache.get("lih") is None
