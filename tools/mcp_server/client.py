"""HTTP client layer for the dense_evolution_mcp adapter: the shared
httpx.AsyncClient, the low-level _request helper every tool calls, and the
@catch_errors decorator that replaces the try/except-return-_handle_error(e)
boilerplate every tool used to repeat individually.

KERNEL_URL and _TEST_TRANSPORT live here (not config.py) specifically
because tests/integration/test_mcp_server.py mutates them directly
(`from mcp_server import client as mcp_client; mcp_client.KERNEL_URL = ...`)
to exercise the unreachable-kernel and in-process-ASGI-transport paths --
_get_client/_request must see those mutations as their OWN module globals,
which only works if the value's single source of truth lives in the same
module that reads it. Moving _TEST_TRANSPORT into a proper pytest fixture
instead of a raw mutable global is planned for Phase 4 (see prog.txt
Sezione 3.3), not done here.
"""
import functools
import os
from typing import Optional

import httpx

from .config import DEFAULT_TIMEOUT

KERNEL_URL = os.environ.get("DENSE_EVOLUTION_KERNEL_URL", "http://127.0.0.1:8800").rstrip("/")

# Set by tests to route through httpx.ASGITransport straight into the real,
# in-process local_site.app.server.app (see tests/integration/test_mcp_server.py) --
# None here means "use a real TCP connection to KERNEL_URL", unchanged from
# before this existed. This is the one seam in the whole adapter: it lets
# tests exercise every tool function against the real FastAPI kernel (real
# DenseSVSimulator, real PennyLane Hamiltonians -- no mocked physics) without
# a live subprocess bound to a real port in CI.
_TEST_TRANSPORT = None

# Lazily-created, reused across calls -- _request used to open a fresh
# httpx.AsyncClient (TCP handshake to the kernel) per tool call, real
# overhead for an MCP session that calls many tools in a row. Keyed on
# (_TEST_TRANSPORT identity, KERNEL_URL): production code never changes
# KERNEL_URL after import (it's a plain module-level constant read once
# from os.environ), but the test suite mutates both KERNEL_URL and
# _TEST_TRANSPORT directly to exercise the unreachable-kernel path, and its
# autouse fixture swaps _TEST_TRANSPORT to a fresh ASGITransport before
# every test and back to None after -- a client cached against a stale
# transport or base_url must not be reused across either kind of swap.
_shared_client: Optional[httpx.AsyncClient] = None
_shared_client_key = None  # (transport identity, base_url) the cached client was built with


def _get_client() -> httpx.AsyncClient:
    global _shared_client, _shared_client_key
    current_key = (id(_TEST_TRANSPORT), KERNEL_URL)
    if _shared_client is None or _shared_client_key != current_key:
        # Client-level timeout is just the fallback ceiling -- every real
        # call site passes its own timeout= to _request, sized to that
        # endpoint's actual expected cost (a health check and a 10-minute
        # VQE run have nothing in common).
        _shared_client = httpx.AsyncClient(base_url=KERNEL_URL, transport=_TEST_TRANSPORT, timeout=DEFAULT_TIMEOUT)
        _shared_client_key = current_key
    return _shared_client


async def _request(method: str, path: str, timeout: float = DEFAULT_TIMEOUT, **kwargs) -> dict:
    """Reusable request helper for every tool. `timeout` (seconds) should be
    sized to the specific endpoint's real expected cost -- see each tool's
    own call site."""
    client = _get_client()
    try:
        resp = await client.request(method, path, timeout=timeout, **kwargs)
    except httpx.ConnectError:
        raise RuntimeError(
            f"Dense Evolution kernel not reachable at {KERNEL_URL}. Start it with "
            "`dense-evolution serve` (or `python -m local_site.app.server` from the "
            "repo root), then retry. Use dense_evolution_health to check connectivity."
        )
    except httpx.TimeoutException:
        raise RuntimeError(
            f"Request to the Dense Evolution kernel timed out after {timeout:g}s -- the "
            "simulation may be too large or slow for this request (e.g. high qubit "
            "count, many VQE iterations, or a long MD trajectory). Try reducing its size."
        )
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        raise RuntimeError(f"Dense Evolution kernel returned HTTP {resp.status_code}: {detail}")
    return resp.json()


def _handle_error(e: Exception) -> str:
    """Consistent error formatting across all tools."""
    return f"Error: {e}"


def catch_errors(func):
    """Wraps an async tool function so any exception raised anywhere in its
    body becomes the same "Error: ..." string every tool used to return
    from its own try/except -- replaces ~20 repeated
    `try: ... except Exception as e: return _handle_error(e)` blocks with
    one decorator.

    A strict superset of the old behavior for the handful of tools whose
    try block only wrapped part of the function -- e.g.
    dense_evolution_run_circuit's post-request image saving/truncation used
    to sit OUTSIDE its try block, so an exception there would have
    propagated uncaught; now it's caught too.

    Tools that do their own internal per-item exception handling
    (dense_evolution_energy_scan, dense_evolution_wormhole_scan -- a single
    failed point shouldn't abort the whole batch) are unaffected by this:
    it only adds an outer safety net around exceptions those inner
    try/excepts don't already catch (e.g. a bug in the post-loop
    aggregation code), it does not change their partial-failure behavior.
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            return _handle_error(e)
    return wrapper
