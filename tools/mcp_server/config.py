"""Central configuration for the dense_evolution_mcp adapter: environment-
derived settings and the MCP tool-annotation presets shared across every
tool.

KERNEL_URL is deliberately NOT here despite conceptually belonging
alongside these -- it's defined directly in client.py instead, since
tests/integration/test_mcp_server.py mutates it directly
(`mcp_client.KERNEL_URL = "http://127.0.0.1:1"`, to exercise the
unreachable-kernel path) and that mutation has to be visible to
_get_client/_request, which read it as their own module global. A
`from .config import KERNEL_URL` copy here would silently stop seeing that
mutation the moment client.py became a separate module from server.py --
see client.py's own module docstring for the full explanation.
"""
import os
from pathlib import Path

IMAGE_OUTPUT_DIR = Path(
    os.environ.get("DENSE_EVOLUTION_MCP_IMAGE_DIR", str(Path.home() / ".dense_evolution_mcp" / "images"))
)
# Every include_visualizations=True call writes a new timestamped PNG with
# no cleanup -- a long-running MCP session (or an agent looping over many
# circuits) grows this directory without bound. Cap it to the most
# recently written files; 0 or negative disables pruning entirely.
IMAGE_MAX_FILES = int(os.environ.get("DENSE_EVOLUTION_MCP_IMAGE_MAX_FILES", "500"))

READ_ONLY_IDEMPOTENT = {
    "readOnlyHint": True,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": False,
}
# Circuit/VQE/MD runs don't mutate any stored resource -- nothing server-side
# persists between calls -- so they're "read-only" in the MCP sense too, but
# re-running with the same seed can differ slightly run-to-run (floating
# point reduction order in the linear-algebra backend), so idempotentHint
# is left False for anything that actually executes a simulation.
COMPUTE = {
    "readOnlyHint": True,
    "destructiveHint": False,
    "idempotentHint": False,
    "openWorldHint": False,
}

DEFAULT_TIMEOUT = 60.0  # fallback for any call site that doesn't pass timeout= explicitly
