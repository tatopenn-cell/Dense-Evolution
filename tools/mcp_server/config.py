"""Central configuration for the dense_evolution_mcp adapter: the MCP
tool-annotation presets shared across every tool, plus DEFAULT_TIMEOUT.

Two settings that conceptually belong here are deliberately NOT here,
because the test suite mutates them directly and that mutation has to
reach the specific function that reads them as its own module global (a
`from .config import X` copy elsewhere would silently stop seeing the
mutation): KERNEL_URL/_TEST_TRANSPORT live in client.py (see its
docstring), and IMAGE_OUTPUT_DIR/IMAGE_MAX_FILES live in utils/images.py
(see its docstring) since that's where _save_png/_prune_old_images moved
in Phase 2 of the refactor (prog.txt Sezione 3).
"""

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
