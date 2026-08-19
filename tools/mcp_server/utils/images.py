"""Image saving, rotating cleanup, and metadata for the dense_evolution_mcp
adapter. Images (circuit diagrams, histograms, Q-sphere, Bloch vectors)
come back from the kernel as base64 PNGs; see the top-level server.py
module docstring for why they're written to disk instead of inlined.

IMAGE_OUTPUT_DIR/IMAGE_MAX_FILES are defined HERE rather than in config.py,
for the same reason KERNEL_URL/_TEST_TRANSPORT live in client.py rather
than config.py (see client.py's own docstring): the test suite mutates
both directly (`mcp_images.IMAGE_OUTPUT_DIR = tmp_path`), and that
mutation only works if the value's single source of truth lives in the
same module that reads it -- _prune_old_images/_save_png, which moved here
in Phase 2 of the refactor (see prog.txt Sezione 3).

Each saved PNG also gets an optional sidecar `<name>_<ts>.json` with
whatever metadata the caller passes in (tool name, qasm, seed, ...) --
BUG FIX: previously a saved image's filename (`circuit_<timestamp>.png`)
carried no way to trace it back to the circuit/tool call that produced it,
so an agent (or a human) looking at ~/.dense_evolution_mcp/images later
had no way to tell which image was which beyond the generic `name` prefix
shared by every call of the same kind.
"""
import base64
import json
import os
import time
from pathlib import Path
from typing import Optional

IMAGE_OUTPUT_DIR = Path(
    os.environ.get("DENSE_EVOLUTION_MCP_IMAGE_DIR", str(Path.home() / ".dense_evolution_mcp" / "images"))
)
# Every include_visualizations=True call writes a new timestamped PNG with
# no cleanup -- a long-running MCP session (or an agent looping over many
# circuits) grows this directory without bound. Cap it to the most
# recently written files; 0 or negative disables pruning entirely.
IMAGE_MAX_FILES = int(os.environ.get("DENSE_EVOLUTION_MCP_IMAGE_MAX_FILES", "500"))


def _prune_old_images() -> None:
    """Keep at most IMAGE_MAX_FILES PNGs in IMAGE_OUTPUT_DIR, deleting the
    oldest by mtime first (and their metadata sidecar, if any). Read at
    call time (not module import) so tests that monkeypatch
    IMAGE_OUTPUT_DIR/IMAGE_MAX_FILES take effect."""
    if IMAGE_MAX_FILES <= 0:
        return
    files = sorted(IMAGE_OUTPUT_DIR.glob("*.png"), key=lambda p: p.stat().st_mtime)
    excess_count = len(files) - IMAGE_MAX_FILES
    for path in files[:excess_count]:
        path.unlink(missing_ok=True)
        path.with_suffix(".json").unlink(missing_ok=True)


def _save_png(b64_png: Optional[str], name: str, metadata: Optional[dict] = None) -> Optional[str]:
    """Decode a base64 PNG from the kernel and write it to disk, returning
    the path instead of the raw base64 -- see module docstring. If
    `metadata` is given, also writes a `<same-stem>.json` sidecar next to
    the PNG (e.g. {"tool": "dense_evolution_run_circuit", "qasm": ...,
    "seed": ..., "n_qubits": ...})."""
    if not b64_png:
        return None
    IMAGE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = IMAGE_OUTPUT_DIR / f"{name}_{int(time.time() * 1000)}.png"
    path.write_bytes(base64.b64decode(b64_png))
    if metadata is not None:
        path.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    _prune_old_images()
    return str(path)
