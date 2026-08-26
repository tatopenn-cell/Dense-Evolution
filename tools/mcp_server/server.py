#!/usr/bin/env python3
"""
MCP server for Dense-Evolution's Composer API (dense_evolution_mcp).

This is a thin adapter, not a reimplementation: every tool here calls the
same local FastAPI kernel the published Composer web page talks to
(local_site/app/server.py, started with `dense-evolution serve`, listening
on http://127.0.0.1:8800 by default). All computation -- circuit
simulation, VQE, molecular Hamiltonians, QM/MM forces, ZNE mitigation --
happens inside dense_evolution/dashboard_core exactly as it does for the
web UI; this file only exposes those same endpoints as MCP tools so an
agent can drive them directly instead of a browser.

Requires the kernel to be running separately:
    pip install dense-evolution[composer]
    dense-evolution serve
(or `python -m local_site.app.server` from the repo root)

Override the kernel URL with the DENSE_EVOLUTION_KERNEL_URL env var if it's
not on the default host/port.

Images (circuit diagrams, histograms, Q-sphere, Bloch vectors) come back
from the kernel as base64 PNGs meant for a browser <img> tag -- inlining
that into a tool's text response would flood an agent's context with a
wall of base64 for a picture it can't even see inline. Instead this
adapter decodes and writes each one to DENSE_EVOLUTION_MCP_IMAGE_DIR
(default ~/.dense_evolution_mcp/images) and returns the file path, which
Claude Code (or any agent with file access) can open directly. Large
numeric arrays (statevector, probabilities) are similarly truncated to
their most significant entries rather than dumped in full -- the kernel's
own response can be tens of thousands of floats for a 20+ qubit circuit.

Structure (prog.txt Sezione 3, now complete): settings live in config.py,
the HTTP client + error handling in client.py, the Pydantic input schemas
in models.py (Phase 1); image saving/truncation/molecule-catalog caching
in utils/ and molecules.py (Phase 2); the 21 tools themselves, split by
topic, in tools/ (Phase 3, this file). This file's only job is to create
the MCPServer instance, import each tools/*.py module so its `@mcp.tool`
decorators register against it, re-export every tool function (so
`from mcp_server.server import dense_evolution_health` keeps working for
existing callers, including this repo's own test suite), and provide the
`main()` console-script entry point.

`mcp` is created here BEFORE the `from .tools....` imports below, and
every tools/*.py module does `from ..server import mcp` -- Python resolves
this correctly despite looking circular: by the time those imports run,
`mcp_server.server` is already in `sys.modules` (registered before this
file's body starts executing) with `mcp` already assigned, so each
submodule's `from ..server import mcp` finds it immediately. Reordering
the `mcp = MCPServer(...)` line to after the tool imports would break this.
"""
from mcp.server import MCPServer

mcp = MCPServer("dense_evolution_mcp")

from .tools.system_tools import (  # noqa: E402
    dense_evolution_health, dense_evolution_list_gates, dense_evolution_list_molecules,
    dense_evolution_list_noise_models, dense_evolution_list_presets, dense_evolution_system_limits,
)
from .tools.circuit_tools import dense_evolution_build_circuit, dense_evolution_run_circuit  # noqa: E402
from .tools.chemistry_tools import (  # noqa: E402
    dense_evolution_custom_molecule_energy, dense_evolution_energy_scan, dense_evolution_md_trajectory,
    dense_evolution_mix_molecules, dense_evolution_molecule_energy, dense_evolution_qmmm_forces,
    dense_evolution_run_vqe,
)
from .tools.mitigation_tools import (  # noqa: E402
    dense_evolution_mitigate_density_matrix, dense_evolution_mitigate_zne, dense_evolution_vector_healing,
)
from .tools.wormhole_tools import (  # noqa: E402
    dense_evolution_wormhole_scan, dense_evolution_wormhole_select_instance, dense_evolution_wormhole_teleportation,
)
from .tools.noise_tools import (  # noqa: E402
    dense_evolution_cosmic_ray_burst, dense_evolution_oscillating_noise, dense_evolution_density_matrix_channel,
)


def main():
    """Console-script entry point (`dense-evolution mcp`, see
    dense_evolution/cli.py) -- identical to running this file directly.
    stdio transport: this process is meant to be launched by an MCP
    client (Claude Code, Claude Desktop, ...) as a subprocess, not run
    standalone in a terminal."""
    mcp.run()  # pragma: no cover -- blocks on the real stdio transport loop


if __name__ == "__main__":
    main()
