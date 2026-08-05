# dense_evolution_mcp

An MCP server that lets Claude (or any MCP client) drive Dense-Evolution's
Composer kernel directly -- run circuits, compute molecular ground-state
energies, run VQE, get QM/MM forces, run MD trajectories, and apply ZNE
mitigation -- without going through the web UI.

It is a thin adapter, not a reimplementation: every tool calls the same
local FastAPI kernel the published Composer page (`docs/composer.md`) uses
as its execution backend (`local_site/app/server.py`). All real computation
happens in `dense_evolution`/`dashboard_core`, exactly as it does for the
web UI.

## 1. Start the kernel

The kernel is a separate process from this MCP server. From the repo root:

```bash
pip install -e ".[composer]"
dense-evolution serve
# or: python -m local_site.app.server
```

It listens on `http://127.0.0.1:8800` by default. Leave it running.

## 2. Install this server's dependencies

Now part of the package as an extra, so either works:

```bash
pip install -e ".[mcp]"           # if you have the repo checked out
pip install "dense-evolution[mcp]"  # from PyPI, once published
# or, standalone without the extras mechanism:
pip install -r mcp_server/requirements.txt
```

## 3. Register it with your MCP client

**Claude Code:**

```bash
claude mcp add dense_evolution -- dense-evolution mcp
```

`dense-evolution mcp` is the console-script entry point (added in
`dense_evolution/cli.py`, mirrors `dense-evolution serve`); it just calls
this file's `main()`. Running `python /absolute/path/to/mcp_server/server.py`
directly still works too, e.g. if you haven't installed the package.

**Manual `.mcp.json` / `claude_desktop_config.json` entry:**

```json
{
  "mcpServers": {
    "dense_evolution": {
      "command": "python",
      "args": ["/absolute/path/to/mcp_server/server.py"]
    }
  }
}
```

If the kernel is running on a non-default host/port, set
`DENSE_EVOLUTION_KERNEL_URL` (default `http://127.0.0.1:8800`) in the
server's environment.

## Tools

16 tools, one per Composer kernel endpoint:

| Tool | What it does |
|---|---|
| `dense_evolution_health` | Check the kernel is up; version, hostname, free RAM |
| `dense_evolution_system_limits` | Max safe qubit count right now (live RAM-based) |
| `dense_evolution_list_presets` | Built-in example OpenQASM circuits |
| `dense_evolution_list_gates` | Gate palette for the graphical builder |
| `dense_evolution_list_noise_models` | Available Kraus noise channels |
| `dense_evolution_list_molecules` | Catalog molecules + qubit counts |
| `dense_evolution_build_circuit` | Gate-op list -> OpenQASM |
| `dense_evolution_run_circuit` | Run OpenQASM (dense or MPS backend) |
| `dense_evolution_molecule_energy` | Ground-state energy, catalog molecule |
| `dense_evolution_mix_molecules` | Weighted mix of two catalog Hamiltonians |
| `dense_evolution_custom_molecule_energy` | Ground-state energy, arbitrary molecule (<=12 qubits) |
| `dense_evolution_run_vqe` | Real VQE optimization (hardware-efficient or UCCSD) |
| `dense_evolution_qmmm_forces` | Hellmann-Feynman nuclear forces |
| `dense_evolution_md_trajectory` | Velocity-Verlet MD trajectory |
| `dense_evolution_mitigate_zne` | Zero-noise extrapolation, scalar observable |
| `dense_evolution_mitigate_density_matrix` | Zero-noise extrapolation, full density matrix |

## Design notes

- **Images are never inlined.** The kernel returns circuit/histogram/
  Q-sphere/Bloch plots as base64 PNG, meant for a browser `<img>` tag.
  Inlining that into a tool's text response would flood an agent's context
  with a wall of base64 for a picture it can't render inline. When
  `dense_evolution_run_circuit(include_visualizations=true)`, this adapter
  decodes and writes each PNG to `DENSE_EVOLUTION_MCP_IMAGE_DIR` (default
  `~/.dense_evolution_mcp/images`) and returns the file path instead.
- **Large arrays are truncated.** A run on 20+ qubits can return a
  probability/statevector array with over a million entries. This adapter
  returns the top ~25 by magnitude plus a total count, not the raw array.
  The shot-based `counts` histogram is always returned in full since it's
  naturally bounded by the `shots` parameter.
- **Errors are actionable.** If the kernel isn't running, every tool
  returns the exact command to start it instead of a raw connection
  traceback.

## Testing

```bash
python -m py_compile mcp_server/server.py   # syntax check
python mcp_server/server.py                 # runs the stdio server directly
```

With the kernel running, `python -c "import asyncio, server; asyncio.run(server.dense_evolution_health())"`
from inside `mcp_server/` is a quick smoke test without a full MCP client.
