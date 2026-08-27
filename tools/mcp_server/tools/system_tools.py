"""Tools: kernel status and catalogs (presets, gates, noise models,
molecules). Registered against the shared `mcp` instance created in
server.py -- see that module's docstring for why importing `mcp` back
from there (rather than the other way around) is safe despite looking
circular."""
import json

from .. import client
from ..client import _request, catch_errors
from ..config import READ_ONLY_IDEMPOTENT
from ..models import ListMoleculesInput
from ..molecules import _get_annotated_molecule_catalog, _molecule_catalog_cache
from ..server import mcp
from ..utils import images as _images


@mcp.tool(name="dense_evolution_health", annotations={"title": "Check Dense Evolution kernel status", **READ_ONLY_IDEMPOTENT})
@catch_errors
async def dense_evolution_health() -> str:
    """Check whether the local Dense Evolution Composer kernel is running and reachable.

    Always call this first if unsure whether the kernel is up -- every other
    tool in this server depends on it. Returns the kernel's dense_evolution
    version, hostname, and free/total RAM (useful to sanity-check qubit-count
    limits before requesting a large simulation).

    Returns:
        str: JSON with {status, dense_evolution_version, hostname,
        total_ram_gb, available_ram_gb, ram_percent_free}, or an
        "Error: ..." string if the kernel is not running.
    """
    return json.dumps(await _request("GET", "/api/health", timeout=5.0), indent=2)


@mcp.tool(name="dense_evolution_system_limits", annotations={"title": "Get max safe qubit count", **READ_ONLY_IDEMPOTENT})
@catch_errors
async def dense_evolution_system_limits() -> str:
    """Get the maximum qubit count this machine can currently simulate with
    a dense statevector, computed from actual free RAM right now (not a
    fixed constant) -- call before requesting a large `dense_evolution_run_circuit`.

    Returns:
        str: JSON describing the current safe qubit ceiling for the dense backend.
    """
    return json.dumps(await _request("GET", "/api/system_limits", timeout=5.0), indent=2)


@mcp.tool(name="dense_evolution_kernel_status", annotations={"title": "Inspect this MCP adapter's own local state", **READ_ONLY_IDEMPOTENT})
@catch_errors
async def dense_evolution_kernel_status() -> str:
    """Report this MCP adapter's own local configuration and diagnostics --
    distinct from dense_evolution_health, which only proxies the kernel's
    own /api/health. Useful for debugging the adapter process itself:
    which kernel URL it's pointed at, whether that kernel currently
    answers, how many circuit/histogram/Q-sphere PNGs have piled up on
    disk (see the module docstring on why images are saved to disk
    instead of inlined), and how warm the molecule-catalog cache is.

    Returns:
        str: JSON with {kernel_url, kernel_reachable, image_output_dir,
        image_count, image_max_files, molecule_cache_entries}.
    """
    try:
        await _request("GET", "/api/health", timeout=5.0)
        reachable = True
    except Exception:
        reachable = False

    image_dir = _images.IMAGE_OUTPUT_DIR
    image_count = len(list(image_dir.glob("*.png"))) if image_dir.exists() else 0

    return json.dumps({
        "kernel_url": client.KERNEL_URL,
        "kernel_reachable": reachable,
        "image_output_dir": str(image_dir),
        "image_count": image_count,
        "image_max_files": _images.IMAGE_MAX_FILES,
        "molecule_cache_entries": len(_molecule_catalog_cache),
    }, indent=2)


@mcp.tool(name="dense_evolution_list_presets", annotations={"title": "List preset OpenQASM circuits", **READ_ONLY_IDEMPOTENT})
@catch_errors
async def dense_evolution_list_presets() -> str:
    """List the built-in example OpenQASM circuits bundled with the Composer
    (e.g. Bell state, GHZ, QFT) -- useful as ready-made input for
    `dense_evolution_run_circuit` without writing QASM by hand.

    Returns:
        str: JSON mapping preset names to their OpenQASM source text.
    """
    return json.dumps(await _request("GET", "/api/presets", timeout=5.0), indent=2)


@mcp.tool(name="dense_evolution_list_gates", annotations={"title": "List available quantum gates", **READ_ONLY_IDEMPOTENT})
@catch_errors
async def dense_evolution_list_gates() -> str:
    """List every gate the graphical circuit builder (and therefore
    `dense_evolution_build_circuit`) supports, with their display metadata.

    Returns:
        str: JSON gate palette (name, symbol, qubit arity, etc. per gate).
    """
    return json.dumps(await _request("GET", "/api/palette", timeout=5.0), indent=2)


@mcp.tool(name="dense_evolution_list_noise_models", annotations={"title": "List available noise models", **READ_ONLY_IDEMPOTENT})
@catch_errors
async def dense_evolution_list_noise_models() -> str:
    """List the real Kraus-channel noise models available for
    `dense_evolution_run_circuit` and the mitigation tools (e.g.
    depolarizing, amplitude damping, bit-flip).

    Returns:
        str: JSON mapping noise model names to their parameters/description.
    """
    return json.dumps(await _request("GET", "/api/noise_models", timeout=5.0), indent=2)


@mcp.tool(name="dense_evolution_list_molecules", annotations={"title": "List catalog molecules", **READ_ONLY_IDEMPOTENT})
@catch_errors
async def dense_evolution_list_molecules(params: ListMoleculesInput) -> str:
    """List every molecule in the built-in Hartree-Fock catalog, each with
    its real qubit count under the requested mapping. Use this to find valid
    `name` values for `dense_evolution_molecule_energy`, `_mix_molecules`,
    `_run_vqe`, `_qmmm_forces`, `_md_trajectory`, and `_energy_scan`.

    Each entry includes a short `id` (e.g. "H2", "LiH", "HeH+") derived from
    the catalog's full descriptive name -- pass either form to any tool that
    takes a molecule `name`; the short id is easier to copy exactly across
    multiple tool calls than the full "H2 (Idrogeno) - R = 0.7414 A
    [equilibrio reale]"-style string.

    Args:
        params (ListMoleculesInput): mapping -- 'jordan_wigner' (default) or 'bravyi_kitaev'.

    Returns:
        str: JSON list of {id, full_name, symbols, geometry, charge, n_qubits} per molecule.
    """
    return json.dumps(await _get_annotated_molecule_catalog(params.mapping), indent=2)
