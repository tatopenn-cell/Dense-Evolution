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

Structure (Phase 1 of the modularization in prog.txt Sezione 3): settings
live in config.py, the HTTP client + error handling in client.py, and the
Pydantic input schemas in models.py. This file keeps the MCPServer
instance, the tool functions themselves, and the small helpers
(image saving/truncation, molecule-name resolution) that only these tools
use. Splitting the tools themselves into per-topic modules
(tools/system_tools.py, tools/circuit_tools.py, ...) is Phase 3, not done
here.
"""

import asyncio
import base64
import json
import time
from pathlib import Path
from typing import Optional

# mcp>=2.0.0 renamed FastMCP -> MCPServer and moved it from
# mcp.server.fastmcp to mcp.server.mcpserver (re-exported at mcp.server
# directly). Everything else -- @mcp.tool(name=..., annotations={...}) with
# a plain dict, Pydantic model params, mcp.run() -- is unchanged between
# the two; this was the only line that needed to move.
from mcp.server import MCPServer

from .client import _request, catch_errors
from .config import COMPUTE, IMAGE_MAX_FILES, IMAGE_OUTPUT_DIR, READ_ONLY_IDEMPOTENT
from .models import (
    BuildCircuitInput, CustomMoleculeInput, EnergyScanInput, ListMoleculesInput,
    MdTrajectoryInput, MitigateDensityMatrixInput, MitigateZneInput, MixMoleculesInput,
    MoleculeEnergyInput, QmmmForcesInput, RunCircuitInput, RunVqeInput, VectorHealingInput,
    WormholeScanInput, WormholeSelectInstanceInput, WormholeTeleportationInput,
)

mcp = MCPServer("dense_evolution_mcp")


# --------------------------------------------------------------------------
# Shared utilities (image saving/pruning, truncation) -- only these tools
# use them, so they stay here rather than moving to config/client/models.
# --------------------------------------------------------------------------

def _prune_old_images() -> None:
    """Keep at most IMAGE_MAX_FILES PNGs in IMAGE_OUTPUT_DIR, deleting the
    oldest by mtime first. Read at call time (not module import) so tests
    that monkeypatch IMAGE_OUTPUT_DIR/IMAGE_MAX_FILES take effect."""
    if IMAGE_MAX_FILES <= 0:
        return
    files = sorted(IMAGE_OUTPUT_DIR.glob("*.png"), key=lambda p: p.stat().st_mtime)
    excess_count = len(files) - IMAGE_MAX_FILES
    for path in files[:excess_count]:
        path.unlink(missing_ok=True)


def _save_png(b64_png: Optional[str], name: str) -> Optional[str]:
    """Decode a base64 PNG from the kernel and write it to disk, returning
    the path instead of the raw base64 -- see module docstring."""
    if not b64_png:
        return None
    IMAGE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = IMAGE_OUTPUT_DIR / f"{name}_{int(time.time() * 1000)}.png"
    path.write_bytes(base64.b64decode(b64_png))
    _prune_old_images()
    return str(path)


def _truncate_statevector(rows: list, top_k: int = 25) -> dict:
    """Full statevectors can be thousands of entries; agents almost always
    care about the dominant amplitudes, not the full dense array."""
    sorted_rows = sorted(rows, key=lambda r: -r["abs"])
    return {
        "total_nonzero_amplitudes": len(rows),
        "shown": min(top_k, len(rows)),
        "top_amplitudes_by_magnitude": sorted_rows[:top_k],
    }


def _truncate_probabilities(probs: list, top_k: int = 25) -> dict:
    indexed = sorted(enumerate(probs), key=lambda t: -t[1])[:top_k]
    return {
        "total_basis_states": len(probs),
        "shown": min(top_k, len(probs)),
        "top_states_by_probability": [{"index": i, "probability": p} for i, p in indexed],
    }


# --------------------------------------------------------------------------
# Molecule name resolution: short ids <-> the kernel's full catalog keys
# --------------------------------------------------------------------------
#
# The kernel's own catalog keys are long, human-readable strings, e.g.
# "H2 (Idrogeno) - R = 0.7414 A [equilibrio reale]" -- fine for a web page
# label, error-prone for an agent to reproduce verbatim across several tool
# calls (exact punctuation, accented characters, etc.). Rather than change
# the kernel's catalog (that key is also what the published Composer page
# uses), this adapter derives a short id from each key's leading token
# (e.g. "H2", "LiH", "HeH+") and accepts either form everywhere a molecule
# `name` is expected. Derived from the live catalog, not hardcoded, so it
# stays correct if the catalog grows.

_molecule_alias_cache: dict | None = None  # short id (lowercased) -> full catalog key


def _short_id(full_key: str) -> str:
    return full_key.split(" (")[0].split(" -")[0].strip()


async def _get_annotated_molecule_catalog(mapping: str) -> list:
    """Catalog entries with a short `id` field added, and refreshes the
    alias cache used by _resolve_molecule_name."""
    global _molecule_alias_cache
    catalog = await _request("GET", "/api/hamiltonians", timeout=10.0, params={"mapping": mapping})
    _molecule_alias_cache = {}
    annotated = []
    for full_key, spec in catalog.items():
        short = _short_id(full_key)
        _molecule_alias_cache[short.lower()] = full_key
        annotated.append({"id": short, "full_name": full_key, **spec})
    return annotated


async def _resolve_molecule_name(name: str) -> str:
    """Accept either a short id ('H2') or the full catalog key and return
    the full catalog key the kernel expects. Falls back to returning the
    input unchanged if it's neither -- the kernel's own 404 (with the name
    as given) is a clearer error than silently guessing."""
    global _molecule_alias_cache
    if _molecule_alias_cache is None:
        await _get_annotated_molecule_catalog("jordan_wigner")
    if name in _molecule_alias_cache.values():
        return name
    return _molecule_alias_cache.get(name.lower(), name)


# --------------------------------------------------------------------------
# Tools: kernel status
# --------------------------------------------------------------------------

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


# --------------------------------------------------------------------------
# Tools: catalogs (presets, gates, noise models, molecules)
# --------------------------------------------------------------------------

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


# --------------------------------------------------------------------------
# Tools: circuit building and execution
# --------------------------------------------------------------------------

@mcp.tool(name="dense_evolution_build_circuit", annotations={"title": "Build OpenQASM from gate operations", **READ_ONLY_IDEMPOTENT})
@catch_errors
async def dense_evolution_build_circuit(params: BuildCircuitInput) -> str:
    """Convert a list of gate operations (as used by the graphical circuit
    builder) into real OpenQASM text, ready to pass to `dense_evolution_run_circuit`.

    Args:
        params (BuildCircuitInput): n_qubits, ops (see dense_evolution_list_gates for valid gate names).

    Returns:
        str: JSON {"qasm": "..."} on success, or "Error: ..." if the op list is invalid.
    """
    data = await _request("POST", "/api/build_from_ops", timeout=30.0, json=params.model_dump())
    return json.dumps(data, indent=2)


@mcp.tool(name="dense_evolution_run_circuit", annotations={"title": "Run an OpenQASM circuit", **COMPUTE})
@catch_errors
async def dense_evolution_run_circuit(params: RunCircuitInput) -> str:
    """Run real OpenQASM on dense_evolution's DenseSVSimulator (or the MPS
    backend for large circuits) and return measurement counts, probabilities,
    and statevector amplitudes. Above the dense backend's safe qubit ceiling
    (see dense_evolution_system_limits), automatically switches to an MPS
    top-k-states approximation instead of failing.

    Large statevectors/probability arrays are truncated to their most
    significant entries (see 'shown' vs 'total_*' fields) to keep the
    response usable in an agent's context -- the full histogram is always
    returned in 'counts' since it's naturally bounded by `shots`.

    Args:
        params (RunCircuitInput): qasm, shots, seed, noise_model, noise_p, backend,
            include_visualizations (see field descriptions).

    Returns:
        str: JSON with n_qubits, backend, counts, truncated probabilities/statevector,
        fidelity_vs_ideal, and (if include_visualizations) paths to saved PNG files.
        For circuits above the dense limit on the 'mps' backend, returns a
        differently-shaped {"large_scale": true, "top_k_states": [...], ...} response.
    """
    payload = params.model_dump(exclude={"include_visualizations"})
    data = await _request("POST", "/api/run", timeout=60.0, json=payload)

    if data.get("large_scale"):
        result = {k: v for k, v in data.items() if k != "circuit_png"}
        if params.include_visualizations:
            result["circuit_png_path"] = _save_png(data.get("circuit_png"), "circuit_large_scale")
        return json.dumps(result, indent=2)

    result = {
        "n_qubits": data["n_qubits"],
        "backend": data["backend"],
        "counts": data["counts"],
        "probabilities": _truncate_probabilities(data["probabilities"]),
        "statevector": _truncate_statevector(data["statevector"]),
        "fidelity_vs_ideal": data.get("fidelity_vs_ideal"),
        "mps_max_bond_used": data.get("mps_max_bond_used"),
        "mps_memory_mb": data.get("mps_memory_mb"),
        "mps_avg_jsd": data.get("mps_avg_jsd"),
    }
    if params.include_visualizations:
        result["circuit_png_path"] = _save_png(data.get("circuit_png"), "circuit")
        result["histogram_png_path"] = _save_png(data.get("histogram_png"), "histogram")
        result["qsphere_png_path"] = _save_png(data.get("qsphere_png"), "qsphere")
        result["bloch_png_path"] = _save_png(data.get("bloch_png"), "bloch")
    return json.dumps(result, indent=2)


# --------------------------------------------------------------------------
# Tools: molecular Hamiltonians
# --------------------------------------------------------------------------

@mcp.tool(name="dense_evolution_molecule_energy", annotations={"title": "Get catalog molecule ground-state energy", **COMPUTE})
@catch_errors
async def dense_evolution_molecule_energy(params: MoleculeEnergyInput) -> str:
    """Compute the exact ground-state energy of a catalog molecule via real
    Hartree-Fock + Jordan-Wigner/Bravyi-Kitaev Hamiltonian construction and
    exact dense diagonalization.

    Args:
        params (MoleculeEnergyInput): name (short id or full catalog name), mapping.

    Returns:
        str: JSON with n_qubits, symbols, geometry, charge, ground_state_energy_hartree.
        "Error: ..." with a 404-style message if `name` is not in the catalog
        (call dense_evolution_list_molecules to see valid names/ids).
    """
    resolved = await _resolve_molecule_name(params.name)
    payload = {**params.model_dump(), "name": resolved}
    return json.dumps(await _request("POST", "/api/hamiltonian/molecule", timeout=60.0, json=payload), indent=2)


@mcp.tool(name="dense_evolution_mix_molecules", annotations={"title": "Mix two catalog Hamiltonians", **COMPUTE})
@catch_errors
async def dense_evolution_mix_molecules(params: MixMoleculesInput) -> str:
    """Compute H_mix = weight_a*H_a + weight_b*H_b for two catalog molecules
    that share the same qubit count (same electron space), and diagonalize
    all three (H_a, H_b, H_mix) for their ground-state energies. Mixing
    molecules with different qubit counts is physically meaningless and is
    rejected with a clear error.

    Args:
        params (MixMoleculesInput): name_a, name_b (short id or full catalog name), weight_a, weight_b, mapping.

    Returns:
        str: JSON with n_qubits, energy_a, energy_b, energy_mixed (all in Hartree).
    """
    name_a = await _resolve_molecule_name(params.name_a)
    name_b = await _resolve_molecule_name(params.name_b)
    payload = {**params.model_dump(), "name_a": name_a, "name_b": name_b}
    return json.dumps(await _request("POST", "/api/hamiltonian/mix", timeout=60.0, json=payload), indent=2)


@mcp.tool(name="dense_evolution_custom_molecule_energy", annotations={"title": "Get custom molecule ground-state energy", **COMPUTE})
@catch_errors
async def dense_evolution_custom_molecule_energy(params: CustomMoleculeInput) -> str:
    """Compute the ground-state energy of an arbitrary molecule (not in the
    catalog) from its atomic symbols and geometry, via the same Hartree-Fock
    pipeline as the catalog. Small molecules only -- exact dense
    diagonalization caps out at 12 qubits, rejected before PennyLane runs if
    the electron/orbital count would exceed that.

    Args:
        params (CustomMoleculeInput): symbols, geometry, charge, mapping.
            len(symbols) must equal len(geometry).

    Returns:
        str: JSON with n_qubits, ground_state_energy_hartree, or "Error: ..."
        if the molecule needs more than 12 qubits.
    """
    return json.dumps(await _request("POST", "/api/hamiltonian/custom", timeout=60.0, json=params.model_dump()), indent=2)


@mcp.tool(name="dense_evolution_energy_scan", annotations={"title": "Scan ground-state energy over several geometries", **COMPUTE})
@catch_errors
async def dense_evolution_energy_scan(params: EnergyScanInput) -> str:
    """Compute the ground-state energy at each of several geometries in one
    call -- e.g. a bond-length dissociation curve or a bond-angle sweep --
    instead of calling dense_evolution_custom_molecule_energy once per
    point. Points are evaluated concurrently against the kernel. A point
    that fails (e.g. too many qubits for exact diagonalization) is reported
    with its own error and does not abort the rest of the scan.

    Args:
        params (EnergyScanInput): symbols, geometries (list of points, max 50),
            charge, mapping, labels (optional, one per point).

    Returns:
        str: JSON with:
        {
            "n_points": int,
            "results": [{"label": ..., "n_qubits": int, "ground_state_energy_hartree": float} | {"label": ..., "error": str}, ...],
            "minimum": {"label": ..., "ground_state_energy_hartree": float} | null  # over successful points only
        }
    """
    if params.labels is not None and len(params.labels) != len(params.geometries):
        raise ValueError(f"{len(params.labels)} labels but {len(params.geometries)} geometries -- must match.")
    labels = params.labels if params.labels is not None else list(range(len(params.geometries)))

    async def _one_point(label, geometry):
        if len(params.symbols) != len(geometry):
            return {"label": label, "error": f"{len(params.symbols)} symbols but {len(geometry)} geometry rows"}
        try:
            data = await _request(
                "POST", "/api/hamiltonian/custom", timeout=60.0,
                json={"symbols": params.symbols, "geometry": geometry, "charge": params.charge, "mapping": params.mapping},
            )
            return {"label": label, "n_qubits": data["n_qubits"], "ground_state_energy_hartree": data["ground_state_energy_hartree"]}
        except Exception as e:
            return {"label": label, "error": str(e)}

    results = await asyncio.gather(*(_one_point(l, g) for l, g in zip(labels, params.geometries)))
    successful = [r for r in results if "ground_state_energy_hartree" in r]
    minimum = min(successful, key=lambda r: r["ground_state_energy_hartree"]) if successful else None
    return json.dumps({"n_points": len(results), "results": results, "minimum": minimum}, indent=2)


# --------------------------------------------------------------------------
# Tools: VQE
# --------------------------------------------------------------------------

@mcp.tool(name="dense_evolution_run_vqe", annotations={"title": "Run VQE ground-state optimization", **COMPUTE})
@catch_errors
async def dense_evolution_run_vqe(params: RunVqeInput) -> str:
    """Run real VQE (Adam gradient descent with adjoint differentiation)
    against a molecule's Jordan-Wigner Hamiltonian, from a fresh random
    start every call -- not a cached/precomputed result. Can take a while
    for 'uccsd' ansatz or high maxiter; consider dense_evolution_health's
    RAM figures and start with a small maxiter/n_layers to gauge cost first.

    Args:
        params (RunVqeInput): either `name` (short id or full catalog name)
            or `symbols`+`geometry` (custom) must be given, plus
            ansatz_type, n_layers, maxiter, step_size, beta1, beta2, seed.

    Returns:
        str: JSON with the optimized energy, convergence history, and
        optimized parameters. "Error: ..." if neither name nor
        symbols+geometry is provided, or the molecule is unknown.
    """
    payload = params.model_dump()
    if params.name:
        payload["name"] = await _resolve_molecule_name(params.name)
    return json.dumps(await _request("POST", "/api/vqe", timeout=600.0, json=payload), indent=2)


# --------------------------------------------------------------------------
# Tools: QM/MM forces and molecular dynamics
# --------------------------------------------------------------------------

@mcp.tool(name="dense_evolution_qmmm_forces", annotations={"title": "Compute Hellmann-Feynman nuclear forces", **COMPUTE})
@catch_errors
async def dense_evolution_qmmm_forces(params: QmmmForcesInput) -> str:
    """Compute real Hellmann-Feynman nuclear forces (F = -d<psi|H(R)|psi>/dR
    via PennyLane autodiff, not finite differences) on a catalog molecule's
    real Hartree-Fock ground state.

    Args:
        params (QmmmForcesInput): name (short id or full catalog name), mapping.

    Returns:
        str: JSON with per-atom force vectors and related energetics.
    """
    resolved = await _resolve_molecule_name(params.name)
    payload = {**params.model_dump(), "name": resolved}
    return json.dumps(await _request("POST", "/api/qmmm_forces", timeout=120.0, json=payload), indent=2)


@mcp.tool(name="dense_evolution_md_trajectory", annotations={"title": "Run a molecular dynamics trajectory", **COMPUTE})
@catch_errors
async def dense_evolution_md_trajectory(params: MdTrajectoryInput) -> str:
    """Run a real Velocity-Verlet MD trajectory driven by Hellmann-Feynman
    forces at every step, for a catalog molecule.

    Args:
        params (MdTrajectoryInput): name (short id or full catalog name),
            n_steps, dt_fs, mapping, recompute_electronic_state (true = true
            ab-initio MD, capped at 30 steps; false = fixed electronic
            state, capped at 200 steps).

    Returns:
        str: JSON with the trajectory (positions/energies per step).
        "Error: ..." if n_steps is out of range for the chosen mode.
    """
    resolved = await _resolve_molecule_name(params.name)
    payload = {**params.model_dump(), "name": resolved}
    return json.dumps(await _request("POST", "/api/md_trajectory", timeout=600.0, json=payload), indent=2)


# --------------------------------------------------------------------------
# Tools: error mitigation
# --------------------------------------------------------------------------

@mcp.tool(name="dense_evolution_mitigate_zne", annotations={"title": "Zero-Noise Extrapolation on an expectation value", **COMPUTE})
@catch_errors
async def dense_evolution_mitigate_zne(params: MitigateZneInput) -> str:
    """Run real Zero-Noise Extrapolation: measure a Pauli expectation value
    at several noise scales under a real Kraus noise channel, then
    extrapolate back to zero noise.

    Args:
        params (MitigateZneInput): qasm, pauli_string, noise_model, noise_p,
            seed, extrapolation_method.

    Returns:
        str: JSON with n_qubits, ideal_expectation, noise_factors,
        noisy_expectations, zne_extrapolated, extrapolation_method.
    """
    return json.dumps(await _request("POST", "/api/mitigate", timeout=120.0, json=params.model_dump()), indent=2)


@mcp.tool(name="dense_evolution_mitigate_density_matrix", annotations={"title": "Density-matrix Zero-Noise Extrapolation", **COMPUTE})
@catch_errors
async def dense_evolution_mitigate_density_matrix(params: MitigateDensityMatrixInput) -> str:
    """Run real density-matrix ZNE: Monte-Carlo density-matrix estimate at
    1x/2x/3x noise_p, extrapolated and projected onto the nearest physical
    state, graded by Uhlmann fidelity against the true ideal state.

    Args:
        params (MitigateDensityMatrixInput): qasm, noise_model, noise_p, seed.

    Returns:
        str: JSON with n_qubits, noise_factors, fidelity_raw, fidelity_corrected.
    """
    return json.dumps(await _request("POST", "/api/mitigate_matrix", timeout=120.0, json=params.model_dump()), indent=2)


@mcp.tool(name="dense_evolution_vector_healing", annotations={"title": "Heal a noisy vector sequence", **COMPUTE})
@catch_errors
async def dense_evolution_vector_healing(params: VectorHealingInput) -> str:
    """Run a real predictive-healing pass over a noisy (n_steps, dim)
    vector sequence -- e.g. VQE convergence telemetry or an MD
    trajectory. Per step, a Phi-Trigger (dense_evolution.healing) decides
    whether the change from a local baseline looks like genuine dynamics
    (kept as-is) or static noise (replaced by the local median). NaN/Inf
    entries are sanitized first regardless of that decision.

    Args:
        params (VectorHealingInput): vectors, radius_baseline.

    Returns:
        str: JSON with healed_vectors, fallback_triggered,
        adaptive_radius_used, reconstruction_error.
    """
    return json.dumps(await _request("POST", "/api/vector_healing", timeout=30.0, json=params.model_dump()), indent=2)


# --------------------------------------------------------------------------
# Tools: traversable-wormhole-inspired quantum teleportation (SYK model)
# --------------------------------------------------------------------------

@mcp.tool(name="dense_evolution_wormhole_select_instance",
          annotations={"title": "Select a good SYK instance for wormhole teleportation", **COMPUTE})
@catch_errors
async def dense_evolution_wormhole_select_instance(params: WormholeSelectInstanceInput) -> str:
    """Find a binary-sparse-SYK random-instance seed suitable for the
    traversable-wormhole-inspired teleportation protocol
    (`dense_evolution_wormhole_teleportation`).

    A uniformly-random seed does NOT reliably show the protocol's
    sign-dependent teleportation signal -- verified directly across many
    seeds (some give a clean peak, some the wrong sign for most of a
    sweep, some are flat noise). arXiv:2604.10090 didn't use an arbitrary
    instance either: they picked one "selected for favorable commutation
    properties" among their chosen terms. This tool reproduces that same
    selection criterion -- screening `n_candidates` seeds by their exact
    commuting/anticommuting term-pair count and returning the one closest
    to `target_commuting` -- rather than trusting a random seed.

    Always call this before `dense_evolution_wormhole_teleportation` /
    `dense_evolution_wormhole_scan` unless you already have a known-good
    seed (e.g. 61, for the defaults n_majorana=8/k_terms=10/target=34,
    the exact match found and used throughout this project's own
    verification -- see research/wormhole_syk.md).

    Args:
        params (WormholeSelectInstanceInput): n_majorana, k_terms, J,
            n_candidates, target_commuting.

    Returns:
        str: JSON {seed, n_majorana, k_terms, commuting, anticommuting,
        target_commuting} -- pass `seed` straight into
        dense_evolution_wormhole_teleportation / _wormhole_scan.
    """
    return json.dumps(await _request("POST", "/api/wormhole_select_instance", timeout=60.0, json=params.model_dump()), indent=2)


@mcp.tool(name="dense_evolution_wormhole_teleportation",
          annotations={"title": "Run traversable-wormhole-inspired quantum teleportation", **COMPUTE})
@catch_errors
async def dense_evolution_wormhole_teleportation(params: WormholeTeleportationInput) -> str:
    """Run one point of the real traversable-wormhole-inspired quantum
    teleportation protocol (Gao-Jafferis-Wall theory, arXiv:2604.10090)
    on a binary sparse SYK model: two coupled chaotic Hamiltonians (L,R),
    a message injected into L via a reference-qubit pair (P,Q), a real
    bilinear L-R coupling exp(i*mu*V), and a readout that is NOT a
    single-qubit expectation value (which the no-signaling theorem
    forbids from ever showing this signal) but the mutual information
    between the reference qubit P and a qubit read out from R.

    Returns a single mutual-information value for the given mu -- the
    physically meaningful result is the *difference* between a positive-
    and negative-mu run at the same (n_majorana, k_terms, seed, t0, t1).
    Call this tool twice with opposite-sign mu, or use
    dense_evolution_wormhole_scan to sweep many (t1, mu) combinations in
    one batched call.

    Requires a well-selected seed (see
    dense_evolution_wormhole_select_instance) -- an arbitrary random seed
    will likely not show a clean signal.

    Args:
        params (WormholeTeleportationInput): n_majorana, k_terms, J, mu,
            t0, t1, seed, with_message, backend, n_steps_evolution,
            n_steps_coupling.

    Returns:
        str: JSON {mutual_information_pt, backend, n_majorana, k_terms,
        mu, t0, t1, seed, with_message}.
    """
    return json.dumps(await _request("POST", "/api/wormhole_teleportation", timeout=120.0, json=params.model_dump()), indent=2)


@mcp.tool(name="dense_evolution_wormhole_scan",
          annotations={"title": "Sweep the wormhole teleportation signal over t1", **COMPUTE})
@catch_errors
async def dense_evolution_wormhole_scan(params: WormholeScanInput) -> str:
    """Sweep `t1_values` for the traversable-wormhole-inspired
    teleportation protocol, running both +mu_magnitude and -mu_magnitude
    at every point -- one call instead of 2*len(t1_values) separate
    dense_evolution_wormhole_teleportation calls. Returns each point's
    mutual information for both signs plus their difference (mu<0 minus
    mu>0 in this project's own convention), the standard readout for the
    protocol's qualitative signature: a smooth peak in that difference
    across the sweep (known peak for seed=61/n_majorana=8/k_terms=10/
    t0=0.3: around t1≈0.6-0.7).

    Unlike dense_evolution_energy_scan, points run sequentially, not
    concurrently: each single teleportation call does real exact
    diagonalization (or Trotterized circuit execution) of the full joint
    L+R+P+Q system and takes several seconds on its own (verified:
    concurrent calls to this specific endpoint crashed the kernel process
    outright -- a real BLAS/eigh thread-safety issue under this protocol's
    heavier-than-usual concurrent linear algebra, not present in the
    lighter Hamiltonian-diagonalization calls energy_scan batches). A
    full 20-point sweep can take several minutes; start with fewer points
    to gauge cost.

    A point that fails does not abort the rest of the sweep; its error is
    reported alongside the successful points.

    Args:
        params (WormholeScanInput): n_majorana, k_terms, J, mu_magnitude,
            t0, t1_values (list, max 20 points), seed, with_message,
            backend, n_steps_evolution, n_steps_coupling.

    Returns:
        str: JSON with:
        {
            "n_points": int,
            "results": [{"t1": float, "mu_positive": float, "mu_negative": float, "delta": float} |
                        {"t1": float, "error": str}, ...],
            "peak": {"t1": ..., "delta": ...} | null  # point with the largest delta, successful points only
        }
    """
    async def _one_point(t1):
        base = dict(
            n_majorana=params.n_majorana, k_terms=params.k_terms, J=params.J, t0=params.t0, t1=t1,
            seed=params.seed, with_message=params.with_message, backend=params.backend,
            n_steps_evolution=params.n_steps_evolution, n_steps_coupling=params.n_steps_coupling,
        )
        try:
            pos = await _request("POST", "/api/wormhole_teleportation", timeout=120.0, json={**base, "mu": params.mu_magnitude})
            neg = await _request("POST", "/api/wormhole_teleportation", timeout=120.0, json={**base, "mu": -params.mu_magnitude})
            i_pos, i_neg = pos["mutual_information_pt"], neg["mutual_information_pt"]
            return {"t1": t1, "mu_positive": i_pos, "mu_negative": i_neg, "delta": i_neg - i_pos}
        except Exception as e:
            return {"t1": t1, "error": str(e)}

    results = [await _one_point(t1) for t1 in params.t1_values]
    successful = [r for r in results if "delta" in r]
    peak = max(successful, key=lambda r: r["delta"]) if successful else None
    return json.dumps({"n_points": len(results), "results": results, "peak": peak}, indent=2)


def main():
    """Console-script entry point (`dense-evolution mcp`, see
    dense_evolution/cli.py) -- identical to running this file directly.
    stdio transport: this process is meant to be launched by an MCP
    client (Claude Code, Claude Desktop, ...) as a subprocess, not run
    standalone in a terminal."""
    mcp.run()


if __name__ == "__main__":
    main()
