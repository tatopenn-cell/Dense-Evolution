"""Tools: circuit building and execution."""
import json

from ..client import _request, catch_errors
from ..config import COMPUTE, READ_ONLY_IDEMPOTENT
from ..models import BuildCircuitInput, RunCircuitInput
from ..server import mcp
from ..utils.images import _save_png
from ..utils.truncation import _truncate_probabilities, _truncate_statevector


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
            top_k, include_visualizations (see field descriptions).

    Returns:
        str: JSON with n_qubits, backend, counts, truncated probabilities/statevector,
        fidelity_vs_ideal, and (if include_visualizations) paths to saved PNG files.
        For circuits above the dense limit on the 'mps' backend, returns a
        differently-shaped {"large_scale": true, "top_k_states": [...], ...} response.
    """
    payload = params.model_dump(exclude={"include_visualizations", "top_k"})
    data = await _request("POST", "/api/run", timeout=60.0, json=payload)

    image_metadata = {
        "tool": "dense_evolution_run_circuit", "qasm": params.qasm, "shots": params.shots,
        "seed": params.seed, "noise_model": params.noise_model, "noise_p": params.noise_p,
        "backend": params.backend,
    }

    if data.get("large_scale"):
        result = {k: v for k, v in data.items() if k != "circuit_png"}
        if params.include_visualizations:
            result["circuit_png_path"] = _save_png(data.get("circuit_png"), "circuit_large_scale", metadata=image_metadata)
        return json.dumps(result, indent=2)

    result = {
        "n_qubits": data["n_qubits"],
        "backend": data["backend"],
        "counts": data["counts"],
        "probabilities": _truncate_probabilities(data["probabilities"], top_k=params.top_k),
        "statevector": _truncate_statevector(data["statevector"], top_k=params.top_k),
        "fidelity_vs_ideal": data.get("fidelity_vs_ideal"),
        "mps_max_bond_used": data.get("mps_max_bond_used"),
        "mps_memory_mb": data.get("mps_memory_mb"),
        "mps_avg_jsd": data.get("mps_avg_jsd"),
    }
    if params.include_visualizations:
        result["circuit_png_path"] = _save_png(data.get("circuit_png"), "circuit", metadata=image_metadata)
        result["histogram_png_path"] = _save_png(data.get("histogram_png"), "histogram", metadata=image_metadata)
        result["qsphere_png_path"] = _save_png(data.get("qsphere_png"), "qsphere", metadata=image_metadata)
        result["bloch_png_path"] = _save_png(data.get("bloch_png"), "bloch", metadata=image_metadata)
    return json.dumps(result, indent=2)
