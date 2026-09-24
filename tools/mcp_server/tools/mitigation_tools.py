"""Tools: error mitigation (ZNE, density-matrix ZNE, vector healing)."""
import json

from ..client import _request, catch_errors
from ..config import COMPUTE
from ..models import MitigateDensityMatrixInput, MitigateZneInput, VectorHealingInput
from ..server import mcp


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


@mcp.tool(name="dense_evolution_mitigate_coherence", annotations={"title": "Coherence-predictive density-matrix ZNE", **COMPUTE})
@catch_errors
async def dense_evolution_mitigate_coherence(params: MitigateDensityMatrixInput) -> str:
    """Run real coherence-L1-predictive density-matrix ZNE: the same
    Monte-Carlo density-matrix construction as
    dense_evolution_mitigate_density_matrix, but extrapolated via a signal
    that covers phase-type noise (phaseflip, dephasing) the classical-JSD
    signal behind that other tool is structurally blind to -- it only ever
    reads the density matrix's diagonal. Use this one for phase-type noise
    models, the JSD one for amplitude/bitflip-type noise.

    Args:
        params (MitigateDensityMatrixInput): qasm, noise_model, noise_p, seed.

    Returns:
        str: JSON with n_qubits, noise_factors, fidelity_raw, fidelity_corrected.
    """
    return json.dumps(await _request("POST", "/api/mitigate_coherence", timeout=120.0, json=params.model_dump()), indent=2)


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
