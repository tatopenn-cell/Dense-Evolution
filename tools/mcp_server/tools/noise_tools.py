"""Tools: standalone noise profiles and density-matrix channels."""
import json

from ..client import _request, catch_errors
from ..config import COMPUTE
from ..models import CosmicRayBurstInput, OscillatingNoiseInput, DensityMatrixChannelInput
from ..server import mcp


@mcp.tool(name="dense_evolution_cosmic_ray_burst", annotations={"title": "Cosmic-ray burst noise profile", **COMPUTE})
@catch_errors
async def dense_evolution_cosmic_ray_burst(params: CosmicRayBurstInput) -> str:
    """Real time-dependent decay-probability profile for a cosmic-ray/
    gamma-ray-induced quasiparticle burst, reproducing a real measured
    event from arXiv:2104.05219 on a 26-qubit chip: a two-stage rise then
    a single-exponential recovery.

    Args:
        params (CosmicRayBurstInput): baseline_gamma, times_us.

    Returns:
        str: JSON with times_us, baseline_gamma, decay_probabilities, peak_ratio.
    """
    return json.dumps(await _request("POST", "/api/cosmic_ray_burst", timeout=30.0, json=params.model_dump()), indent=2)


@mcp.tool(name="dense_evolution_oscillating_noise", annotations={"title": "Oscillating noise-scale profile", **COMPUTE})
@catch_errors
async def dense_evolution_oscillating_noise(params: OscillatingNoiseInput) -> str:
    """A noise strength that oscillates instead of scaling smoothly with
    a ZNE-style scale factor -- for checking whether a mitigation
    technique that assumes smooth noise-vs-scale still works when that
    assumption breaks down.

    Args:
        params (OscillatingNoiseInput): base_p, freq, amp, factors.

    Returns:
        str: JSON with base_p, freq, amp, factors, p_eff.
    """
    return json.dumps(await _request("POST", "/api/oscillating_noise", timeout=30.0, json=params.model_dump()), indent=2)


@mcp.tool(name="dense_evolution_density_matrix_channel", annotations={"title": "Apply a density-matrix noise channel", **COMPUTE})
@catch_errors
async def dense_evolution_density_matrix_channel(params: DensityMatrixChannelInput) -> str:
    """Apply a density-matrix-level noise channel to the ideal density
    matrix of a QASM circuit -- distinct from per-qubit statevector
    noise (see dense_evolution_run_circuit's noise_model/noise_p):
    'global_depolarizing' mixes the whole register toward the fully
    mixed state as one unit (a SPAM-style error), 'amplitude_damping' is
    single-qubit asymmetric energy relaxation.

    Args:
        params (DensityMatrixChannelInput): qasm, channel, param.

    Returns:
        str: JSON with n_qubits, channel, param, ideal_diagonal, noisy_diagonal, trace.
    """
    return json.dumps(await _request("POST", "/api/density_matrix_channel", timeout=30.0, json=params.model_dump()), indent=2)
