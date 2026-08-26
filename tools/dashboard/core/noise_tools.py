"""
Composer-facing wrappers around dense_evolution.noise's standalone noise
profiles and density-matrix channels -- the ones that take a formula and
some parameters, not a circuit, so they don't fit run_circuit_from_qasm's
shape.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

import dense_evolution as de

__all__ = [
    'CosmicRayBurstResult', 'run_cosmic_ray_burst',
    'OscillatingNoiseResult', 'run_oscillating_noise',
    'DensityMatrixChannelResult', 'run_density_matrix_channel',
]


@dataclass
class CosmicRayBurstResult:
    times_us: List[float]
    baseline_gamma: float
    decay_probabilities: List[float]
    peak_ratio: float


def run_cosmic_ray_burst(baseline_gamma: float, times_us: Optional[List[float]] = None) -> CosmicRayBurstResult:
    """Real cosmic-ray/gamma-ray-induced quasiparticle burst profile
    (dense_evolution.cosmic_ray_burst_profile), reproducing a real
    measured event from arXiv:2104.05219 on a 26-qubit chip. Default
    `times_us` samples the impact instant, the ~10us and ~1ms rise
    checkpoints the paper describes, and a point well into the ~25ms
    recovery."""
    if times_us is None:
        times_us = [0.0, 10.0, 1000.0, 50000.0]
    times_arr = np.asarray(times_us, dtype=float)
    profile = np.asarray(de.cosmic_ray_burst_profile(times_arr, baseline_gamma=baseline_gamma))
    return CosmicRayBurstResult(
        times_us=[float(t) for t in times_arr],
        baseline_gamma=baseline_gamma,
        decay_probabilities=[float(p) for p in profile],
        peak_ratio=float(np.max(profile) / baseline_gamma),
    )


@dataclass
class OscillatingNoiseResult:
    base_p: float
    freq: float
    amp: float
    factors: List[float]
    p_eff: List[float]


def run_oscillating_noise(base_p: float, freq: float, amp: float,
                           factors: Optional[List[float]] = None) -> OscillatingNoiseResult:
    """Noise strength that oscillates instead of scaling smoothly with
    `factor` (dense_evolution.oscillating_p_eff) -- for checking whether a
    mitigation technique that assumes smooth noise-vs-scale still works
    when that assumption breaks. Default `factors` sample one full period
    at `freq`."""
    if factors is None:
        factors = [0.0, 1.0, 2.0, 3.0]
    p_eff = [float(de.oscillating_p_eff(base_p, f, freq, amp)) for f in factors]
    return OscillatingNoiseResult(base_p=base_p, freq=freq, amp=amp, factors=list(factors), p_eff=p_eff)


@dataclass
class DensityMatrixChannelResult:
    n_qubits: int
    channel: str
    param: float
    ideal_diagonal: List[float]
    noisy_diagonal: List[float]
    trace: float


def run_density_matrix_channel(qasm_text: str, channel: str, param: float) -> DensityMatrixChannelResult:
    """Apply a density-matrix-level noise channel (as opposed to
    NoiseModel's per-qubit statevector channels) to the ideal density
    matrix of a QASM circuit.

    channel: 'global_depolarizing' (dense_evolution.global_depolarizing_channel,
    mixes the WHOLE register toward the fully mixed state as one unit --
    a state-prep/measurement error reported as a single joint parameter)
    or 'amplitude_damping' (dense_evolution.amplitude_damping_channel,
    single-qubit only, asymmetric |1>->|0> energy relaxation)."""
    if channel not in ('global_depolarizing', 'amplitude_damping'):
        raise ValueError(f"channel must be 'global_depolarizing' or 'amplitude_damping', got {channel!r}")

    parsed = de.QASMParser().parse(qasm_text)
    n_qubits = parsed.n_qubits
    sim = de.DenseSVSimulator(n_qubits)
    sim.run_circuit(parsed.to_tuples())
    sv = np.asarray(sim.get_statevector())
    rho = np.outer(sv, sv.conj())

    if channel == 'global_depolarizing':
        rho_noisy = de.global_depolarizing_channel(rho, param)
    else:
        if n_qubits != 1:
            raise ValueError(f"amplitude_damping is single-qubit only, got a {n_qubits}-qubit circuit")
        rho_noisy = de.amplitude_damping_channel(rho, param)

    return DensityMatrixChannelResult(
        n_qubits=n_qubits,
        channel=channel,
        param=param,
        ideal_diagonal=[float(x) for x in np.diag(rho).real],
        noisy_diagonal=[float(x) for x in np.diag(rho_noisy).real],
        trace=float(np.trace(rho_noisy).real),
    )
