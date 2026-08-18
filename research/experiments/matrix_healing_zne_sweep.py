"""
Broader sweep for the density-matrix ZNE healing experiment
(experiments/matrix_healing_zne.py): does the measured improvement hold
across more qubits and other noise channels, or was it specific to the
one 2-qubit/depolarizing configuration first tested?

Same honesty constraint as the original experiment: rho_ideal is used
ONLY to grade raw vs. corrected fidelity at the end, never as input to
the noise ensemble, the extrapolation, or the physical projection.

K_TRAJECTORIES/SEEDS were originally 150/3 -- too few. richardson_extrapolate's
3-point Lagrange coefficients (3, -3, 1) amplify statistical (shot) noise in
the input by a factor related to their sum of squares (19x) relative to a
single raw measurement, so an undersampled Monte Carlo density-matrix
estimate makes the *corrected* result unreliable even when the correction
itself is sound -- confirmed directly: re-running the original phaseflip/
amplitude_damping failures at higher K (300-1200) with more seeds turned
"unreliable, sometimes net-negative" into "consistently and strongly
positive" (see git history for the isolation test). K_TRAJECTORIES=400 and
5 seeds here is chosen to keep this Monte Carlo noise floor well below the
effect size being measured, not because more is free.

    python experiments/matrix_healing_zne_sweep.py
"""
import os
import sys

import numpy as np
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dense_evolution as de
from dense_evolution.circuits.registry import NoiseModel
from dense_evolution.mitigation import uhlmann_fidelity, zne_density_matrix

QUBIT_COUNTS = (2, 3, 4, 5)
NOISE_MODELS = ("depolarizing", "bitflip", "phaseflip", "amplitude_damping", "combined")
BASE_P = 0.05
SCALES = (1.0, 2.0, 3.0)
K_TRAJECTORIES = 400
SEEDS = (0, 1, 2, 3, 4)


def ghz_sv(n_qubits):
    sim = de.DenseSVSimulator(n_qubits)
    ops = [("h", 0)] + [("cx", 0, i) for i in range(1, n_qubits)]
    sim.run_circuit(ops)
    return np.asarray(sim.get_statevector())


def noisy_density_matrix(ideal_sv, n_qubits, model, p, k, rng):
    dim = len(ideal_sv)
    rho = np.zeros((dim, dim), dtype=np.complex128)
    for _ in range(k):
        sv_noisy = NoiseModel.apply_to_sv(ideal_sv.copy(), n_qubits, model, p, rng=rng)
        rho += np.outer(sv_noisy, sv_noisy.conj())
    rho /= k
    return jnp.asarray(rho, dtype=jnp.complex128)


def run_one(n_qubits, model, seed):
    ideal_sv = ghz_sv(n_qubits)
    rho_ideal = jnp.asarray(np.outer(ideal_sv, ideal_sv.conj()), dtype=jnp.complex128)
    rng = np.random.default_rng(seed)

    rho_at_scales = jnp.stack([
        noisy_density_matrix(ideal_sv, n_qubits, model, BASE_P * scale, K_TRAJECTORIES, rng)
        for scale in SCALES
    ])
    raw = uhlmann_fidelity(rho_at_scales[0], rho_ideal)
    corrected = zne_density_matrix(rho_at_scales, SCALES)
    corrected_fid = uhlmann_fidelity(corrected, rho_ideal)
    return raw, corrected_fid


def main():
    print(f"{'qubits':>6} {'noise':>18} {'avg_raw':>10} {'avg_corr':>10} {'avg_delta':>11} "
          f"{'std_delta':>10} {'wins/total':>11}")
    print("-" * 84)
    overall_deltas = []
    for n_qubits in QUBIT_COUNTS:
        for model in NOISE_MODELS:
            deltas, raws, corrs = [], [], []
            for seed in SEEDS:
                raw, corrected = run_one(n_qubits, model, seed)
                raws.append(raw)
                corrs.append(corrected)
                deltas.append(corrected - raw)
            deltas = np.array(deltas)
            overall_deltas.extend(deltas.tolist())
            wins = int(np.sum(deltas > 0))
            print(f"{n_qubits:>6} {model:>18} {np.mean(raws):>10.4f} {np.mean(corrs):>10.4f} "
                  f"{np.mean(deltas):>+11.4f} {np.std(deltas):>10.4f} {wins}/{len(deltas):>9}")

    overall_deltas = np.array(overall_deltas)
    print("-" * 72)
    print(f"overall: {len(overall_deltas)} runs, "
          f"{int(np.sum(overall_deltas > 0))} positive, "
          f"{int(np.sum(overall_deltas < 0))} negative, "
          f"mean delta {overall_deltas.mean():+.4f}, "
          f"min {overall_deltas.min():+.4f}, max {overall_deltas.max():+.4f}")


if __name__ == "__main__":
    main()
