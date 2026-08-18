"""
Density-matrix ZNE healing experiment.

Builds a noiseless circuit's ideal density matrix, a Monte-Carlo-ensemble
noisy density matrix at 3 depolarizing-noise scales, then compares raw
(base-scale) fidelity against corrected (Richardson-extrapolated +
Smolin-Gambetta-Smith-projected, via `dense_evolution.mitigation.
zne_density_matrix`) fidelity -- both against the true ideal state, used
ONLY here for grading, never as input to the noise ensemble, the
extrapolation, or the physical-projection step (that would be oracle
access, not error mitigation).

This is the exact experiment `zne_density_matrix`'s docstring reports the
measured finding from -- run it yourself to reproduce (or change SEEDS to
check other random draws):

    python experiments/matrix_healing_zne.py
"""
import os
import sys

import numpy as np
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dense_evolution as de
from dense_evolution.circuits.registry import NoiseModel
from dense_evolution.mitigation import uhlmann_fidelity, zne_density_matrix

N_QUBITS = 2
BASE_P = 0.05
SCALES = (1.0, 2.0, 3.0)
K_TRAJECTORIES = 200
SEEDS = (0, 1, 2, 3)


def bell_state_sv():
    sim = de.DenseSVSimulator(N_QUBITS)
    sim.run_circuit([("h", 0), ("cx", 0, 1)])
    return np.asarray(sim.get_statevector())


def noisy_density_matrix(ideal_sv, p, k, rng):
    dim = len(ideal_sv)
    rho = np.zeros((dim, dim), dtype=np.complex128)
    for _ in range(k):
        sv_noisy = NoiseModel.apply_to_sv(ideal_sv.copy(), N_QUBITS, 'depolarizing', p, rng=rng)
        rho += np.outer(sv_noisy, sv_noisy.conj())
    rho /= k
    return jnp.asarray(rho, dtype=jnp.complex128)


def run_one_seed(ideal_sv, rho_ideal, seed):
    rng = np.random.default_rng(seed)
    rho_at_scales = jnp.stack([
        noisy_density_matrix(ideal_sv, BASE_P * scale, K_TRAJECTORIES, rng)
        for scale in SCALES
    ])
    raw_fidelity = uhlmann_fidelity(rho_at_scales[0], rho_ideal)
    corrected = zne_density_matrix(rho_at_scales, SCALES)
    corrected_fidelity = uhlmann_fidelity(corrected, rho_ideal)
    return raw_fidelity, corrected_fidelity


def main():
    ideal_sv = bell_state_sv()
    rho_ideal = jnp.asarray(np.outer(ideal_sv, ideal_sv.conj()), dtype=jnp.complex128)

    raw_vals, corrected_vals = [], []
    for seed in SEEDS:
        raw, corrected = run_one_seed(ideal_sv, rho_ideal, seed)
        delta = corrected - raw
        print(f"seed={seed}: raw={raw:.6f}  corrected={corrected:.6f}  delta={delta:+.6f}")
        raw_vals.append(raw)
        corrected_vals.append(corrected)

    raw_vals, corrected_vals = np.array(raw_vals), np.array(corrected_vals)
    print()
    print(f"avg raw fidelity:       {raw_vals.mean():.6f}")
    print(f"avg corrected fidelity: {corrected_vals.mean():.6f}")
    print(f"avg delta:              {(corrected_vals - raw_vals).mean():+.6f}")
    print(f"delta range:            [{(corrected_vals - raw_vals).min():+.6f}, "
          f"{(corrected_vals - raw_vals).max():+.6f}]")


if __name__ == "__main__":
    main()
