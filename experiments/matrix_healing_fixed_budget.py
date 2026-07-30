"""
Does splitting a FIXED total measurement budget across more noise-scale
points actually help, or does the earlier "more points helps" finding
only look good because it secretly spent more measurements (more K per
point) than the 3-point baseline?

This is the fair comparison: same total number of Monte Carlo trajectories
across all configurations, just split differently -- 3 points x K=400
(1200 total) vs. 5 points x K=240 (1200 total) vs. 7 points x K=171
(~1200 total). Answers "what did we actually win, all else equal."

    python experiments/matrix_healing_fixed_budget.py
"""
import os
import sys

import numpy as np
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dense_evolution as de
from dense_evolution.registry import NoiseModel
from dense_evolution.mitigation import (
    uhlmann_fidelity, richardson_extrapolate, project_to_physical, zne_density_matrix,
)

N_QUBITS = 4
BASE_P = 0.05
SEEDS = (0, 1, 2, 3, 4)
NOISE_MODELS = ("depolarizing", "bitflip", "phaseflip", "amplitude_damping", "combined")

CONFIGS = {
    "3pt K=400 (baseline, classic ZNE)": ((1.0, 2.0, 3.0), 400, "richardson"),
    "5pt K=240 (deg2 lstsq)":            ((1.0, 2.0, 3.0, 4.0, 5.0), 240, "lstsq2"),
    "7pt K=171 (deg2 lstsq)":            ((1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0), 171, "lstsq2"),
}


def ghz_sv(n):
    sim = de.DenseSVSimulator(n)
    ops = [("h", 0)] + [("cx", 0, i) for i in range(1, n)]
    sim.run_circuit(ops)
    return np.asarray(sim.get_statevector())


def noisy_rho(ideal_sv, model, p, k, rng):
    dim = len(ideal_sv)
    rho = np.zeros((dim, dim), dtype=np.complex128)
    for _ in range(k):
        sv = NoiseModel.apply_to_sv(ideal_sv.copy(), N_QUBITS, model, p, rng=rng)
        rho += np.outer(sv, sv.conj())
    rho /= k
    return jnp.asarray(rho, dtype=jnp.complex128)


def run_one(ideal_sv, rho_ideal, model, scales, k, method, seed):
    rng = np.random.default_rng(seed)
    rho_at_scales = jnp.stack([noisy_rho(ideal_sv, model, BASE_P * s, k, rng) for s in scales])
    raw = uhlmann_fidelity(rho_at_scales[0], rho_ideal)
    if method == "richardson":
        corrected_rho = project_to_physical(richardson_extrapolate(rho_at_scales, scales))
    else:
        corrected_rho = zne_density_matrix(rho_at_scales, scales, degree=2)
    corrected = uhlmann_fidelity(corrected_rho, rho_ideal)
    return raw, corrected


def main():
    ideal_sv = ghz_sv(N_QUBITS)
    rho_ideal = jnp.asarray(np.outer(ideal_sv, ideal_sv.conj()), dtype=jnp.complex128)

    print(f"n_qubits={N_QUBITS}, base_p={BASE_P}, ~1200 total trajectories per run regardless of config\n")
    summary = {}
    for label, (scales, k, method) in CONFIGS.items():
        actual_budget = k * len(scales)
        all_deltas = []
        for model in NOISE_MODELS:
            deltas = []
            for seed in SEEDS:
                raw, corrected = run_one(ideal_sv, rho_ideal, model, scales, k, method, seed)
                deltas.append(corrected - raw)
            all_deltas.extend(deltas)
        summary[label] = (np.array(all_deltas), actual_budget)

    print(f"{'config':>36} {'actual_budget':>14} {'mean_delta':>12} {'std_delta':>10} {'wins/total':>12}")
    print("-" * 90)
    for label, (arr, budget) in summary.items():
        print(f"{label:>36} {budget:>14} {arr.mean():>+12.4f} {arr.std():>10.4f} "
              f"{int(np.sum(arr>0))}/{len(arr):>10}")


if __name__ == "__main__":
    main()
