"""
Vector healing outlier-correction experiment.

Not to be confused with matrix_healing_zne.py -- that's density-matrix ZNE
(dense_evolution.mitigation), this is ia_utils.vector_healing's Phi-Trigger
engine (dense_evolution.healing) applied to a real vector *sequence* (a VQE
parameter/energy trajectory, MD telemetry, or similar).

Two honest measurements, not just the flattering one:

1. Outlier correction: a smooth ground-truth trajectory (an exponential
   decay + linear drift, the shape of a real VQE energy convergence curve)
   is corrupted at several outlier rates, then healed. Measures how much
   `enhanced_dense_healing_hybrid` actually reduces the error against the
   known ground truth, averaged over several seeds per rate.
2. Distortion on clean data: the same trajectory with realistic small
   jitter but ZERO injected outliers, healed anyway. Checks whether the
   Phi-Trigger over-smooths genuine dynamics it was never asked to fix --
   `ia_utils/vector_healing.py`'s own comments already flag that its
   internal "static vs dynamic" heuristic can fire on structurally noisy
   -but-valid data, so this measures how much that costs in practice
   rather than assuming it's negligible.

Run it yourself to reproduce (or change RATES/SEEDS to check other draws):

    python experiments/vector_healing_outlier_correction.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ia_utils.vector_healing import enhanced_dense_healing_hybrid

N_STEPS = 60
DIM = 4
OUTLIER_RATES = (0.05, 0.15, 0.30)
OUTLIER_SCALE = 40.0       # magnitude of injected outliers relative to the trajectory's own scale
JITTER_SIGMA = 0.02        # realistic small per-step noise, always present
SEEDS = (0, 1, 2, 3, 4)


def ground_truth_trajectory(n_steps=N_STEPS, dim=DIM):
    """Smooth synthetic trajectory shaped like a real VQE energy/parameter
    convergence: exponential relaxation toward a fixed point plus a slow
    linear drift, one independent curve per dimension."""
    t = np.linspace(0, 1, n_steps)
    rng = np.random.default_rng(12345)  # fixed: the "true" curve itself is not randomized per trial
    traj = np.zeros((n_steps, dim))
    for d in range(dim):
        target = rng.uniform(-1, 1)
        rate = rng.uniform(2.0, 5.0)
        drift = rng.uniform(-0.1, 0.1)
        traj[:, d] = target * (1 - np.exp(-rate * t)) + drift * t
    return traj


def add_jitter(traj, sigma, seed):
    rng = np.random.default_rng(seed)
    return traj + rng.normal(scale=sigma, size=traj.shape)


def inject_outliers(traj, rate, scale, seed):
    rng = np.random.default_rng(seed)
    corrupted = traj.copy()
    n_steps = traj.shape[0]
    n_outliers = max(1, int(round(rate * n_steps)))
    outlier_idx = rng.choice(n_steps, size=n_outliers, replace=False)
    corrupted[outlier_idx] = rng.normal(scale=scale, size=(n_outliers, traj.shape[1]))
    return corrupted, outlier_idx


def mse(a, b):
    return float(np.mean((a - b) ** 2))


def run_outlier_correction():
    truth = ground_truth_trajectory()
    print("=== 1. Outlier correction ===")
    for rate in OUTLIER_RATES:
        raw_errors, healed_errors = [], []
        for seed in SEEDS:
            jittered = add_jitter(truth, JITTER_SIGMA, seed)
            corrupted, outlier_idx = inject_outliers(jittered, rate, OUTLIER_SCALE, seed + 1000)
            healed, _meta = enhanced_dense_healing_hybrid(corrupted)
            raw_errors.append(mse(corrupted, truth))
            healed_errors.append(mse(healed, truth))
        raw_errors, healed_errors = np.array(raw_errors), np.array(healed_errors)
        reduction = 1 - healed_errors.mean() / raw_errors.mean()
        print(f"  outlier_rate={rate:.0%}: raw MSE={raw_errors.mean():.4f}  "
              f"healed MSE={healed_errors.mean():.4f}  reduction={reduction:+.1%}")


def run_clean_data_distortion():
    truth = ground_truth_trajectory()
    print("\n=== 2. Distortion on clean data (no injected outliers) ===")
    jittered_errors, healed_errors = [], []
    for seed in SEEDS:
        jittered = add_jitter(truth, JITTER_SIGMA, seed)
        healed, meta = enhanced_dense_healing_hybrid(jittered)
        jittered_errors.append(mse(jittered, truth))
        healed_errors.append(mse(healed, truth))
        print(f"  seed={seed}: jittered MSE={jittered_errors[-1]:.5f}  "
              f"healed MSE={healed_errors[-1]:.5f}  fallback_triggered={meta['fallback_triggered']}")
    jittered_errors, healed_errors = np.array(jittered_errors), np.array(healed_errors)
    change = healed_errors.mean() / jittered_errors.mean() - 1
    print(f"  avg jittered MSE={jittered_errors.mean():.5f}  avg healed MSE={healed_errors.mean():.5f}  "
          f"change={change:+.1%} ({'worse' if change > 0 else 'better/unchanged'} -- "
          f"no real corruption here, so any distortion introduced is unwanted cost)")


def main():
    run_outlier_correction()
    run_clean_data_distortion()


if __name__ == "__main__":
    main()
