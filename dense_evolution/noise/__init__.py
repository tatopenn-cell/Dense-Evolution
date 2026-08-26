"""Every way to put noise into a Dense-Evolution simulation, in one place.

- **`NoiseModel`** (`.kraus_channels`, dispatching to one file per channel
  under `.kraus`: `ideal`, `depolarizing`, `bitflip`, `phaseflip`,
  `amplitude_damping`, `combined`) -- 6 stochastic single-qubit Kraus
  channels, applied directly to a statevector via `apply_to_sv`.
- **`NoiseSpec`** (`.differentiable`) -- the native JAX-differentiable
  representation of a noise configuration, so noise strength itself can be
  a traced/differentiable value inside `circuit_to_energy_fn`.
- **Coherent adversarial noise** (`.coherent_attack`) -- a genuinely
  continuous, multi-qubit coherent error channel (`apply_rz_all`) and a
  JAX-differentiable search (`craft_adversarial_delta`,
  `craft_adversarial_delta_constrained`) for a worst-case direction against
  a stabilizer code's syndrome -- promoted from Dense-Evolution-Discovery's
  Steane [[7,1,3]] investigation, including its honest negative result
  (see `coherent_attack`'s module docstring).
- **Density-matrix channels** (`.density_matrix_channels`) --
  `global_depolarizing_channel`, `amplitude_damping_channel`: noise applied
  directly to a density matrix instead of a statevector, for density-matrix
  ZNE's noise ensemble.
- **`cosmic_ray_burst_profile`** (`.cosmic_ray`) -- a real, time-dependent
  noise-strength profile for a cosmic-ray-induced quasiparticle burst.
- **`oscillating_p_eff`** (`.oscillating`) -- a noise strength that
  oscillates instead of scaling smoothly, for stress-testing mitigation
  techniques that assume smoothness.

Real device noise from a Qiskit backend's own calibration data
(`noise_model_from_qiskit_backend`) lives in `dense_evolution.interop`,
not here, since it needs a Qiskit `BackendV2` object as input rather than
a noise-specific dependency.

Everything in this package previously lived scattered across
`dense_evolution.circuits.registry` (alongside unrelated hardware-detection
code) and `dense_evolution.mitigation.zne` (alongside unrelated mitigation
techniques, which only ever cancel noise, never generate it). Both modules
re-export the relevant names from here for backward compatibility, but
`dense_evolution.noise` is the canonical import path for new code.
"""
from .kraus_channels import NoiseModel
from .differentiable import NoiseSpec
from .coherent_attack import (
    apply_rz_all,
    x_stabilizer_leakage,
    craft_adversarial_delta,
    project_l2_linf,
    craft_adversarial_delta_constrained,
    decoder_failure_rate,
    random_delta_failure_stats,
)
from .density_matrix_channels import global_depolarizing_channel, amplitude_damping_channel
from .cosmic_ray import cosmic_ray_burst_profile
from .oscillating import oscillating_p_eff

__all__ = [
    "NoiseModel",
    "NoiseSpec",
    "apply_rz_all",
    "x_stabilizer_leakage",
    "craft_adversarial_delta",
    "project_l2_linf",
    "craft_adversarial_delta_constrained",
    "decoder_failure_rate",
    "random_delta_failure_stats",
    "global_depolarizing_channel",
    "amplitude_damping_channel",
    "cosmic_ray_burst_profile",
    "oscillating_p_eff",
]
