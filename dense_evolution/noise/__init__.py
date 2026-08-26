"""Every way to put noise into a Dense-Evolution simulation, in one place.

Three complementary mechanisms:

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

Real device noise from a Qiskit backend's own calibration data
(`noise_model_from_qiskit_backend`) lives in `dense_evolution.interop`,
not here, since it needs a Qiskit `BackendV2` object as input rather than
a noise-specific dependency.

This subpackage previously lived inside `dense_evolution.circuits.registry`
alongside unrelated hardware-detection code (`QuantumHardwareRegistry`);
`registry.py` re-exports `NoiseModel`/`NoiseSpec` from here for backward
compatibility, but `dense_evolution.noise` is the canonical import path
for new code.
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
]
