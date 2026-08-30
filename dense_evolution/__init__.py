"""
Dense-Evolution
High-performance quantum statevector simulator optimized for NISQ circuits.
"""
from .backends.statevector import DenseSVSimulator
from .circuits.parser import QASMParser, QASMCircuit
from .circuits.compiler import QuantumTranspiler
from .noise import (NoiseModel, NoiseSpec, global_depolarizing_channel, amplitude_damping_channel,
                     cosmic_ray_burst_profile, oscillating_p_eff)
from .circuits.registry import QuantumHardwareRegistry
from .circuits.gates import GATES, PARAMETRIC_GATES, GATE_IDS
from .backends.chunk import Chunk
# Also bind dense_evolution.chunk as an attribute of this package (the
# shim at dense_evolution/chunk.py is otherwise only reachable via an
# explicit `import dense_evolution.chunk` / `from dense_evolution.chunk
# import ...` -- Python only auto-binds a submodule as a package
# attribute when something actually imports that exact module path, and
# `from .backends.chunk import Chunk` above binds backends.chunk, not
# chunk). Real external code (dashboard_core.hamiltonians) does
# `de.chunk.SafeMemoryGuard()` -- attribute access, not an import -- so
# this import's only job is that side effect.
from . import chunk as chunk
from .config import set_precision
from .interop import (
    from_qiskit, from_pennylane, run_qiskit_circuit, run_pennylane_circuit,
    noise_model_from_qiskit_backend, to_stim,
)
from .solvers.autodiff import circuit_to_energy_fn
from .backends.mps import MPSSimulator
from .mitigation.zne import (richardson_extrapolate, zero_noise_extrapolation, polynomial_extrapolate,
                          bounded_exponential_extrapolate,
                          project_to_physical, uhlmann_fidelity, zne_density_matrix,
                          jsd_predictive_zne_density_matrix,
                          richardson_extrapolate_jit, zero_noise_extrapolation_jit,
                          polynomial_extrapolate_jit, uhlmann_fidelity_jit, zne_density_matrix_jit)
from .circuits.diagram import plot_circuit
from .circuits.topology import entangling_layer
from .physics.observables import (pauli_expectation, pauli_sum_expectation, pauli_hamiltonian_to_matrix,
                                   pauli_sum_matvec, multiply_pauli_terms)
from .physics.states import ghz_state
from .utils.measurement import sample_counts, statevector_fidelity
from .circuits.qft import qft
from .circuits.random_circuit import random_circuit
from .utils.drawing import draw_circuit
from .solvers.harrison_tb import (ELEMENTS as HARRISON_ELEMENTS, ETA as HARRISON_ETA,
                                   sp3_dimer_hamiltonian, zincblende_hamiltonian)
from .solvers.vhd_tb import (MATERIALS as VHD_MATERIALS, sp3s_star_hamiltonian,
                              direct_gap_at_gamma, band_extrema_along_path)
from .physics.fermions import majorana_pauli_terms, total_parity_operator
from .physics.entropy import partial_trace, von_neumann_entropy, mutual_information
from .circuits.trotter import (pauli_rotation_ops, trotter_evolve_ops, continuous_pulse_evolve,
                                continuous_dissipative_evolve)
from .circuits.uccsd import find_excitations, single_excitation_ops, double_excitation_ops
from .physics.qec import (pauli_commutes, compute_syndrome, erasure_aware_decode, pymatching_decode,
                           blind_minimum_weight_decode, decode_with_erasure_fallback,
                           counts_in_intervals_dimension, nearest_coset_decode)

__version__ = "8.1.68"

__all__ = [
    "__version__",
    # Precision -- process-wide JAX config, set explicitly (see config.py)
    "set_precision",
    # Backends -- the compute engines
    "DenseSVSimulator", "MPSSimulator",
    # Circuits -- representation, parsing, compilation
    "QASMParser", "QASMCircuit", "QuantumTranspiler",
    "NoiseModel", "NoiseSpec", "QuantumHardwareRegistry",
    "GATES", "PARAMETRIC_GATES", "GATE_IDS",
    "entangling_layer", "qft",
    "pauli_rotation_ops", "trotter_evolve_ops", "continuous_pulse_evolve",
    "continuous_dissipative_evolve",
    "find_excitations", "single_excitation_ops", "double_excitation_ops",
    # Chunking / anti-OOM
    "Chunk",
    # Interop -- Qiskit / PennyLane / STIM bridges
    "from_qiskit", "from_pennylane", "run_qiskit_circuit", "run_pennylane_circuit",
    "noise_model_from_qiskit_backend", "to_stim",
    # Solvers -- VQE/autodiff, tight-binding
    "circuit_to_energy_fn",
    "HARRISON_ELEMENTS", "HARRISON_ETA", "sp3_dimer_hamiltonian", "zincblende_hamiltonian",
    "VHD_MATERIALS", "sp3s_star_hamiltonian", "direct_gap_at_gamma", "band_extrema_along_path",
    # Mitigation -- Zero-Noise Extrapolation
    "richardson_extrapolate", "zero_noise_extrapolation", "polynomial_extrapolate",
    "bounded_exponential_extrapolate",
    "project_to_physical", "uhlmann_fidelity", "zne_density_matrix",
    "jsd_predictive_zne_density_matrix", "global_depolarizing_channel", "amplitude_damping_channel",
    "cosmic_ray_burst_profile", "oscillating_p_eff",
    "richardson_extrapolate_jit", "zero_noise_extrapolation_jit",
    "polynomial_extrapolate_jit", "uhlmann_fidelity_jit", "zne_density_matrix_jit",
    # Physics -- states, observables, entropy, fermions, QEC
    "ghz_state",
    "pauli_expectation", "pauli_sum_expectation", "pauli_hamiltonian_to_matrix", "pauli_sum_matvec",
    "multiply_pauli_terms",
    "partial_trace", "von_neumann_entropy", "mutual_information",
    "majorana_pauli_terms", "total_parity_operator",
    "pauli_commutes", "compute_syndrome", "erasure_aware_decode", "pymatching_decode", "blind_minimum_weight_decode",
    "decode_with_erasure_fallback", "counts_in_intervals_dimension", "nearest_coset_decode",
    # Utils -- drawing, measurement, random circuits
    "draw_circuit", "plot_circuit", "sample_counts", "statevector_fidelity", "random_circuit",
]
