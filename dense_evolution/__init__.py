"""
Dense-Evolution
High-performance quantum statevector simulator optimized for NISQ circuits.
"""
from .backends.statevector import DenseSVSimulator
from .circuits.parser import QASMParser, QASMCircuit
from .circuits.compiler import QuantumTranspiler
from .noise import (NoiseModel, NoiseSpec, global_depolarizing_channel, amplitude_damping_channel,
                     phaseflip_channel_exact, cosmic_ray_burst_profile, oscillating_p_eff,
                     pink_noise_p_eff)
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
from .mitigation.zne import (richardson_extrapolate, richardson_amplification_factor,
                          zero_noise_extrapolation, polynomial_extrapolate,
                          bounded_exponential_extrapolate,
                          project_to_physical, bloch_zne, bloch_zne_jit, uhlmann_fidelity, zne_density_matrix,
                          jsd_predictive_zne_density_matrix,
                          coherence_predictive_zne_density_matrix,
                          classically_augmented_zne_phaseflip,
                          richardson_extrapolate_jit, zero_noise_extrapolation_jit,
                          polynomial_extrapolate_jit, uhlmann_fidelity_jit, zne_density_matrix_jit)
from .circuits.diagram import plot_circuit
from .circuits.topology import entangling_layer
from .physics.observables import (pauli_expectation, pauli_sum_expectation, pauli_hamiltonian_to_matrix,
                                   pauli_sum_matvec, multiply_pauli_terms,
                                   pauli_sum_matvec_jax, pauli_sum_expectation_jax, PauliSumOperator)
from .physics.states import ghz_state
from .utils.measurement import sample_counts, statevector_fidelity
from .circuits.qft import qft
from .circuits.arithmetic import (
    add_registers, subtract_registers, add_constant,
    compare_registers, compare_constant,
    add_registers_mod, add_constant_mod, multiply_add_mod, multiply_mod, power_mod,
)
from .circuits.arithmetic_qasm import (
    cuccaro_adder_qasm, draper_adder_qasm, constant_adder_qasm, modular_constant_adder_qasm,
    cmult_mod_qasm, controlled_ua_qasm,
)
from .circuits.shor import shor_order_finding
from .circuits.postselect import postselect
from .circuits.random_circuit import random_circuit
from .utils.drawing import draw_circuit
from .solvers.harrison_tb import (ELEMENTS as HARRISON_ELEMENTS, ETA as HARRISON_ETA,
                                   sp3_dimer_hamiltonian, zincblende_hamiltonian)
from .solvers.vhd_tb import (MATERIALS as VHD_MATERIALS, sp3s_star_hamiltonian,
                              direct_gap_at_gamma, band_extrema_along_path)
from .physics.fermions import majorana_pauli_terms, total_parity_operator, hubbard_hamiltonian_pauli_terms, square_lattice_edges
from .physics.entropy import partial_trace, von_neumann_entropy, mutual_information, central_charge
from .circuits.trotter import (pauli_rotation_ops, trotter_evolve_ops, continuous_pulse_evolve,
                                continuous_dissipative_evolve)
from .circuits.uccsd import find_excitations, single_excitation_ops, double_excitation_ops
from .physics.qec import (pauli_commutes, compute_syndrome, erasure_aware_decode, pymatching_decode,
                           blind_minimum_weight_decode, decode_with_erasure_fallback,
                           counts_in_intervals_dimension, nearest_coset_decode, erasure_ml_decode,
                   peeling_decode, union_find_decode, matching_erasure_decode,
                           estimate_edge_probabilities_from_detection_events)

__version__ = "8.3.2"

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
    "add_registers", "subtract_registers", "add_constant",
    "compare_registers", "compare_constant",
    "add_registers_mod", "add_constant_mod", "multiply_add_mod", "multiply_mod", "power_mod",
    "cuccaro_adder_qasm", "draper_adder_qasm", "constant_adder_qasm", "modular_constant_adder_qasm",
    "cmult_mod_qasm", "controlled_ua_qasm", "shor_order_finding",
    "postselect",
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
    "richardson_extrapolate", "richardson_amplification_factor",
    "zero_noise_extrapolation", "polynomial_extrapolate",
    "bounded_exponential_extrapolate",
    "project_to_physical", "bloch_zne", "bloch_zne_jit", "uhlmann_fidelity", "zne_density_matrix",
    "jsd_predictive_zne_density_matrix", "coherence_predictive_zne_density_matrix",
    "classically_augmented_zne_phaseflip",
    "global_depolarizing_channel", "amplitude_damping_channel", "phaseflip_channel_exact",
    "cosmic_ray_burst_profile", "oscillating_p_eff", "pink_noise_p_eff",
    "richardson_extrapolate_jit", "zero_noise_extrapolation_jit",
    "polynomial_extrapolate_jit", "uhlmann_fidelity_jit", "zne_density_matrix_jit",
    # Physics -- states, observables, entropy, fermions, QEC
    "ghz_state",
    "pauli_expectation", "pauli_sum_expectation", "pauli_hamiltonian_to_matrix", "pauli_sum_matvec",
    "multiply_pauli_terms",
    "pauli_sum_matvec_jax", "pauli_sum_expectation_jax", "PauliSumOperator",
    "partial_trace", "von_neumann_entropy", "mutual_information", "central_charge",
    "majorana_pauli_terms", "total_parity_operator", "hubbard_hamiltonian_pauli_terms", "square_lattice_edges",
    "pauli_commutes", "compute_syndrome", "erasure_aware_decode", "pymatching_decode", "blind_minimum_weight_decode",
    "decode_with_erasure_fallback", "counts_in_intervals_dimension", "nearest_coset_decode",
    "erasure_ml_decode", "peeling_decode", "union_find_decode", "matching_erasure_decode",
    "estimate_edge_probabilities_from_detection_events",
    # Utils -- drawing, measurement, random circuits
    "draw_circuit", "plot_circuit", "sample_counts", "statevector_fidelity", "random_circuit",
]
