"""
Dense-Evolution
High-performance quantum statevector simulator optimized for NISQ circuits.
"""
from .backends.statevector import DenseSVSimulator
from .circuits.parser import QASMParser, QASMCircuit
from .circuits.compiler import QuantumTranspiler
from .circuits.registry import NoiseModel, NoiseSpec, QuantumHardwareRegistry
from .circuits.gates import GATES, PARAMETRIC_GATES, GATE_IDS
from .chunk import Chunk
from .interop import (
    from_qiskit, from_pennylane, run_qiskit_circuit, run_pennylane_circuit,
    noise_model_from_qiskit_backend, to_stim,
)
from .autodiff import circuit_to_energy_fn
from .backends.mps import MPSSimulator
from .mitigation.zne import (richardson_extrapolate, zero_noise_extrapolation, polynomial_extrapolate,
                          project_to_physical, uhlmann_fidelity, zne_density_matrix,
                          jsd_predictive_zne_density_matrix,
                          richardson_extrapolate_jit, zero_noise_extrapolation_jit,
                          polynomial_extrapolate_jit, uhlmann_fidelity_jit, zne_density_matrix_jit)
from .circuits.topology import entangling_layer
from .physics.observables import pauli_expectation, pauli_sum_expectation, pauli_hamiltonian_to_matrix
from .physics.states import ghz_state
from .utils.measurement import sample_counts, statevector_fidelity
from .circuits.qft import qft
from .random_circuit import random_circuit
from .utils.drawing import draw_circuit
from .harrison_tb import (ELEMENTS as HARRISON_ELEMENTS, ETA as HARRISON_ETA,
                           sp3_dimer_hamiltonian, zincblende_hamiltonian)
from .vhd_tb import (MATERIALS as VHD_MATERIALS, sp3s_star_hamiltonian,
                      direct_gap_at_gamma, band_extrema_along_path)
from .physics.fermions import majorana_pauli_terms
from .physics.entropy import partial_trace, von_neumann_entropy, mutual_information
from .circuits.trotter import pauli_rotation_ops, trotter_evolve_ops
from .physics.qec import pauli_commutes, compute_syndrome, erasure_aware_decode

__version__ = "8.1.60"
