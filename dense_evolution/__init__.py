"""
Dense-Evolution
High-performance quantum statevector simulator optimized for NISQ circuits.
"""
from .simulator import DenseSVSimulator
from .parser import QASMParser, QASMCircuit
from .compiler import QuantumTranspiler
from .registry import NoiseModel, NoiseSpec, QuantumHardwareRegistry
from .gates import GATES, PARAMETRIC_GATES, GATE_IDS
from .chunk import Chunk
from .interop import from_qiskit, from_pennylane, run_qiskit_circuit, run_pennylane_circuit
from .autodiff import circuit_to_energy_fn
from .mps import MPSSimulator
from .mitigation import (richardson_extrapolate, zero_noise_extrapolation, polynomial_extrapolate,
                          project_to_physical, uhlmann_fidelity, zne_density_matrix,
                          richardson_extrapolate_jit, zero_noise_extrapolation_jit,
                          polynomial_extrapolate_jit, uhlmann_fidelity_jit, zne_density_matrix_jit)
from .topology import entangling_layer
from .observables import pauli_expectation, pauli_sum_expectation, pauli_hamiltonian_to_matrix
from .states import ghz_state
from .measurement import sample_counts, statevector_fidelity
from .qft import qft
from .random_circuit import random_circuit
from .drawing import draw_circuit
from .harrison_tb import (ELEMENTS as HARRISON_ELEMENTS, ETA as HARRISON_ETA,
                           sp3_dimer_hamiltonian, zincblende_hamiltonian)
from .vhd_tb import (MATERIALS as VHD_MATERIALS, sp3s_star_hamiltonian,
                      direct_gap_at_gamma)

__version__ = "8.1.46"
