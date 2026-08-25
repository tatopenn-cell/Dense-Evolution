"""
dashboard_core -- real compute + visualization layer for app_dashboard.py.

Rebuilt from scratch on the structure of IBM Quantum Composer (circuit
editor, statevector, probabilities, Q-sphere), wired to dense_evolution's
actual DenseSVSimulator and to Qiskit's own real visualization functions
-- no hand-rolled plotting, no synthetic data. The previous dashboard_core
(VQE, molecular Hamiltonians, mitigation, AI vector-healing) lives intact
on the feature/streamlit-dashboard and feature/ipywidgets-dash-panel
branches and will be reintegrated selectively once this base is solid.
"""

import dense_evolution as _de

# dense_evolution no longer forces jax_enable_x64 as an import-time side
# effect (see dense_evolution/config.py) -- it's enabled lazily, only by
# the specific call that needs complex128 (e.g. DenseSVSimulator). Some
# paths here (MPSSimulator-only branches in engine.py, Chunk usage in
# system_limits.py/wormhole.py) never construct a DenseSVSimulator, so
# this dashboard -- which always wants float64 -- sets it explicitly
# once, up front, instead of relying on that now-removed side effect.
_de.set_precision(True)

from .qasm_library import QASM_LIBRARY, gate_tuples_to_qasm
from .engine import (
    SimulationResult, run_circuit_from_qasm,
    LargeScaleMPSResult, run_large_circuit_mps, MPS_DENSE_CONTRACTION_LIMIT,
)
from .visuals import (
    draw_circuit_figure, histogram_figure, qsphere_figure, bloch_multivector_figure,
)
from .graphical_builder import GATE_PALETTE, ops_to_native_tuples
from .circuit_builder_component import mount_circuit_builder
from .hamiltonians import (
    MOLECULE_CATALOG, build_molecular_hamiltonian, get_compatible_molecules,
    get_all_molecules, get_molecule_n_qubits,
    get_molecular_hamiltonian_matrix, ground_state_energy, ground_state_energy_sparse,
    linear_chain_geometry, ring_geometry, mix_hamiltonians,
)
from .mitigation import (
    MitigationResult, run_zne_mitigation, DensityMatrixZNEResult, run_density_matrix_zne,
)
from .system_limits import max_safe_dense_qubits
from .vqe import run_vqe
from .qmmm import (
    ATOMIC_MASSES_AMU,
    compute_hellmann_feynman_forces, md_step, run_md_trajectory,
)
from .wormhole import (
    build_sparse_syk_terms, commuting_pair_count, select_good_instance,
    run_wormhole_protocol, run_wormhole_protocol_trotter,
)
from .vector_healing import VectorHealingResult, run_vector_healing

__all__ = [
    'QASM_LIBRARY', 'gate_tuples_to_qasm',
    'SimulationResult', 'run_circuit_from_qasm',
    'LargeScaleMPSResult', 'run_large_circuit_mps', 'MPS_DENSE_CONTRACTION_LIMIT',
    'draw_circuit_figure', 'histogram_figure', 'qsphere_figure', 'bloch_multivector_figure',
    'GATE_PALETTE', 'ops_to_native_tuples',
    'mount_circuit_builder',
    'MOLECULE_CATALOG', 'build_molecular_hamiltonian', 'get_compatible_molecules',
    'get_all_molecules', 'get_molecule_n_qubits',
    'get_molecular_hamiltonian_matrix', 'ground_state_energy', 'ground_state_energy_sparse',
    'linear_chain_geometry', 'ring_geometry', 'mix_hamiltonians',
    'MitigationResult', 'run_zne_mitigation',
    'DensityMatrixZNEResult', 'run_density_matrix_zne',
    'max_safe_dense_qubits',
    'run_vqe',
    'ATOMIC_MASSES_AMU',
    'compute_hellmann_feynman_forces', 'md_step', 'run_md_trajectory',
    'build_sparse_syk_terms', 'commuting_pair_count', 'select_good_instance',
    'run_wormhole_protocol', 'run_wormhole_protocol_trotter',
    'VectorHealingResult', 'run_vector_healing',
]
