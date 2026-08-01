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

from .qasm_library import QASM_LIBRARY
from .engine import SimulationResult, run_circuit_from_qasm
from .visuals import (
    draw_circuit_figure, histogram_figure, qsphere_figure, bloch_multivector_figure,
)
from .graphical_builder import GATE_PALETTE, ops_to_qiskit_circuit
from .circuit_builder_component import mount_circuit_builder

__all__ = [
    'QASM_LIBRARY',
    'SimulationResult', 'run_circuit_from_qasm',
    'draw_circuit_figure', 'histogram_figure', 'qsphere_figure', 'bloch_multivector_figure',
    'GATE_PALETTE', 'ops_to_qiskit_circuit',
    'mount_circuit_builder',
]
