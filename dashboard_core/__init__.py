"""
dashboard_core — compute layer for app_dashboard.py's Quantum Simulator tab.

Split (Phase 1 of the dashboard refactor) from a single 1900-line
dashboard_core.py into submodules by responsibility. This __init__.py is a
backward-compatible façade: it re-exports exactly the names that were
already used externally (by ui_pages/quantum_simulator.py, test_dashboard_core.py,
test_mps.py) before the split, so `import dashboard_core as dc` and every
`dc.xxx` call site keeps working unchanged. Anything not re-exported here
(private helpers, the *_body variants, plot_theme internals) was never part
of that public surface.

Adapted from dash.py (the ipywidgets/Colab notebook export). dash.py is not a
clean importable module (it runs a subprocess pip-install check, builds a full
ipywidgets UI, and does an unconditional `from google.colab import files` on
import), so its logic is extracted and adapted here into plain functions that
take explicit values instead of ipywidgets objects, with no import-time side
effects.
"""

from .qasm_library import QASM_LIBRARY, infer_qubit_count_from_qasm
from .hamiltonians import LIBRERIA_HAMILTONIANE, get_compatible_hamiltonians, save_custom_hamiltonian
from .simulation_runner import run_simulation
from .vqe_engine import QM_MM_HEAVY_QUBIT_THRESHOLD, run_vqe_telemetry
from .md_telemetry import run_md_telemetry
from .metrics import compute_overview_metrics
from .panels import (
    build_panel_overview, build_panel_fisica, build_panel_mosaico,
    build_panel_vqe_results, build_panel_md_results,
    build_panel_performance, build_panel_hamiltonian,
)
from .helix_3d import build_3d_helix_patch
from .provenance import run_benchmark_scan, build_provenance_json
from .mitigation_runner import run_mitigation_sweep
from .mitigation_panel import build_panel_mitigation

__all__ = [
    'QASM_LIBRARY', 'infer_qubit_count_from_qasm',
    'LIBRERIA_HAMILTONIANE', 'get_compatible_hamiltonians', 'save_custom_hamiltonian',
    'run_simulation',
    'QM_MM_HEAVY_QUBIT_THRESHOLD', 'run_vqe_telemetry',
    'run_md_telemetry',
    'compute_overview_metrics',
    'build_panel_overview', 'build_panel_fisica', 'build_panel_mosaico',
    'build_panel_vqe_results', 'build_panel_md_results',
    'build_panel_performance', 'build_panel_hamiltonian',
    'build_3d_helix_patch',
    'run_benchmark_scan', 'build_provenance_json',
    'run_mitigation_sweep', 'build_panel_mitigation',
]
