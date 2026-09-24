"""
Real Hartree-Fock SCF convergence diagnostics
(dense_evolution.native_hf.scf.diagnose_convergence) for an arbitrary
molecule, exposed as a thin Composer-kernel wrapper. Needs the `armor`
extra (dense-armor) -- dense_evolution.native_hf.scf itself does not
require it, but diagnose_convergence's Hampel/Tukey filters do.
"""
from dataclasses import dataclass

from basis_set_exchange.lut import element_Z_from_sym

from dense_evolution.native_hf.bridge import build_qubit_hamiltonian
from dense_evolution.native_hf.scf import diagnose_convergence as _diagnose_convergence

__all__ = ['NativeHfDiagnosticsResult', 'run_native_hf_diagnostics']


@dataclass
class NativeHfDiagnosticsResult:
    n_qubits: int
    converged: bool
    n_iterations: int
    electronic_energy_hartree: float
    total_energy_hartree: float
    n_anomalies_hampel: int
    n_anomalies_tukey: int
    anomaly_fraction_hampel: float
    anomaly_fraction_tukey: float


def run_native_hf_diagnostics(symbols, geometry, charge: int = 0,
                               active_electrons=None, active_orbitals=None) -> NativeHfDiagnosticsResult:
    """Run real Hartree-Fock SCF (dense_evolution.native_hf, not
    PennyLane's own STO-3G table -- this is the fallback path used for
    elements outside it, e.g. Silicon) on an arbitrary molecule, and run
    Dense-Armor's Hampel/Tukey anomaly filters over the real per-iteration
    energy trace instead of trusting the final `converged` flag alone.
    Jordan-Wigner mapping only -- native_hf/bridge.py calls
    qml.jordan_wigner directly, not a general mapper.

    Raises ImportError with an actionable message if the `armor` extra
    (pip install dense-evolution[armor]) isn't installed -- diagnostics
    genuinely can't run without it, this doesn't silently skip them.
    """
    atomic_numbers = [element_Z_from_sym(s) for s in symbols]
    n_electrons = sum(atomic_numbers) - charge
    H, n_qubits, hf_result = build_qubit_hamiltonian(
        atomic_numbers, geometry, n_electrons,
        active_electrons=active_electrons, active_orbitals=active_orbitals,
    )
    diagnostics = _diagnose_convergence(hf_result)
    return NativeHfDiagnosticsResult(
        n_qubits=n_qubits,
        converged=bool(hf_result.converged),
        n_iterations=diagnostics["n_iterations"],
        electronic_energy_hartree=float(hf_result.electronic_energy),
        total_energy_hartree=float(hf_result.total_energy),
        n_anomalies_hampel=diagnostics["n_anomalies_hampel"],
        n_anomalies_tukey=diagnostics["n_anomalies_tukey"],
        anomaly_fraction_hampel=diagnostics["anomaly_fraction_hampel"],
        anomaly_fraction_tukey=diagnostics["anomaly_fraction_tukey"],
    )
