"""Backward-compatibility shim -- the real implementation moved to
dense_evolution.solvers.harrison_tb as part of the Phase 2 subpackage
split (see prog.txt). Kept so `from dense_evolution.harrison_tb import
zincblende_hamiltonian` (used by external consumers, e.g.
Dense-Evolution-Discovery) keeps working unchanged. Import from
dense_evolution.solvers.harrison_tb directly in new code.
"""
from dense_evolution.solvers.harrison_tb import (
    ELEMENTS, ETA, HBAR2_OVER_M_EV_ANG2,
    hopping_integral, sp3_bond_block, sp3_dimer_hamiltonian, zincblende_hamiltonian,
)

__all__ = [
    "ELEMENTS", "ETA", "HBAR2_OVER_M_EV_ANG2",
    "hopping_integral", "sp3_bond_block", "sp3_dimer_hamiltonian", "zincblende_hamiltonian",
]
