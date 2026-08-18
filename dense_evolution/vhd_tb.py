"""Backward-compatibility shim -- the real implementation moved to
dense_evolution.solvers.vhd_tb as part of the Phase 2 subpackage split
(see prog.txt). Kept so `from dense_evolution.vhd_tb import
sp3s_star_hamiltonian` (used by external consumers, e.g.
Dense-Evolution-Discovery) keeps working unchanged. Import from
dense_evolution.solvers.vhd_tb directly in new code.
"""
from dense_evolution.solvers.vhd_tb import (
    Material, MATERIALS, sp3s_star_hamiltonian, direct_gap_at_gamma,
    band_extrema_along_path,
)

__all__ = [
    "Material", "MATERIALS", "sp3s_star_hamiltonian", "direct_gap_at_gamma",
    "band_extrema_along_path",
]
