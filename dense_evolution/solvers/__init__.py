"""Solvers subpackage: algorithms built on top of the backends
(VQE/autodiff gradients, tight-binding solvers)."""
from .autodiff import circuit_to_energy_fn
from .harrison_tb import (
    ELEMENTS, ETA, HBAR2_OVER_M_EV_ANG2,
    hopping_integral, sp3_bond_block, sp3_dimer_hamiltonian, zincblende_hamiltonian,
)
from .vhd_tb import (
    Material, MATERIALS, sp3s_star_hamiltonian, direct_gap_at_gamma,
    band_extrema_along_path,
)

__all__ = [
    "circuit_to_energy_fn",
    "ELEMENTS", "ETA", "HBAR2_OVER_M_EV_ANG2",
    "hopping_integral", "sp3_bond_block", "sp3_dimer_hamiltonian", "zincblende_hamiltonian",
    "Material", "MATERIALS", "sp3s_star_hamiltonian", "direct_gap_at_gamma",
    "band_extrema_along_path",
]
