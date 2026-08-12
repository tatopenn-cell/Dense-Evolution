"""Loading contracted basis-set shells for a molecule.

Basis-set parameters (exponents, contraction coefficients) are fetched
from the Basis Set Exchange (the `basis_set_exchange` PyPI package,
data-only, BSD-licensed -- https://www.basissetexchange.org), so this
module works for any element the basis has data for, not just the
handful PennyLane's own bundled STO-3G table covers (which stops at
Ne and is why Silicon needed a hand-written patch earlier in this
project).

The coefficients BSE reports are for *unnormalized* primitives, so each
primitive's contraction coefficient must be rescaled by its own L2 norm
before it can be summed against another shell's primitives. We get that
norm for free by reusing our own overlap_3d on the primitive against
itself: N = 1/sqrt(<primitive|primitive>).
"""

import dataclasses

import basis_set_exchange as bse
import jax
import jax.numpy as jnp
import numpy as np

from dense_evolution.native_hf.gaussians import GaussianShell3D
from dense_evolution.native_hf.overlap import overlap_3d


@dataclasses.dataclass(frozen=True)
class ContractedShell:
    """One angular-momentum shell (s, p, ...) of a contracted GTO."""

    atom_index: int
    center: jax.Array  # shape (3,), atomic units (Bohr)
    degree: int  # 0 for s, 1 for p, ...
    exponents: jax.Array  # shape (K,)
    coefficients: jax.Array  # shape (K,), already primitive-normalized


def _primitive_norm(exponent: jax.Array, degree: int) -> jax.Array:
    """1/sqrt(self-overlap) for the (degree,0,0) Cartesian component --
    for a shell of pure angular momentum `degree` every Cartesian
    component (e.g. px, py, pz) has the same norm by symmetry, so one
    component is enough."""
    g = GaussianShell3D(degree=degree, exponent=exponent, center=jnp.zeros(3))
    self_overlap = overlap_3d(g, g)[degree, 0, 0, degree, 0, 0]
    return 1.0 / jnp.sqrt(self_overlap)


def _contracted_shell_from_bse(shell: dict, center: jax.Array, atom_index: int) -> list[ContractedShell]:
    """BSE groups shells like STO-3G's "SP" as one entry with two rows of
    coefficients (one for the S part, one for the P part) sharing the
    same exponents; we split those back into separate ContractedShells
    since our integral code handles one angular momentum at a time."""
    angular_momenta = shell["angular_momentum"]
    exponents = jnp.array([float(e) for e in shell["exponents"]])

    shells = []
    for row, degree in enumerate(angular_momenta):
        coeffs = jnp.array([float(c) for c in shell["coefficients"][row]])
        norms = jnp.array([_primitive_norm(a, degree) for a in exponents])
        shells.append(
            ContractedShell(
                atom_index=atom_index,
                center=center,
                degree=degree,
                exponents=exponents,
                coefficients=coeffs * norms,
            )
        )
    return shells


def load_element_shells(basis_name: str, atomic_number: int, center: jax.Array, atom_index: int) -> list[ContractedShell]:
    data = bse.get_basis(basis_name, elements=[atomic_number])
    electron_shells = data["elements"][str(atomic_number)]["electron_shells"]

    if any(s["function_type"] not in ("gto", "gto_cartesian") for s in electron_shells):
        raise NotImplementedError("Only Cartesian Gaussian basis sets are currently supported.")

    max_degree = max(
        degree for shell in electron_shells for degree in shell["angular_momentum"]
    )
    if max_degree > 1:
        from basis_set_exchange.lut import element_sym_from_Z

        sym = element_sym_from_Z(atomic_number).capitalize()
        raise NotImplementedError(
            f"native_hf's overlap/kinetic/Coulomb integrals only implement s and p "
            f"shells (degree <= 1); {sym} (Z={atomic_number}) needs a degree-{max_degree} "
            f"shell (d-orbitals or higher) in {basis_name}. Not a silent approximation -- "
            f"this element genuinely isn't supported by this engine yet."
        )

    out = []
    for shell in electron_shells:
        out.extend(_contracted_shell_from_bse(shell, center, atom_index))
    return out


def build_molecule_shells(atomic_numbers: list[int], geometry_bohr: np.ndarray, basis_name: str) -> list[ContractedShell]:
    """geometry_bohr: shape (n_atoms, 3), atomic units."""
    shells = []
    for i, (z, r) in enumerate(zip(atomic_numbers, geometry_bohr)):
        shells.extend(load_element_shells(basis_name, z, jnp.asarray(r), i))
    return shells


_DEGREE_TO_N_CARTESIAN = {0: 1, 1: 3}  # s: 1 component, p: 3 (x,y,z)


def n_cartesian_functions(shells: list[ContractedShell]) -> int:
    return sum(_DEGREE_TO_N_CARTESIAN[s.degree] for s in shells)
