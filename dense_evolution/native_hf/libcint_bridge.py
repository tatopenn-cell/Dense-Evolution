"""Optional PySCF/libcint bridge for native_hf's electron-repulsion tensor.

native_hf's own build_repulsion_tensor (assembly.py) computes the ERI
tensor via JAX-jitted Obara-Saika recursions -- correct, differentiable,
but pays a real JIT-compilation tax on mixed-angular-momentum bases (see
prog.txt: ~532s on Ne/6-31G* even after the primitive-count-padding fix).
libcint (the C library PySCF itself uses) computes the identical tensor
in ~0.006s -- no JIT, no per-basis compile cost, ever, since its kernels
are C templates compiled once when PySCF itself was built.

This is NOT a replacement for native_hf's own engine: nothing in this
codebase differentiates through native_hf's raw integrals today (checked
directly, not assumed), so the autodiff PySCF/libcint can't offer here
costs nothing in practice yet -- but PySCF is a heavy optional dependency
with no Windows wheel (needs a C toolchain to build from source there),
and depending on it contradicts this project's own "no C++ hand-written"
positioning if made anything but opt-in. Install with the `libcint`
extra; everything here raises ImportError with that instruction if PySCF
isn't present, never silently falls back.

Cartesian-component convention mismatch: libcint's own per-shell AO order
and normalization are NOT the same as native_hf's own `cartesian_powers`
convention (see cartesian.py's own docstring) -- confirmed empirically via
`mol.ao_labels()` and `mol.intor('int1e_ovlp')` on Ne/6-31G*, not assumed
from either codebase's documentation:
  - p (degree 1): libcint orders px,py,pz; native_hf orders px,pz,py.
    Every p component has the same self-overlap in both conventions (px,
    py, pz are equivalent by symmetry), so this needs reordering only.
  - d (degree 2): libcint orders xx,xy,xz,yy,yz,zz; native_hf orders
    xx,xz,xy,zz,yz,yy. libcint's raw per-component overlap is NOT
    normalized to 1 the way native_hf's is (xx/yy/zz self-overlap
    2.51327412, xy/xz/yz self-overlap 0.83775804 on Ne/6-31G* -- ratio
    exactly 3, consistent with the same sqrt(3) factor
    cartesian_normalization_ratios already accounts for on native_hf's
    own side), so this needs both reordering AND a per-AO rescale.

Rather than hardcoding libcint's normalization as a formula, the rescale
is computed from libcint's own overlap diagonal at call time (1/sqrt of
it) -- exact for whatever exponent/element/degree is actually in play,
not an assumption about libcint's internal convention that could break on
a different libcint version or basis set. Degrees above 2 raise
NotImplementedError naming the real limitation (native_hf itself caps at
degree 2 today, so this never binds in practice, but the mapping below is
only verified through d, not d and beyond).
"""
import numpy as np

from dense_evolution.native_hf.basis import build_molecule_shells, n_cartesian_functions
from dense_evolution.native_hf.cartesian import cartesian_powers

_LIBCINT_CARTESIAN_ORDER = {
    0: [(0, 0, 0)],
    1: [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
    2: [(2, 0, 0), (1, 1, 0), (1, 0, 1), (0, 2, 0), (0, 1, 1), (0, 0, 2)],
}


def _permutation_libcint_to_native_hf(degree: int) -> np.ndarray:
    """perm such that native_hf_ordered[i] == libcint_ordered[perm[i]]."""
    if degree not in _LIBCINT_CARTESIAN_ORDER:
        raise NotImplementedError(
            f"libcint_bridge only has a verified AO-order mapping for degree <= 2 "
            f"(checked against mol.ao_labels()); degree={degree} would need the same "
            f"empirical check first, not an assumed extension of the pattern."
        )
    native_order = [tuple(int(x) for x in p) for p in cartesian_powers(degree)]
    libcint_order = _LIBCINT_CARTESIAN_ORDER[degree]
    return np.array([libcint_order.index(p) for p in native_order])


def _match_pyscf_shell_ao_starts(mol, shells: list) -> list:
    """For each native_hf shell (in native_hf's own list order), find the
    matching PySCF basis shell and return its AO start offset in PySCF's
    own ordering.

    Confirmed empirically on Ne/6-31G* that the two engines do NOT list
    shells in the same order: native_hf keeps the basis-definition's own
    file order (s,s,p,s,p,d -- interleaved), while libcint groups all
    shells of a given atom by ascending angular-momentum degree
    (s,s,s,p,p,d). Matching by identity (atom, degree, exponents) instead
    of assuming the two lists line up positionally is correct regardless
    of which grouping convention either engine happens to use, and isn't
    an assumption about libcint's grouping rule generalizing to other
    elements/bases."""
    ao_loc = mol.ao_loc_nr()
    candidates = [
        {
            "atom": mol.bas_atom(ib),
            "degree": mol.bas_angular(ib),
            "exponents": np.sort(np.asarray(mol.bas_exp(ib)))[::-1],
            "ao_start": int(ao_loc[ib]),
            "used": False,
        }
        for ib in range(mol.nbas)
    ]

    starts = []
    for shell in shells:
        shell_exponents = np.sort(np.asarray(shell.exponents))[::-1]
        match = next(
            (
                c
                for c in candidates
                if not c["used"]
                and c["atom"] == shell.atom_index
                and c["degree"] == shell.degree
                and c["exponents"].shape[0] == shell_exponents.shape[0]
                and np.allclose(c["exponents"], shell_exponents, rtol=1e-6)
            ),
            None,
        )
        if match is None:
            raise ValueError(
                f"could not match native_hf shell (atom={shell.atom_index}, "
                f"degree={shell.degree}, exponents={shell.exponents}) to any "
                f"PySCF basis shell -- the two engines may disagree on this "
                f"basis set's actual shell composition, not just its ordering."
            )
        match["used"] = True
        starts.append(match["ao_start"])
    return starts


def _import_pyscf():
    try:
        from pyscf import gto
    except ImportError as exc:  # pragma: no cover -- only reachable without pyscf installed, which CI here always has
        raise ImportError(
            "native_hf.libcint_bridge requires PySCF, an optional dependency "
            "(pip install dense-evolution[libcint]); it is not installed. "
            "PySCF has no Windows wheel -- installing it there requires a C "
            "toolchain to build from source."
        ) from exc
    return gto


def build_repulsion_tensor_libcint(atomic_numbers: list, geometry_bohr: np.ndarray, basis_name: str) -> np.ndarray:
    """Same contract as assembly.build_repulsion_tensor, but computed via
    PySCF/libcint instead of native_hf's own JAX recursions -- takes
    (atomic_numbers, geometry_bohr, basis_name) rather than a shells list
    since PySCF needs to build its own `Mole` from the same spec, not
    native_hf's ContractedShell objects.

    geometry_bohr: shape (n_atoms, 3), atomic units, same convention as
    build_molecule_shells."""
    gto = _import_pyscf()
    shells = build_molecule_shells(atomic_numbers, geometry_bohr, basis_name)
    n = n_cartesian_functions(shells)

    atom_spec = [
        (int(z), (float(r[0]), float(r[1]), float(r[2])))
        for z, r in zip(atomic_numbers, geometry_bohr)
    ]
    mol = gto.M(atom=atom_spec, basis=basis_name, unit="Bohr", cart=True)
    if mol.nao != n:
        raise ValueError(
            f"AO count mismatch between native_hf ({n}) and PySCF ({mol.nao}) "
            f"for basis {basis_name!r} -- the two engines disagree on this basis "
            f"set's shell composition, not just component ordering."
        )

    overlap_diag = np.diag(mol.intor("int1e_ovlp"))
    rescale = 1.0 / np.sqrt(overlap_diag)

    V = np.asarray(mol.intor("int2e", aosym="s1")).reshape(n, n, n, n)
    V = (
        V
        * rescale[:, None, None, None]
        * rescale[None, :, None, None]
        * rescale[None, None, :, None]
        * rescale[None, None, None, :]
    )

    ao_starts = _match_pyscf_shell_ao_starts(mol, shells)
    perm = np.empty(n, dtype=np.int64)
    offset = 0
    for shell, pyscf_start in zip(shells, ao_starts):
        block_size = len(cartesian_powers(shell.degree))
        perm[offset : offset + block_size] = pyscf_start + _permutation_libcint_to_native_hf(shell.degree)
        offset += block_size

    return V[perm][:, perm][:, :, perm][:, :, :, perm]
