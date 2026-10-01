"""libcint bridge for native_hf's one- and two-electron integrals.

native_hf's own build_repulsion_tensor (assembly.py) computes the ERI
tensor via JAX-jitted Obara-Saika recursions -- correct, differentiable,
but pays a JIT-compilation tax on mixed-angular-momentum bases (~532s on
Ne/6-31G* even after the primitive-count-padding fix), and its one-electron
assembly ran out of memory during XLA compilation for a 4-heavy-atom
molecule at 6-31G* on a Kaggle CPU kernel. libcint (Sun, J. Comput. Chem.
36, 1664 (2015), BSD-2) computes the identical integrals with C kernels
compiled once, ahead of time.

libcint is linked statically, together with csrc/cint_driver.c, into one
shared library shipped inside the platform wheels of dense-evolution
(dense_evolution/native_hf/_libcint/, built by
.github/scripts/build-libcint.sh in .github/workflows/libcint.yml) and
called through ctypes, with the atm/bas/env arrays built here from
native_hf's own ContractedShell list. The driver loops over shell blocks
in C (8-fold ERI symmetry, int2e_optimizer) and writes every integral
already rescaled and in native_hf's AO order. A source install has no
bundled library: run build-libcint.sh and point DENSE_EVOLUTION_LIBCINT at
the resulting file.

Shells are passed to libcint in native_hf's own order, so AO blocks line
up shell by shell. Two convention differences remain inside each shell,
confirmed empirically on Ne/6-31G*:
  - Cartesian component order: libcint orders px,py,pz and
    xx,xy,xz,yy,yz,zz; native_hf's cartesian_powers orders px,pz,py and
    xx,xz,xy,zz,yz,yy.
  - Normalization: libcint's raw Cartesian d components are not unit
    self-overlap (xx/yy/zz vs. xy/xz/yz differ by exactly a factor of 3).
    native_hf's primitive-normalized coefficients differ from libcint's
    radial convention by a per-shell constant too. Both are removed by a
    per-AO rescale computed from libcint's own overlap diagonal at call
    time, exact for whatever exponent/element/degree is in play.
Degrees above 2 raise NotImplementedError (native_hf itself caps at 2).
"""
import ctypes
import os
import sys
from pathlib import Path

import numpy as np

from dense_evolution.native_hf.basis import build_molecule_shells
from dense_evolution.native_hf.cartesian import cartesian_powers

_LIBCINT_CARTESIAN_ORDER = {
    0: [(0, 0, 0)],
    1: [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
    2: [(2, 0, 0), (1, 1, 0), (1, 0, 1), (0, 2, 0), (0, 1, 1), (0, 0, 2)],
}
_LIB_NAMES = {"win32": "libdecint.dll", "darwin": "libdecint.dylib"}
_PTR_ENV_START = 20
_POINT_NUC = 1
_KINDS = {"ovlp": 0, "kin": 1, "nuc": 2}
_lib = None


def load_libcint() -> ctypes.CDLL:
    """The bundled libcint + driver shared library (or DENSE_EVOLUTION_LIBCINT's),
    loaded once; ImportError if neither exists."""
    global _lib
    if _lib is not None:
        return _lib
    path = os.environ.get("DENSE_EVOLUTION_LIBCINT") or str(
        Path(__file__).with_name("_libcint") / _LIB_NAMES.get(sys.platform, "libdecint.so")
    )
    if not os.path.isfile(path):
        raise ImportError(
            f"native_hf.libcint_bridge needs the bundled libcint library, expected at {path}. "
            "It ships inside the dense-evolution wheels for Windows, macOS and Linux "
            "(pip install dense-evolution); for a source install, run "
            ".github/scripts/build-libcint.sh and set DENSE_EVOLUTION_LIBCINT to the library file."
        )
    lib = ctypes.CDLL(path)
    tail = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
    lib.de_int1e.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int] + [ctypes.c_void_p] * 3 + tail
    lib.de_int1e.restype = None
    lib.de_int2e.argtypes = [ctypes.c_void_p, ctypes.c_int] + [ctypes.c_void_p] * 3 + tail
    lib.de_int2e.restype = None
    _lib = lib
    return lib


def _permutation_libcint_to_native_hf(degree: int) -> np.ndarray:
    """perm such that native_hf_ordered[i] == libcint_ordered[perm[i]]."""
    if degree not in _LIBCINT_CARTESIAN_ORDER:
        raise NotImplementedError(
            f"libcint_bridge only has a verified AO-order mapping for degree <= 2; "
            f"degree={degree} would need the same empirical check first, not an "
            f"assumed extension of the pattern."
        )
    native_order = [tuple(int(x) for x in p) for p in cartesian_powers(degree)]
    libcint_order = _LIBCINT_CARTESIAN_ORDER[degree]
    return np.array([libcint_order.index(p) for p in native_order])


class _Cint:
    """atm/bas/env arrays for one molecule plus per-shell AO offsets, the
    native_hf AO permutation, evaluated by the C driver."""

    def __init__(self, atomic_numbers: list, geometry_bohr: np.ndarray, basis_name: str):
        self.lib = load_libcint()
        shells = build_molecule_shells(atomic_numbers, geometry_bohr, basis_name)
        env = [0.0] * _PTR_ENV_START
        atm, bas = [], []
        for z, r in zip(atomic_numbers, np.asarray(geometry_bohr, dtype=float)):
            atm.append([int(z), len(env), _POINT_NUC, len(env) + 3, 0, 0])
            env.extend([r[0], r[1], r[2], 0.0])
        for s in shells:
            e = np.asarray(s.exponents, dtype=float)
            c = np.asarray(s.coefficients, dtype=float)
            bas.append([s.atom_index, s.degree, e.shape[0], 1, 0, len(env), len(env) + e.shape[0], 0])
            env.extend(e)
            env.extend(c)
        self.atm = np.ascontiguousarray(atm, dtype=np.int32)
        self.bas = np.ascontiguousarray(bas, dtype=np.int32)
        self.env = np.ascontiguousarray(env, dtype=np.float64)
        self.sizes = [len(cartesian_powers(s.degree)) for s in shells]
        self.offsets = np.concatenate([[0], np.cumsum(self.sizes)]).astype(int)
        self.n = int(self.offsets[-1])
        self.perm = np.concatenate(
            [o + _permutation_libcint_to_native_hf(s.degree) for o, s in zip(self.offsets, shells)]
        )
        self.ao_loc = np.ascontiguousarray(self.offsets, dtype=np.int32)
        self.pos = np.ascontiguousarray(np.argsort(self.perm), dtype=np.int32)

    def _tail(self):
        return (self.atm.ctypes.data, self.atm.shape[0], self.bas.ctypes.data, self.bas.shape[0], self.env.ctypes.data)

    def one(self, kind: str, r: np.ndarray, pos: np.ndarray) -> np.ndarray:
        out = np.empty((self.n, self.n))
        r = np.ascontiguousarray(r, dtype=np.float64)
        self.lib.de_int1e(_KINDS[kind], out.ctypes.data, self.n, self.ao_loc.ctypes.data, r.ctypes.data, pos.ctypes.data, *self._tail())
        return out

    def rescale(self) -> np.ndarray:
        S = self.one("ovlp", np.ones(self.n), np.arange(self.n, dtype=np.int32))
        return 1.0 / np.sqrt(np.diag(S))

    def two(self, r: np.ndarray) -> np.ndarray:
        out = np.empty((self.n,) * 4)
        self.lib.de_int2e(out.ctypes.data, self.n, self.ao_loc.ctypes.data, r.ctypes.data, self.pos.ctypes.data, *self._tail())
        return out


def build_overlap_and_core_hamiltonian_libcint(
    atomic_numbers: list, geometry_bohr: np.ndarray, basis_name: str
) -> tuple[np.ndarray, np.ndarray]:
    """Same contract as assembly.build_overlap_matrix + build_core_hamiltonian
    combined, computed via libcint. Returns (S, H_core) in native_hf's own AO
    ordering/normalization, ready for scf.run_scf alongside
    build_repulsion_tensor_libcint's output."""
    c = _Cint(atomic_numbers, geometry_bohr, basis_name)
    r = c.rescale()
    return c.one("ovlp", r, c.pos), c.one("kin", r, c.pos) + c.one("nuc", r, c.pos)


def build_repulsion_tensor_libcint(atomic_numbers: list, geometry_bohr: np.ndarray, basis_name: str) -> np.ndarray:
    """Same contract as assembly.build_repulsion_tensor, computed via libcint
    with 8-fold permutational symmetry; takes (atomic_numbers, geometry_bohr,
    basis_name) rather than a shells list.

    geometry_bohr: shape (n_atoms, 3), atomic units, same convention as
    build_molecule_shells."""
    c = _Cint(atomic_numbers, geometry_bohr, basis_name)
    return c.two(c.rescale())
