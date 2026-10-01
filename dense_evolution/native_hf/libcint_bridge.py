"""libcint bridge for native_hf's one- and two-electron integrals.

native_hf's own build_repulsion_tensor (assembly.py) computes the ERI
tensor via JAX-jitted Obara-Saika recursions -- correct, differentiable,
but pays a JIT-compilation tax on mixed-angular-momentum bases (~532s on
Ne/6-31G* even after the primitive-count-padding fix), and its one-electron
assembly ran out of memory during XLA compilation for a 4-heavy-atom
molecule at 6-31G* on a Kaggle CPU kernel. libcint (Sun, J. Comput. Chem.
36, 1664 (2015), BSD-2) computes the identical integrals with C kernels
compiled once, ahead of time.

The libcint shared library ships inside the platform wheels of
dense-evolution (dense_evolution/native_hf/_libcint/, built by
.github/workflows/libcint.yml) and is called directly through ctypes, with
the atm/bas/env arrays built here from native_hf's own ContractedShell
list. A source install has no bundled library: build libcint and point
DENSE_EVOLUTION_LIBCINT at the shared library file.

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
_LIB_NAMES = {"win32": "libcint.dll", "darwin": "libcint.dylib"}
_PTR_ENV_START = 20
_POINT_NUC = 1
_INTEGRALS = ("int1e_ovlp_cart", "int1e_kin_cart", "int1e_nuc_cart", "int2e_cart")
_lib = None


def load_libcint() -> ctypes.CDLL:
    """The bundled libcint shared library (or DENSE_EVOLUTION_LIBCINT's),
    loaded once; ImportError if neither exists."""
    global _lib
    if _lib is not None:
        return _lib
    path = os.environ.get("DENSE_EVOLUTION_LIBCINT") or str(
        Path(__file__).with_name("_libcint") / _LIB_NAMES.get(sys.platform, "libcint.so")
    )
    if not os.path.isfile(path):
        raise ImportError(
            f"native_hf.libcint_bridge needs the libcint shared library, expected at {path}. "
            "It ships inside the dense-evolution wheels for Windows, macOS and Linux "
            "(pip install dense-evolution); for a source install, build libcint "
            "(https://github.com/sunqm/libcint) and set DENSE_EVOLUTION_LIBCINT to the library file."
        )
    lib = ctypes.CDLL(path)
    argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int, ctypes.c_void_p, ctypes.c_int] + [ctypes.c_void_p] * 3
    for name in _INTEGRALS:
        fn = getattr(lib, name)
        fn.argtypes = argtypes
        fn.restype = ctypes.c_size_t
    lib.int2e_optimizer.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
    lib.int2e_optimizer.restype = None
    lib.CINTdel_optimizer.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    lib.CINTdel_optimizer.restype = None
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
    native_hf AO permutation, and block-wise integral evaluation."""

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
        self.shls = np.zeros(4, dtype=np.int32)
        self.buf = np.empty(max(self.sizes) ** 4)
        self._args = (
            self.shls.ctypes.data,
            self.atm.ctypes.data,
            self.atm.shape[0],
            self.bas.ctypes.data,
            self.bas.shape[0],
            self.env.ctypes.data,
        )

    def _call(self, fn, opt=None):
        s, a, na, b, nb, e = self._args
        fn(self.buf.ctypes.data, None, s, a, na, b, nb, e, opt, None)

    def one(self, name: str) -> np.ndarray:
        fn = getattr(self.lib, name)
        out = np.empty((self.n, self.n))
        o, d = self.offsets, self.sizes
        for i in range(len(d)):
            for j in range(len(d)):
                self.shls[:2] = (i, j)
                self._call(fn)
                out[o[i] : o[i] + d[i], o[j] : o[j] + d[j]] = self.buf[: d[i] * d[j]].reshape(d[j], d[i]).T
        return out

    def two(self) -> np.ndarray:
        fn = self.lib.int2e_cart
        opt = ctypes.c_void_p()
        s, a, na, b, nb, e = self._args
        self.lib.int2e_optimizer(ctypes.byref(opt), a, na, b, nb, e)
        out = np.empty((self.n,) * 4)
        o, d = self.offsets, self.sizes
        sl = [slice(o[i], o[i] + d[i]) for i in range(len(d))]
        try:
            for i in range(len(d)):
                for j in range(i + 1):
                    for k in range(i + 1):
                        for l in range(k + 1 if k < i else j + 1):
                            self.shls[:] = (i, j, k, l)
                            self._call(fn, opt)
                            blk = self.buf[: d[i] * d[j] * d[k] * d[l]].reshape(d[l], d[k], d[j], d[i]).T
                            I, J, K, L = sl[i], sl[j], sl[k], sl[l]
                            out[I, J, K, L] = blk
                            out[J, I, K, L] = blk.transpose(1, 0, 2, 3)
                            out[I, J, L, K] = blk.transpose(0, 1, 3, 2)
                            out[J, I, L, K] = blk.transpose(1, 0, 3, 2)
                            out[K, L, I, J] = blk.transpose(2, 3, 0, 1)
                            out[L, K, I, J] = blk.transpose(3, 2, 0, 1)
                            out[K, L, J, I] = blk.transpose(2, 3, 1, 0)
                            out[L, K, J, I] = blk.transpose(3, 2, 1, 0)
        finally:
            self.lib.CINTdel_optimizer(ctypes.byref(opt))
        return out

    def rescale(self) -> np.ndarray:
        return 1.0 / np.sqrt(np.diag(self.one("int1e_ovlp_cart")))


def build_overlap_and_core_hamiltonian_libcint(
    atomic_numbers: list, geometry_bohr: np.ndarray, basis_name: str
) -> tuple[np.ndarray, np.ndarray]:
    """Same contract as assembly.build_overlap_matrix + build_core_hamiltonian
    combined, computed via libcint. Returns (S, H_core) in native_hf's own AO
    ordering/normalization, ready for scf.run_scf alongside
    build_repulsion_tensor_libcint's output."""
    c = _Cint(atomic_numbers, geometry_bohr, basis_name)
    S = c.one("int1e_ovlp_cart")
    r = 1.0 / np.sqrt(np.diag(S))
    scale = r[:, None] * r[None, :]
    H_core = (c.one("int1e_kin_cart") + c.one("int1e_nuc_cart")) * scale
    p = c.perm
    return (S * scale)[p][:, p], H_core[p][:, p]


def build_repulsion_tensor_libcint(atomic_numbers: list, geometry_bohr: np.ndarray, basis_name: str) -> np.ndarray:
    """Same contract as assembly.build_repulsion_tensor, computed via libcint
    with 8-fold permutational symmetry; takes (atomic_numbers, geometry_bohr,
    basis_name) rather than a shells list.

    geometry_bohr: shape (n_atoms, 3), atomic units, same convention as
    build_molecule_shells."""
    c = _Cint(atomic_numbers, geometry_bohr, basis_name)
    r = c.rescale()
    V = c.two() * r[:, None, None, None] * r[None, :, None, None] * r[None, None, :, None] * r[None, None, None, :]
    p = c.perm
    return V[p][:, p][:, :, p][:, :, :, p]
