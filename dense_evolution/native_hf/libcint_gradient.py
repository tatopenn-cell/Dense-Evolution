"""
Hartree-Fock nuclear gradients with libcint integrals (RHF, UHF, CUHF).

The SCF is the library's own (`run_scf`, `run_uhf`, `run_cuhf`) on libcint
integrals. At the converged SCF the energy is stationary, so

    dE/dR_A = Tr[gS dS/dR_A] + Tr[gH dH/dR_A] + sum gV dV/dR_A + dV_nn/dR_A

with gS = -W (energy-weighted density), gH and gV the explicit derivatives of
the energy with respect to H_core and the repulsion tensor (Pople, Krishnan,
Schlegel & Binkley 1979, eq. 21; same densities and W as the custom VJPs in
scf.py). dS/dR and dV/dR come from libcint's ip1 integrals through the C driver
(`de_int1e_ip1_ovlp`, `de_int2e_ip1`); dH_core/dR is a central finite
difference of the libcint one-electron integrals (the nuclear-attraction
derivative integrals are not wired yet).
"""
import ctypes

import numpy as np

from dense_evolution.native_hf.libcint_bridge import _Cint, load_libcint
from dense_evolution.native_hf.scf import run_cuhf, run_scf, run_uhf

__all__ = ["hf_gradient_libcint"]


def _derivative_lib():
    lib = load_libcint()
    tail = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
    for name in ("de_int1e_ip1_ovlp", "de_int2e_ip1"):
        fn = getattr(lib, name)
        fn.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] + [ctypes.c_void_p] * 3 + tail
        fn.restype = None
    return lib


def _ip1(c, r, name, shape):
    lib = _derivative_lib()
    out = [np.zeros(shape) for _ in range(3)]
    getattr(lib, name)(out[0].ctypes.data, out[1].ctypes.data, out[2].ctypes.data, c.n,
                       c.ao_loc.ctypes.data, r.ctypes.data, c.pos.ctypes.data, *c._tail())
    return out


def _atom_of_ao(c):
    atoms = np.repeat(c.bas[:, 0], c.sizes)
    afo = np.empty(c.n, dtype=int)
    afo[c.pos] = atoms
    return afo


def _assemble_dS(M, on_atom):
    m = on_atom.astype(float)
    return m[:, None] * M.T + m[None, :] * M


def _assemble_dV(M, on_atom):
    m = on_atom.astype(float)
    return -(m[:, None, None, None] * M
             + m[None, :, None, None] * M.transpose(1, 0, 2, 3)
             + m[None, None, :, None] * M.transpose(2, 3, 0, 1)
             + m[None, None, None, :] * M.transpose(3, 2, 1, 0))


def _core_hamiltonian(atomic_numbers, geometry_bohr, basis_name):
    c = _Cint(atomic_numbers, geometry_bohr, basis_name)
    r = c.rescale()
    return c.one("kin", r, c.pos) + c.one("nuc", r, c.pos)


def _energy_derivatives(S, H, V, n_electrons, nuclear_charges, geometry, method, n_unpaired):
    if method == "rhf":
        res = run_scf(S, H, V, n_electrons, nuclear_charges, geometry)
        n_occ = n_electrons // 2
        C = np.asarray(res.orbital_coefficients)
        eps = np.asarray(res.orbital_energies)
        P = np.asarray(res.density_matrix)
        W = C[:, :n_occ] @ np.diag(2.0 * eps[:n_occ]) @ C[:, :n_occ].T
        gV = 2.0 * np.einsum("ab,cd->abcd", P, P) - np.einsum("ac,bd->abcd", P, P)
        return res.total_energy, -W, 2.0 * P, gV
    run = run_uhf if method == "uhf" else run_cuhf
    res = run(S, H, V, n_electrons, nuclear_charges, geometry, n_unpaired=n_unpaired)
    Pa = np.asarray(res.density_matrix_alpha)
    Pb = np.asarray(res.density_matrix_beta)
    W = np.zeros_like(Pa)
    for C, eps, k in ((res.orbital_coefficients_alpha, res.orbital_energies_alpha, res.n_alpha),
                      (res.orbital_coefficients_beta, res.orbital_energies_beta, res.n_beta)):
        C = np.asarray(C)
        W += C[:, :k] @ np.diag(np.asarray(eps)[:k]) @ C[:, :k].T
    Pt = Pa + Pb
    gV = (0.5 * np.einsum("ab,cd->abcd", Pt, Pt)
          - 0.5 * np.einsum("ac,bd->abcd", Pa, Pa)
          - 0.5 * np.einsum("ac,bd->abcd", Pb, Pb))
    return res.total_energy, -W, Pt, gV


def hf_gradient_libcint(atomic_numbers, nuclear_charges, n_electrons, basis_name, geometry_bohr,
                        method="rhf", n_unpaired=None, h_fd=1e-4):
    """Total HF energy and nuclear gradient (Hartree/Bohr, shape (n_atoms, 3)).

    Parameters
    ----------
    atomic_numbers, nuclear_charges, n_electrons, basis_name
        As in `differentiable.build_energy_fn`.
    geometry_bohr
        Nuclear positions, shape (n_atoms, 3), in Bohr.
    method
        "rhf", "uhf" or "cuhf" (ROHF as constrained UHF).
    n_unpaired
        Unpaired electrons for "uhf" / "cuhf"; default `n_electrons % 2`.
    h_fd
        Step of the central finite difference used for dH_core/dR, in Bohr.

    Returns
    -------
    (energy, gradient)
    """
    if method not in ("rhf", "uhf", "cuhf"):
        raise ValueError(f"method must be 'rhf', 'uhf' or 'cuhf', got {method!r}")
    if n_unpaired is None:
        n_unpaired = n_electrons % 2
    geometry = np.asarray(geometry_bohr, dtype=float)
    c = _Cint(atomic_numbers, geometry, basis_name)
    r = c.rescale()
    S = c.one("ovlp", r, c.pos)
    H = c.one("kin", r, c.pos) + c.one("nuc", r, c.pos)
    V = c.two(r)
    E, gS, gH, gV = _energy_derivatives(S, H, V, n_electrons, list(nuclear_charges), geometry,
                                        method, n_unpaired)
    Sx = _ip1(c, r, "de_int1e_ip1_ovlp", (c.n, c.n))
    Vx = _ip1(c, r, "de_int2e_ip1", (c.n,) * 4)
    afo = _atom_of_ao(c)
    q = np.asarray(nuclear_charges, dtype=float)
    grad = np.zeros_like(geometry)
    for A in range(len(atomic_numbers)):
        on_atom = afo == A
        for ax in range(3):
            gp, gm = geometry.copy(), geometry.copy()
            gp[A, ax] += h_fd
            gm[A, ax] -= h_fd
            dH = (_core_hamiltonian(atomic_numbers, gp, basis_name)
                  - _core_hamiltonian(atomic_numbers, gm, basis_name)) / (2.0 * h_fd)
            grad[A, ax] = (np.sum(gS * _assemble_dS(Sx[ax], on_atom))
                           + np.sum(gH * dH)
                           + np.sum(gV * _assemble_dV(Vx[ax], on_atom)))
        for B in range(len(atomic_numbers)):
            if B != A:
                d = geometry[A] - geometry[B]
                grad[A] -= q[A] * q[B] * d / np.linalg.norm(d) ** 3
    return float(E), grad
