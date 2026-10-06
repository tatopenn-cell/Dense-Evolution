"""Differentiable UHF and CUHF energies: gradient vs central finite
differences.

Closed-shell UHF/CUHF must reproduce the RHF energy and gradient.
Open-shell OH doublet and O2 triplet check the analytic gradient of
the UHF/CUHF energy against central finite differences of the same
energy function.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("basis_set_exchange")

from dense_evolution.native_hf.differentiable import build_energy_fn

_BOHR_PER_ANGSTROM = 1.8897259886


@pytest.fixture(autouse=True, scope="module")
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


GEOM_H2O = np.array([[0.0, 0.0, 0.2217],
                     [0.0, 1.4309, -0.8867],
                     [0.0, -1.4309, -0.8867]])
GEOM_OH = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.83]])
GEOM_O2 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.28]])


def _fd_dz(energy_fn, geom, h=1e-4):
    def e_of_z(z):
        g = np.array(geom, dtype=float); g[1, 2] = z
        return float(energy_fn(jnp.asarray(g)))
    z0 = float(geom[1, 2])
    return (e_of_z(z0 + h) - e_of_z(z0 - h)) / (2 * h)


def test_closed_shell_uhf_equals_rhf_energy_and_gradient():
    fn_rhf = build_energy_fn([8, 1, 1], [8., 1., 1.], 10, "sto-3g", GEOM_H2O, method="rhf")
    fn_uhf = build_energy_fn([8, 1, 1], [8., 1., 1.], 10, "sto-3g", GEOM_H2O, method="uhf", n_unpaired=0)
    g = jnp.asarray(GEOM_H2O)
    e_r = float(fn_rhf(g)); e_u = float(fn_uhf(g))
    assert abs(e_u - e_r) < 1e-9
    gr = jax.grad(fn_rhf)(g); gu = jax.grad(fn_uhf)(g)
    assert float(jnp.max(jnp.abs(gr - gu))) < 1e-6


def test_closed_shell_cuhf_equals_rhf_energy_and_gradient():
    fn_rhf  = build_energy_fn([8, 1, 1], [8., 1., 1.], 10, "sto-3g", GEOM_H2O, method="rhf")
    fn_cuhf = build_energy_fn([8, 1, 1], [8., 1., 1.], 10, "sto-3g", GEOM_H2O, method="cuhf", n_unpaired=0)
    g = jnp.asarray(GEOM_H2O)
    assert abs(float(fn_cuhf(g)) - float(fn_rhf(g))) < 1e-9
    gr = jax.grad(fn_rhf)(g); gc = jax.grad(fn_cuhf)(g)
    assert float(jnp.max(jnp.abs(gr - gc))) < 1e-6


def test_oh_doublet_uhf_gradient_matches_finite_difference():
    fn = build_energy_fn([8, 1], [8., 1.], 9, "sto-3g", GEOM_OH, method="uhf", n_unpaired=1)
    g = jnp.asarray(GEOM_OH)
    an = float(jax.grad(fn)(g)[1, 2])
    fd = _fd_dz(fn, GEOM_OH, h=1e-4)
    assert an == pytest.approx(fd, abs=1e-5)


def test_oh_doublet_cuhf_gradient_matches_finite_difference():
    fn = build_energy_fn([8, 1], [8., 1.], 9, "sto-3g", GEOM_OH, method="cuhf", n_unpaired=1)
    g = jnp.asarray(GEOM_OH)
    an = float(jax.grad(fn)(g)[1, 2])
    fd = _fd_dz(fn, GEOM_OH, h=1e-4)
    assert an == pytest.approx(fd, abs=1e-5)


def test_o2_triplet_uhf_gradient_matches_finite_difference():
    fn = build_energy_fn([8, 8], [8., 8.], 16, "sto-3g", GEOM_O2, method="uhf", n_unpaired=2)
    g = jnp.asarray(GEOM_O2)
    an = float(jax.grad(fn)(g)[1, 2])
    fd = _fd_dz(fn, GEOM_O2, h=1e-4)
    assert an == pytest.approx(fd, abs=1e-5)


def test_o2_triplet_cuhf_gradient_matches_finite_difference():
    fn = build_energy_fn([8, 8], [8., 8.], 16, "sto-3g", GEOM_O2, method="cuhf", n_unpaired=2)
    g = jnp.asarray(GEOM_O2)
    an = float(jax.grad(fn)(g)[1, 2])
    fd = _fd_dz(fn, GEOM_O2, h=1e-4)
    assert an == pytest.approx(fd, abs=1e-5)