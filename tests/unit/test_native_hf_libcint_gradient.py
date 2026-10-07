"""HF nuclear gradients with libcint integrals vs central finite differences of the energy."""
import numpy as np
import pytest

pytest.importorskip("basis_set_exchange")

from dense_evolution.native_hf import libcint_bridge  # noqa: E402

try:
    _lib = libcint_bridge.load_libcint()
    _HAS_DERIVATIVES = hasattr(_lib, "de_int2e_ip1")
except ImportError:
    _HAS_DERIVATIVES = False

pytestmark = pytest.mark.skipif(not _HAS_DERIVATIVES, reason="libcint driver without ip1 derivatives")

from dense_evolution.native_hf.libcint_gradient import hf_gradient_libcint  # noqa: E402

BOHR = 1.8897259886
CASES = [
    ("rhf", [8, 1, 1], [8.0, 1.0, 1.0], 10, np.array([[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]]) * BOHR, None),
    ("uhf", [8, 1], [8.0, 1.0], 9, np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.83]]), 1),
    ("cuhf", [8, 8], [8.0, 8.0], 16, np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.28]]), 2),
]


@pytest.mark.parametrize("method, Z, q, ne, geom, nu", CASES)
def test_gradient_matches_finite_differences(method, Z, q, ne, geom, nu):
    E, g = hf_gradient_libcint(Z, q, ne, "sto-3g", geom, method=method, n_unpaired=nu)
    h = 1e-4
    fd = np.zeros_like(geom)
    for i in range(geom.shape[0]):
        for ax in range(3):
            gp, gm = geom.copy(), geom.copy()
            gp[i, ax] += h
            gm[i, ax] -= h
            fd[i, ax] = (hf_gradient_libcint(Z, q, ne, "sto-3g", gp, method=method, n_unpaired=nu)[0]
                         - hf_gradient_libcint(Z, q, ne, "sto-3g", gm, method=method, n_unpaired=nu)[0]) / (2 * h)
    assert np.max(np.abs(g - fd)) < 1e-5
    assert np.allclose(g.sum(axis=0), 0.0, atol=1e-6)


def test_invalid_method():
    with pytest.raises(ValueError):
        hf_gradient_libcint([1, 1], [1.0, 1.0], 2, "sto-3g", np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]]), method="mp2")
