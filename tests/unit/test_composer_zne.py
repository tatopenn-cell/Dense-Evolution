"""docs/assets/composer/py/zne_numpy.py against the functions of the package and of the dashboard.

The page runs the zero-noise extrapolation of the dashboard (run_zne_mitigation) without JAX. The
references are the package's own functions, which the page file does not compute.
"""

import importlib.util
import itertools
from pathlib import Path

import numpy as np
import pytest

import dense_evolution as de

PY = Path(__file__).resolve().parents[2] / "docs" / "assets" / "composer" / "py"


@pytest.fixture(scope="module")
def zn():
    import sys

    sys.path.insert(0, str(PY))
    try:
        spec = importlib.util.spec_from_file_location("zne_numpy", PY / "zne_numpy.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(PY))
    return module


def _state(n, seed):
    rng = np.random.default_rng(seed)
    sv = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
    return sv / np.linalg.norm(sv)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_the_pauli_expectation_is_the_one_of_the_package_for_every_string(zn, n):
    sv = _state(n, 5 + n)
    for chars in itertools.product("IXYZ", repeat=n):
        pauli = "".join(chars)
        want = float(np.real(de.pauli_expectation(sv, pauli)))
        assert zn.pauli_expectation(sv, pauli) == pytest.approx(want, abs=1e-12), pauli


def test_richardson_is_the_one_of_the_package(zn):
    assert zn.richardson_at_zero([0.9, 0.8, 0.65], [1.0, 2.0, 3.0]) == pytest.approx(0.95, abs=1e-12)
    values, factors = [0.7, 0.55, 0.43, 0.31, 0.2], [1.0, 1.5, 2.5, 3.0, 4.5]
    assert zn.richardson_at_zero(values, factors) == pytest.approx(
        float(de.zero_noise_extrapolation(values, factors)), abs=1e-9
    )


def test_the_polynomial_fit_is_the_one_of_the_package(zn):
    values, factors = [0.7, 0.55, 0.43, 0.31, 0.2], [1.0, 2.0, 3.0, 4.0, 5.0]
    assert zn.polynomial_at_zero(values, factors) == pytest.approx(
        float(de.polynomial_extrapolate(values, factors, degree=2)), abs=1e-9
    )


BELL = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[2];\nh q[0];\ncx q[0],q[1];\nmeasure q -> c;\n'


@pytest.mark.parametrize("model", ["depolarizing", "bitflip", "phaseflip", "amplitude_damping", "combined"])
@pytest.mark.parametrize("method", ["richardson", "polynomial"])
def test_the_page_zne_gives_the_numbers_of_run_zne_mitigation(zn, model, method):
    from dashboard_core.mitigation import run_zne_mitigation

    want = run_zne_mitigation(BELL, "ZZ", model, 0.05, seed=3, n_trials=20, extrapolation_method=method)
    parsed = de.QASMParser().parse(BELL)
    sim = de.DenseSVSimulator(parsed.n_qubits, use_float32=False)
    sim.run_circuit(parsed.to_tuples())
    got = zn.zne_from_state(np.asarray(sim.sv), 2, "ZZ", model, 0.05, seed=3, method=method, n_trials=20)
    assert got["noise_factors"] == want.noise_factors
    assert got["ideal_expectation"] == pytest.approx(want.ideal_expectation, abs=1e-12)
    assert got["noisy_expectations"] == pytest.approx(want.noisy_expectations, abs=1e-9)
    assert got["zne_extrapolated"] == pytest.approx(want.zne_extrapolated, abs=1e-9)


def test_wrong_input_is_refused_as_run_zne_mitigation_refuses_it(zn):
    sv = _state(2, 1)
    with pytest.raises(ValueError):
        zn.zne_from_state(sv, 2, "Z", "bitflip", 0.1)
    with pytest.raises(ValueError):
        zn.zne_from_state(sv, 2, "ZZ", "nonsense", 0.1)
    with pytest.raises(ValueError):
        zn.zne_from_state(sv, 2, "ZZ", "bitflip", 0.1, method="spline")
