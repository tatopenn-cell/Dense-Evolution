"""Zero-noise extrapolation in NumPy, for the page: ``tools/dashboard/core/mitigation.run_zne_mitigation``.

The same steps, without JAX: <P> on the ideal state; at each noise scale the mean of ``n_trials`` noisy
trajectories of ``noise_numpy.apply_to_sv`` at ``noise_p * factor`` (the same generator, drawn in the same
order); then an extrapolation to zero noise, the Lagrange polynomial through the points (Richardson, the
unique interpolating polynomial, Temme et al. 2017) or the least-squares polynomial of degree 2 through five
points. A test runs this file and ``run_zne_mitigation`` on the same state and the same seed and compares
the numbers; another compares each function with the one of the package.
"""

import numpy as np

try:
    from . import noise_numpy as nz
except ImportError:
    import noise_numpy as nz

RICHARDSON_FACTORS = (1.0, 2.0, 3.0)
POLYNOMIAL_FACTORS = (1.0, 2.0, 3.0, 4.0, 5.0)
POLYNOMIAL_DEGREE = 2


def pauli_expectation(sv, pauli):
    """<sv|P|sv> for a Pauli string with qubit 0 the leftmost character and the most significant bit of the index."""
    n = len(pauli)
    if len(sv) != 2**n:
        raise ValueError("a statevector of %d amplitudes does not match %d characters" % (len(sv), n))
    idx = np.arange(len(sv), dtype=np.intp)
    flip = 0
    phase = np.ones(len(sv), dtype=np.complex128)
    for q, ch in enumerate(pauli):
        bit = n - 1 - q
        b = (idx >> bit) & 1
        if ch == "X":
            flip |= 1 << bit
        elif ch == "Y":
            flip |= 1 << bit
            phase = phase * np.where(b == 0, 1j, -1j)
        elif ch == "Z":
            phase = phase * (1 - 2 * b)
        elif ch != "I":
            raise ValueError("unknown Pauli character %r" % ch)
    out = np.empty_like(np.asarray(sv, dtype=np.complex128))
    out[idx ^ flip] = phase * sv
    return float(np.real(np.vdot(sv, out)))


def richardson_at_zero(values, factors):
    """The Lagrange polynomial through (factor, value), evaluated at 0."""
    lam = np.asarray(factors, dtype=np.float64)
    total = 0.0
    for i, y in enumerate(values):
        others = np.delete(lam, i)
        total += y * np.prod((0.0 - others) / (lam[i] - others))
    return float(total)


def polynomial_at_zero(values, factors, degree=POLYNOMIAL_DEGREE):
    """The least-squares polynomial of ``degree`` through (factor, value), evaluated at 0."""
    lam = np.asarray(factors, dtype=np.float64)
    design = np.stack([lam**k for k in range(degree + 1)], axis=1)
    coeffs = np.linalg.lstsq(design, np.asarray(values, dtype=np.float64), rcond=None)[0]
    return float(coeffs[0])


def zne_from_state(sv, n, pauli, model, noise_p, seed=None, method="richardson", n_trials=200, factors=None):
    """The result of ``run_zne_mitigation`` for the ideal statevector ``sv`` of ``n`` qubits."""
    if method not in ("richardson", "polynomial"):
        raise ValueError("unknown extrapolation_method %r, must be 'richardson' or 'polynomial'" % method)
    if len(pauli) != n:
        raise ValueError("pauli_string length %d != n_qubits %d" % (len(pauli), n))
    if model not in nz.MODELS:
        raise ValueError("unknown noise model %r, must be one of %s" % (model, nz.MODELS))
    if factors is None:
        factors = POLYNOMIAL_FACTORS if method == "polynomial" else RICHARDSON_FACTORS
    sv = np.asarray(sv, dtype=np.complex128)
    ideal = pauli_expectation(sv, pauli)
    rng = np.random.default_rng(seed)
    means, sems = [], []
    for factor in factors:
        scaled = min(noise_p * factor, 1.0)
        trial = np.empty(n_trials)
        for i in range(n_trials):
            trial[i] = pauli_expectation(nz.apply_to_sv(sv.copy(), n, model, scaled, rng), pauli)
        means.append(float(trial.mean()))
        sems.append(float(trial.std(ddof=1) / np.sqrt(n_trials)))
    if method == "polynomial":
        zne = polynomial_at_zero(means, factors)
    else:
        zne = richardson_at_zero(means, factors)
    return {
        "n_qubits": n,
        "pauli_string": pauli,
        "ideal_expectation": ideal,
        "noise_factors": list(factors),
        "noisy_expectations": means,
        "noisy_sems": sems,
        "zne_extrapolated": zne,
        "extrapolation_method": method,
    }
