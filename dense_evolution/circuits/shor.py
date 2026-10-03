"""
Order finding for Shor's algorithm with one control qubit (Beauregard,
quant-ph/0205095, Sect. 2.5, Fig. 8): 2n + 3 qubits in total.

The 2n control qubits of the textbook circuit are replaced by a single one,
measured and reset after each controlled multiplication; the inverse QFT is
done semiclassically, a phase gate on that qubit set by the outcomes already
measured, then a Hadamard and the measurement. Each controlled-U_{a^(2^k)}
is the gate circuit of `controlled_ua_qasm`.
"""
import numpy as np

from .arithmetic_qasm import controlled_ua_qasm
from .parser import QASMParser

__all__ = ["shor_order_finding"]


def _hadamard_on_first(sv):
    return np.stack([sv[0] + sv[1], sv[0] - sv[1]]) / np.sqrt(2.0)


def shor_order_finding(a, N, rng=None):
    """
    One run of quantum order finding for a modulo N, returns
    (y, m): a measured integer 0 <= y < 2^m with m = 2n, n = N.bit_length().
    y / 2^m is close to s / r for the order r of a and a random s, so the
    continued-fraction expansion of y / 2^m recovers r with good probability
    (`fractions.Fraction(y, 2**m).limit_denominator(N)`).

    Iteration k = m-1, ..., 0 prepares the control in |+>, applies
    controlled-U_{a^(2^k) mod N} to the work register (initially |1>),
    applies the phase -2 pi sum_l r_l / 2^(l-k+1) over the bits r_l already
    measured, a Hadamard, then measures and resets the control.
    """
    from ..backends.statevector import DenseSVSimulator

    a, N = int(a), int(N)
    n = N.bit_length()
    m, nq = 2 * n, 2 * n + 3
    rng = np.random.default_rng(rng)
    sv = np.zeros((2, 2 ** (nq - 1)), dtype=np.complex128)
    sv[0, 1 << (nq - 2)] = 1.0
    bits = {}
    for k in range(m - 1, -1, -1):
        sv = _hadamard_on_first(sv)
        ak = pow(a, 2 ** k, N)
        if ak != 1:
            circ = QASMParser().parse(controlled_ua_qasm(ak, N, n))
            sim = DenseSVSimulator(nq)
            sim.set_initial_state(sv.reshape(-1))
            sim.run_circuit_jit(circ.to_tuples())
            sv = np.asarray(sim.get_statevector()).reshape(2, -1)
        phase = -2 * np.pi * sum(bits[l] / 2 ** (l - k + 1) for l in bits)
        sv[1] *= np.exp(1j * phase)
        sv = _hadamard_on_first(sv)
        p1 = float(np.sum(np.abs(sv[1]) ** 2))
        bits[k] = int(rng.random() < p1)
        kept = sv[bits[k]] / np.sqrt(p1 if bits[k] else 1.0 - p1)
        sv = np.stack([kept, np.zeros_like(kept)])
    y = sum(r << (m - 1 - k) for k, r in bits.items())
    return y, m
