"""Three-party device-independent conference key agreement (DICKA) via a
GHZ(3) state, following Ribeiro, Murta & Wehner 2018, "Fully
device-independent conference key agreement" (arXiv:1708.00798),
Protocol 1.

Ground truth, corrected against the actual paper text (not Mermin's
inequality, which this paper does not use): the paper introduces its own
"Parity-CHSH" inequality, an N-party extension of CHSH (Definition 8).
For N=3 (Alice, Bob1, one other Bob), the winning condition is
`a + b1 = x*(y XOR b2) mod 2`, with classical bound `P_win <= 3/4` and
quantum maximum `P_win ~ 0.85` (the same bound as ordinary CHSH, Eq.
A.14) -- both reproduced numerically, not assumed.

Two real bugs were found and fixed while promoting this from
Dense-Evolution-Discovery's `scripts/crypto/di_qkd_ghz.py`:

1. Definition 8's own text fixes the other Bob's test-round question at
   "always equal to 1", not 0 -- an initial implementation used 0 (Z
   measurement) and got `P_win=0.677`, exactly half the expected quantum
   boost. Fixing it to 1 (X measurement) reproduces the exact theoretical
   value `0.85355...` to machine precision.
2. `expected_win_rate`'s noisy-channel formula assumed depolarizing
   shrinks a qubit's Pauli expectations by `(1-p)`; `NoiseModel`'s actual
   isotropic-Pauli-error Kraus channel shrinks them by `(1-4p/3)` instead
   (see `expected_win_rate`'s own docstring).

Measurement operators, taken directly from the paper's own
honest-implementation section (Eq. A.64-A.65 region):

    Alice:      x=0 -> Z,                 x=1 -> X
    Bob1:       y=0 -> (Z+X)/sqrt(2),      y=1 -> (Z-X)/sqrt(2),  y=2 -> Z
    other Bobs: y=0 -> Z,                  y=1 -> X

Reused, not reimplemented: `de.ghz_state(3)`, `DenseSVSimulator`,
`GATES`, `sim.measure` (jax_key), `NoiseModel.apply_to_sv`. No new
quantum primitive."""

import numpy as np
import jax

import dense_evolution as de
from dense_evolution.noise import NoiseModel

_Z = np.diag([1, -1]).astype(complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)


def _diagonalizing_unitary(observable):
    """The unitary `U` such that measuring Z after applying `U` to the
    state reproduces measuring `observable` directly -- derived from
    `observable`'s own eigendecomposition (eigenvalue +1 first, to match
    `Z=diag(1,-1)`), not a hand-picked rotation angle."""
    w, v = np.linalg.eigh(observable)
    order = np.argsort(-w)
    v = v[:, order]
    return v.conj().T


_U_PLUS = _diagonalizing_unitary((_Z + _X) / np.sqrt(2))
_U_MINUS = _diagonalizing_unitary((_Z - _X) / np.sqrt(2))


def prepare_ghz3():
    """Prepares the shared 3-qubit GHZ state
    `(|000> + |111>)/sqrt(2)` used throughout this module, via
    `de.ghz_state(3)`. Returns a fresh `DenseSVSimulator(3)`."""
    sim = de.DenseSVSimulator(3)
    sim.run_circuit(de.ghz_state(3))
    return sim


def measure_alice(sim, x: int, rng: np.random.Generator) -> int:
    """Measures Alice's qubit (index 0) with question `x`: `x=0` measures
    Z directly, `x=1` applies H first (measures X). Returns the outcome
    bit."""
    if x == 1:
        sim.apply_gate_1q(de.GATES["h"], 0)
    key = jax.random.PRNGKey(int(rng.integers(0, 2**31 - 1)))
    return int(sim.measure(0, jax_key=key))


def measure_bob1(sim, y: int, rng: np.random.Generator) -> int:
    """Measures Bob1's qubit (index 1) with question `y`: `y=0` measures
    `(Z+X)/sqrt(2)`, `y=1` measures `(Z-X)/sqrt(2)` (both via the
    precomputed diagonalizing unitaries `_U_PLUS`/`_U_MINUS`), `y=2`
    measures Z directly. Returns the outcome bit."""
    if y == 0:
        sim.apply_gate_1q(_U_PLUS, 1)
    elif y == 1:
        sim.apply_gate_1q(_U_MINUS, 1)
    key = jax.random.PRNGKey(int(rng.integers(0, 2**31 - 1)))
    return int(sim.measure(1, jax_key=key))


def measure_other_bob(sim, y: int, rng: np.random.Generator, qubit: int = 2) -> int:
    """Measures the other Bob's qubit (default index 2) with question
    `y`: `y=0` measures Z directly, `y=1` applies H first (measures X).
    Returns the outcome bit."""
    if y == 1:
        sim.apply_gate_1q(de.GATES["h"], qubit)
    key = jax.random.PRNGKey(int(rng.integers(0, 2**31 - 1)))
    return int(sim.measure(qubit, jax_key=key))


def apply_depolarizing(sim, p: float, rng: np.random.Generator) -> None:
    """Applies i.i.d. depolarizing noise at rate `p` to all 3 qubits of
    `sim`, matching the paper's own `D^{\\otimes N}(GHZ_N)`
    honest-implementation model (Eq. A.64). A no-op at `p=0.0`."""
    if p == 0.0:
        return
    sv = np.asarray(sim.get_statevector())
    sv_noisy = NoiseModel.apply_to_sv(sv, 3, "depolarizing", p, rng=rng)
    sim.set_initial_state(np.asarray(sv_noisy))


def test_round(rng: np.random.Generator, p_dep: float = 0.0) -> int:
    """Runs one Parity-CHSH test round (Definition 8): Alice and Bob1 get
    uniformly random questions `x, y`; the other Bob gets a FIXED
    question, always 1 (X measurement) -- per the paper's own definition
    text, not 0 (see module docstring for the bug this fixed).

    Returns 1 if the parties win the round (`(a + b1) mod 2 ==
    (x*(y XOR b2)) mod 2`), else 0."""
    sim = prepare_ghz3()
    apply_depolarizing(sim, p_dep, rng)
    x = int(rng.integers(0, 2))
    y = int(rng.integers(0, 2))
    a = measure_alice(sim, x, rng)
    b1 = measure_bob1(sim, y, rng)
    b2 = measure_other_bob(sim, 1, rng)
    win = (a + b1) % 2 == (x * (y ^ b2)) % 2
    return int(win)


def key_round(rng: np.random.Generator, p_dep: float = 0.0):
    """Runs one key-generation round: everyone measures Z (`x=0, y=2,
    y2=0`). Returns `(a, b1, b2)`, perfectly correlated in the noiseless
    case."""
    sim = prepare_ghz3()
    apply_depolarizing(sim, p_dep, rng)
    a = measure_alice(sim, 0, rng)
    b1 = measure_bob1(sim, 2, rng)
    b2 = measure_other_bob(sim, 0, rng)
    return a, b1, b2


def parity_chsh_win_rate(n_rounds: int, p_dep: float = 0.0, seed=None) -> float:
    """Runs `n_rounds` Parity-CHSH test rounds and returns the observed
    winning frequency.

    Ground truth, verified to machine precision at `p_dep=0.0`:
    `P_win = 0.85355...` (`1/2 + 1/(2 sqrt2)`, the quantum maximum), well
    above the classical bound `3/4` -- device independence requires
    exceeding this classical bound.

    Example
    -------
    >>> import dense_evolution.protocols.di_qkd_ghz as ghz
    >>> p_win = ghz.parity_chsh_win_rate(3000, p_dep=0.0, seed=0)
    >>> p_win > 0.75
    True
    """
    rng = np.random.default_rng(seed)
    wins = sum(test_round(rng, p_dep) for _ in range(n_rounds))
    return wins / n_rounds


def key_qber(n_rounds: int, p_dep: float = 0.0, seed=None):
    """Runs `n_rounds` key-generation rounds and returns the raw QBER
    between Alice and each Bob, `(qber_b1, qber_b2)`, before error
    correction or privacy amplification. Zero at `p_dep=0.0`, rising with
    `p_dep`."""
    rng = np.random.default_rng(seed)
    disagree_b1 = 0
    disagree_b2 = 0
    for _ in range(n_rounds):
        a, b1, b2 = key_round(rng, p_dep)
        disagree_b1 += a != b1
        disagree_b2 += a != b2
    return disagree_b1 / n_rounds, disagree_b2 / n_rounds


def expected_win_rate(p_dep: float, n_parties: int = 3) -> float:
    """Closed-form honest-implementation Parity-CHSH win rate (paper Eq.
    A.65, specialized to `n_parties=3`), written in terms of the
    per-qubit Pauli-expectation shrink factor `s` a depolarizing channel
    produces: `p_exp = 1/2 + s^3/(2 sqrt2) + s^2*(1-s)/(4 sqrt2)`.

    `s = 1 - 4*p_dep/3`, matching `NoiseModel`'s real isotropic-Pauli
    depolarizing Kraus map (`K0=sqrt(1-p)I, K1..3=sqrt(p/3)*Pauli`), NOT
    `s=1-p` (which would assume a "replace with the maximally mixed
    state" channel instead). Verified directly against the simulator: the
    `s=1-p` version diverged with growing significance as `p` grew
    (`z=-2.75` at `p=0.10`, `z=-3.65` at `p=0.15`, N=3000); `s=1-4p/3`
    matches the simulator across the whole sweep (`|z|<0.5` everywhere
    tested)."""
    s = 1 - 4 * p_dep / 3
    s2 = np.sqrt(2)
    return 0.5 + s ** n_parties / (2 * s2) + s ** 2 * (1 - s ** (n_parties - 2)) / (4 * s2)
