"""
Thin wrappers around dense_evolution.protocols (crypto-q, promoted from
Dense-Evolution-Discovery issue #189) for the Composer kernel -- see
docs/api/protocols.md for the real, validated numbers behind each
protocol. Each function here just calls straight into the real protocol
module and returns a plain dict; no new cryptographic logic lives here.
"""
from dataclasses import dataclass

from dense_evolution.protocols import bb84 as _bb84
from dense_evolution.protocols import di_qkd_ghz as _di_qkd_ghz
from dense_evolution.protocols import dicka_protocol2 as _dicka

__all__ = ['Bb84Result', 'run_bb84', 'DiQkdGhzResult', 'run_di_qkd_ghz', 'run_dicka_protocol']


@dataclass
class Bb84Result:
    n_rounds: int
    p_channel: float
    eve: bool
    qber: float
    sifted_key_length: int


def run_bb84(n_rounds: int, p_channel: float = 0.0, eve: bool = False, seed=None) -> Bb84Result:
    """BB84: prepare -> channel -> measure -> sift -> QBER
    (dense_evolution.protocols.bb84.bb84_run). QBER=0 on a perfect
    channel, `2*p_channel/3` under isotropic depolarizing noise,
    `0.25` under an intercept-resend attack (`eve=True`) -- see
    docs/api/protocols.md for the validated numbers this reproduces."""
    qber, sifted_key_length = _bb84.bb84_run(n_rounds, p_channel, eve=eve, seed=seed)
    return Bb84Result(
        n_rounds=n_rounds, p_channel=p_channel, eve=eve,
        qber=float(qber), sifted_key_length=int(sifted_key_length),
    )


@dataclass
class DiQkdGhzResult:
    n_rounds: int
    p_dep: float
    win_rate: float
    expected_win_rate: float
    qber_b1: float
    qber_b2: float


def run_di_qkd_ghz(n_rounds: int, p_dep: float = 0.0, seed=None) -> DiQkdGhzResult:
    """Three-party device-independent conference key agreement via GHZ(3)
    (Ribeiro, Murta & Wehner 2018, arXiv:1708.00798;
    dense_evolution.protocols.di_qkd_ghz). `win_rate` is the real,
    simulated Parity-CHSH win rate over `n_rounds` test rounds;
    `expected_win_rate` is the exact theoretical value at this `p_dep`
    (quantum max `0.85355...` at `p_dep=0`) -- comparing the two directly
    checks the simulation against the closed-form prediction, not just
    against itself."""
    win_rate = _di_qkd_ghz.parity_chsh_win_rate(n_rounds, p_dep, seed=seed)
    qber_b1, qber_b2 = _di_qkd_ghz.key_qber(n_rounds, p_dep, seed=seed)
    return DiQkdGhzResult(
        n_rounds=n_rounds, p_dep=p_dep,
        win_rate=float(win_rate), expected_win_rate=float(_di_qkd_ghz.expected_win_rate(p_dep)),
        qber_b1=float(qber_b1), qber_b2=float(qber_b2),
    )


def run_dicka_protocol(n_rounds: int, gamma: float, beta: float, p_dep: float = 0.0, seed=None) -> dict:
    """Full multi-round DICKA structure (Appendix Protocol 2 of Ribeiro,
    Murta & Wehner 2018; dense_evolution.protocols.dicka_protocol2):
    round selection, parameter estimation, and the abort decision.
    Already returns a plain, JSON-safe dict -- see run_protocol's own
    docstring for exactly what each field means. Deliberately does not
    report a secure key length: Theorem 4's exact value depends on a
    numerical optimization the source paper never reduces to closed form."""
    result = _dicka.run_protocol(n_rounds=n_rounds, gamma=gamma, beta=beta, p_dep=p_dep, seed=seed)
    result["aborted"] = bool(result["aborted"])
    return result
