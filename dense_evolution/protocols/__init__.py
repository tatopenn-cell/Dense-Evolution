"""Quantum cryptography protocols subpackage (crypto-q, issue #189 in
Dense-Evolution-Discovery): BB84 (`bb84`), three-party device-independent
conference key agreement via a GHZ(3) state and its Parity-CHSH game
(`di_qkd_ghz`), and the full multi-round DICKA structure built on top of
it (`dicka_protocol2`). Each is built entirely from existing simulator
primitives (statevector, gates, measurement, the depolarizing channel) --
no new quantum channel was added to the core to support this subpackage.

Promoted from Dense-Evolution-Discovery only after passing that project's
own promotion checklist: a theoretical ground truth verified to tight
tolerance, a falsifiable result, primitives-only implementation, fixed-
seed reproducible tests, and any negative result documented honestly
rather than adjusted away. See each module's own docstring for its
specific validation numbers."""
from .bb84 import bb84_run, bb84_round, prepare_state, measure_in_basis, apply_depolarizing
from .di_qkd_ghz import (
    prepare_ghz3, measure_alice, measure_bob1, measure_other_bob,
    test_round, key_round, parity_chsh_win_rate, key_qber, expected_win_rate,
)
from .dicka_protocol2 import run_round, run_protocol, CLASSICAL_BOUND, QUANTUM_MAX

__all__ = [
    "bb84_run", "bb84_round", "prepare_state", "measure_in_basis", "apply_depolarizing",
    "prepare_ghz3", "measure_alice", "measure_bob1", "measure_other_bob",
    "test_round", "key_round", "parity_chsh_win_rate", "key_qber", "expected_win_rate",
    "run_round", "run_protocol", "CLASSICAL_BOUND", "QUANTUM_MAX",
]
