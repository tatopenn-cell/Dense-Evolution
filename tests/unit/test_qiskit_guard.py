"""Regression test for tests/_qiskit_guard.py (prog.txt test-suite audit,
issue #269 point 4): the shared macOS/qiskit skip condition must track
the real platform, and both call sites must import the same constants
rather than each redefining sys.platform == 'darwin' locally."""
import sys

from _qiskit_guard import QISKIT_UNSTABLE_ON_DARWIN, QISKIT_DARWIN_SKIP_REASON


def test_condition_matches_real_platform():
    assert QISKIT_UNSTABLE_ON_DARWIN == (sys.platform == 'darwin')


def test_skip_reason_mentions_macos_and_qiskit():
    assert 'qiskit' in QISKIT_DARWIN_SKIP_REASON.lower()
    assert 'macos' in QISKIT_DARWIN_SKIP_REASON.lower()
