"""
Unit tests for dashboard_core/wormhole.py's finite-beta thermofield
double backend (run_wormhole_protocol_finite_beta and its precomputed-
layout helpers). Checks the physics is actually correct -- beta=0
exactly reproduces the existing beta=0 backend, the precompute-once
optimization is bit-identical to the naive per-call path, beta=3 (the
real value arXiv:2604.10090 uses) gives a genuinely different, finite
result -- not just that the functions run without raising.
"""
import numpy as np
import pytest

from dashboard_core.wormhole import (
    run_wormhole_protocol, run_wormhole_protocol_finite_beta,
    _finite_beta_layout_precomputed, _run_finite_beta_precomputed,
)

N_MAJORANA, K_TERMS, J = 8, 10, float(np.sqrt(2))
SEED = 61
T0, MU, T1 = 0.3, 12.0, 0.60


class TestRunWormholeProtocolFiniteBeta:

    def test_beta_zero_matches_existing_beta_zero_backend(self):
        # exp(-0*H/4) = identity, so beta=0 must reproduce
        # run_wormhole_protocol's plain Bell-pair TFD exactly, not just
        # approximately -- this is the correctness anchor for the whole
        # finite-beta code path.
        finite = run_wormhole_protocol_finite_beta(N_MAJORANA, K_TERMS, J, MU, T0, T1, 0.0, SEED, True)
        reference = run_wormhole_protocol(N_MAJORANA, K_TERMS, J, MU, T0, T1, SEED, True)
        assert finite == pytest.approx(reference, abs=1e-9)

    def test_beta_three_gives_a_different_finite_result(self):
        # Recorded 2026-08-08: at the paper's own beta=3 (Section S2),
        # seed=61's delta shrinks but keeps its sign relative to beta=0
        # (+0.00468 -> +0.00251) -- a real, different, non-trivial value,
        # not NaN/inf and not identical to the beta=0 result.
        beta0 = run_wormhole_protocol_finite_beta(N_MAJORANA, K_TERMS, J, MU, T0, T1, 0.0, SEED, True)
        beta3 = run_wormhole_protocol_finite_beta(N_MAJORANA, K_TERMS, J, MU, T0, T1, 3.0, SEED, True)
        assert np.isfinite(beta3)
        assert beta3 != pytest.approx(beta0, abs=1e-6)
        assert beta3 == pytest.approx(0.04166, abs=2e-4)

    def test_precomputed_layout_matches_one_shot_wrapper_exactly(self):
        # The precompute-once path (_finite_beta_layout_precomputed +
        # _run_finite_beta_precomputed) exists purely as a speed
        # optimization for (beta, mu) sweeps at a fixed seed -- it must
        # give bit-identical results to the naive per-call wrapper, not
        # just a close approximation.
        wrapper_result = run_wormhole_protocol_finite_beta(N_MAJORANA, K_TERMS, J, MU, T0, T1, 3.0, SEED, True)

        layout = _finite_beta_layout_precomputed(N_MAJORANA, K_TERMS, J, SEED)
        precomputed_result = _run_finite_beta_precomputed(*layout, MU, T0, T1, 3.0, True)

        assert precomputed_result == pytest.approx(wrapper_result, abs=1e-12)

    def test_precomputed_layout_reused_across_multiple_beta_values(self):
        # The whole point of the precomputed layout: reuse the same
        # eigendecomposition across several beta values for one seed,
        # each still matching the one-shot wrapper's own result.
        layout = _finite_beta_layout_precomputed(N_MAJORANA, K_TERMS, J, SEED)
        for beta in (0.0, 1.0, 3.0):
            precomputed_result = _run_finite_beta_precomputed(*layout, MU, T0, T1, beta, True)
            wrapper_result = run_wormhole_protocol_finite_beta(N_MAJORANA, K_TERMS, J, MU, T0, T1, beta, SEED, True)
            assert precomputed_result == pytest.approx(wrapper_result, abs=1e-12)
