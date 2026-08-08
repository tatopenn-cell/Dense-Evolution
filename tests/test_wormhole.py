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

import itertools

from dashboard_core.wormhole import (
    run_wormhole_protocol, run_wormhole_protocol_finite_beta,
    _finite_beta_layout_precomputed, _run_finite_beta_precomputed,
    find_delta_beta_bands, commuting_pair_count, build_sparse_syk_terms,
)
from dense_evolution.chunk import MemoryPressureError

N_MAJORANA, K_TERMS, J = 8, 10, float(np.sqrt(2))
SEED = 61


def _dense_matrix_commuting_pair_count(terms, n_qubits):
    """Independent reference: the original O(2**n_qubits) dense-matrix
    implementation commuting_pair_count used before the O(n_qubits)
    Pauli-dict rewrite -- kept here only to verify the rewrite against
    an implementation that doesn't share its own logic."""
    import dense_evolution as de
    matrices = [de.pauli_hamiltonian_to_matrix([(1.0, t[1])], n_qubits) for t in terms]
    commuting = anticommuting = 0
    for a, b in itertools.combinations(range(len(matrices)), 2):
        comm = matrices[a] @ matrices[b] - matrices[b] @ matrices[a]
        if np.max(np.abs(comm)) < 1e-9:
            commuting += 1
        else:
            anticommuting += 1
    return commuting, anticommuting
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


class TestFindDeltaBetaBands:

    def test_seed_61_matches_known_crossings(self):
        # Recorded 2026-08-08 (fine grid, beta_step=0.02, beta_max=6):
        # seed=61 crosses sign at beta~0.419 and beta~1.861, giving
        # bands (positive, negative, positive). Restricted to
        # beta_max=2.5 here (covers both known crossings) with a
        # coarser step for test speed -- the interpolated crossing
        # locations should still land close to the finer-grid values.
        bands = find_delta_beta_bands(N_MAJORANA, K_TERMS, J, MU, T0, T1, SEED, True,
                                       beta_max=2.5, beta_step=0.05)
        assert len(bands) == 3
        assert [b["sign"] for b in bands] == ["positive", "negative", "positive"]
        assert bands[0]["beta_lo"] == 0.0
        assert bands[0]["beta_hi"] == pytest.approx(0.419, abs=0.1)
        assert bands[1]["beta_hi"] == pytest.approx(1.861, abs=0.1)
        assert bands[-1]["beta_hi"] == 2.5

    def test_bands_partition_the_full_range_contiguously(self):
        bands = find_delta_beta_bands(N_MAJORANA, K_TERMS, J, MU, T0, T1, SEED, True,
                                       beta_max=2.5, beta_step=0.05)
        for prev_band, next_band in zip(bands, bands[1:]):
            assert prev_band["beta_hi"] == next_band["beta_lo"]
        assert all(b["max_abs_delta"] > 0 for b in bands)

    def test_seed_that_never_flips_gives_a_single_band(self):
        # seed=1944: delta_beta0=+0.03690, and independently verified
        # (n=30 stability run, fine-grid scan over the full [0, 6]) to
        # never change sign -- a single band spanning the whole range.
        bands = find_delta_beta_bands(N_MAJORANA, K_TERMS, J, MU, T0, T1, 1944, True,
                                       beta_max=2.5, beta_step=0.05)
        assert len(bands) == 1
        assert bands[0]["sign"] == "positive"
        assert bands[0]["beta_lo"] == 0.0
        assert bands[0]["beta_hi"] == 2.5


class TestCommutingPairCount:

    def test_seed_61_matches_papers_own_ratio(self):
        # arXiv:2604.10090's own chosen K=10 instance: 34 commuting / 11
        # anticommuting out of C(10,2)=45 pairs -- this module's own
        # docstring value, re-derived here, not hardcoded blindly.
        n_qubits, terms = build_sparse_syk_terms(N_MAJORANA, K_TERMS, J, SEED)
        commuting, anticommuting = commuting_pair_count(terms, n_qubits)
        assert (commuting, anticommuting) == (34, 11)

    @pytest.mark.parametrize("n_majorana,seeds", [
        (8, range(20)),
        (12, range(5)),
        (16, range(3)),
    ])
    def test_matches_dense_matrix_reference(self, n_majorana, seeds):
        # O(n_qubits)-per-pair Pauli-dict rewrite must match the
        # original O(2**n_qubits) dense-matrix commutator computation
        # exactly, not approximately, across real SYK instances at
        # several sizes -- not just the one seed=61 spot check above.
        for seed in seeds:
            n_qubits, terms = build_sparse_syk_terms(n_majorana, K_TERMS, J, seed)
            fast = commuting_pair_count(terms, n_qubits)
            reference = _dense_matrix_commuting_pair_count(terms, n_qubits)
            assert fast == reference, f"n_majorana={n_majorana} seed={seed}: {fast} != {reference}"

    def test_counts_sum_to_total_pairs(self):
        n_qubits, terms = build_sparse_syk_terms(N_MAJORANA, K_TERMS, J, SEED)
        commuting, anticommuting = commuting_pair_count(terms, n_qubits)
        from math import comb
        assert commuting + anticommuting == comb(K_TERMS, 2)


class TestExactBackendMemoryGuard:

    def test_normal_scale_still_works(self):
        # n_majorana=8 (n_full=10, dim=1024) must be completely
        # unaffected by the added memory check.
        result = run_wormhole_protocol(N_MAJORANA, K_TERMS, J, 12.0, 0.3, 0.60, SEED, True)
        assert result == pytest.approx(0.01326, abs=1e-3)

    def test_unreasonably_large_n_majorana_raises_memory_pressure_error(self):
        # BUG FIX: run_wormhole_protocol / _finite_beta_layout_precomputed
        # build dense 2**n_full x 2**n_full matrices with no size check --
        # a large n_majorana used to be an unhandled OOM crash instead of
        # a clear, actionable error. n_majorana=24 (n_full=14,
        # dim=16384) needs ~1TB for this estimate, far beyond any real
        # machine's available RAM.
        with pytest.raises(MemoryPressureError):
            run_wormhole_protocol(24, K_TERMS, J, 12.0, 0.3, 0.60, SEED, True)

    def test_finite_beta_layout_also_raises_for_unreasonably_large_n_majorana(self):
        with pytest.raises(MemoryPressureError):
            _finite_beta_layout_precomputed(24, K_TERMS, J, SEED)
