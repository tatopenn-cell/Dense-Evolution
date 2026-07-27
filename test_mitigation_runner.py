"""
Tests for dashboard_core.mitigation_runner.run_mitigation_sweep and
dashboard_core.mitigation_panel.build_panel_mitigation -- Phase 3 of the
dashboard refactor (ZNE + predictive healing as a real dashboard feature,
reusing dense_evolution.mitigation.zero_noise_extrapolation as-is).
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

import dashboard_core as dc


BELL_CIRCUIT = "Bell |Φ+⟩"


def test_run_mitigation_sweep_ideal_raises():
    with pytest.raises(ValueError, match="ideal"):
        dc.run_mitigation_sweep(
            "Libreria Built-in", BELL_CIRCUIT, "", "ideal", 0.05, 256, 42,
        )


def test_run_mitigation_sweep_returns_expected_keys():
    res = dc.run_mitigation_sweep(
        "Libreria Built-in", BELL_CIRCUIT, "", "depolarizing", 0.05, 256, 42,
    )
    expected_keys = {
        "noise_factors", "prob_per_scale", "fidelity_per_scale",
        "prob_zne", "fidelity_zne", "prob_ideal", "healing_enabled",
        "delta_preemp", "n_qubits",
    }
    assert expected_keys == set(res.keys())
    assert res["noise_factors"] == (1.0, 2.0, 3.0)
    assert len(res["prob_per_scale"]) == 3
    assert len(res["fidelity_per_scale"]) == 3
    assert res["healing_enabled"] is False
    assert res["delta_preemp"] is None


def test_run_mitigation_sweep_prob_zne_normalized():
    res = dc.run_mitigation_sweep(
        "Libreria Built-in", BELL_CIRCUIT, "", "depolarizing", 0.08, 256, 7,
    )
    assert float(np.sum(res["prob_zne"])) == pytest.approx(1.0, abs=1e-6)
    assert np.all(res["prob_zne"] >= 0.0)


def test_run_mitigation_sweep_plain_vs_healing_differ():
    common = dict(source_mode="Libreria Built-in", circuit_name=BELL_CIRCUIT,
                  qasm_text="", noise_model="depolarizing", base_noise_p=0.08,
                  shots=256, seed=3)
    plain = dc.run_mitigation_sweep(**common, healing_enabled=False)
    healed = dc.run_mitigation_sweep(**common, healing_enabled=True)
    assert healed["healing_enabled"] is True
    assert healed["delta_preemp"] is not None
    # not asserting a specific direction -- just that the healing branch is
    # actually wired in and changes the result, not silently ignored
    assert plain["fidelity_zne"] != pytest.approx(healed["fidelity_zne"], abs=1e-12)


def test_run_mitigation_sweep_healing_requires_three_factors():
    with pytest.raises(ValueError, match="3"):
        dc.run_mitigation_sweep(
            "Libreria Built-in", BELL_CIRCUIT, "", "depolarizing", 0.05, 256, 42,
            noise_factors=(1.0, 2.0), healing_enabled=True,
        )


def test_run_mitigation_sweep_fidelity_zne_directionally_sane():
    # Error Mitigation (Real-Stress): a 15-qubit circuit with several
    # noisy CX/RZ layers -- fidelity degrades noticeably as noise scales up,
    # a case where ZNE has real signal to extrapolate against (unlike a
    # single-CX Bell state, where fidelity barely moves across 1x-3x).
    res = dc.run_mitigation_sweep(
        "Libreria Built-in", "Error Mitigation (Real-Stress)", "",
        "depolarizing", 0.06, 256, 11,
    )
    fid = res["fidelity_per_scale"]
    # not asserting strict per-step monotonicity: NoiseModel.apply_to_sv
    # draws its channel randomness from OS entropy for a JAX statevector
    # (see registry.py's rng/jax_key docs), so a single-shot fidelity
    # sample can plateau or tie between adjacent scales -- the precondition
    # that actually matters is that noise measurably degraded fidelity
    # overall (base vs. worst), not a smooth curve in between.
    assert fid[0] > fid[-1], "expected fidelity to be lower at the highest noise scale than at the base scale in this fixture"
    base_error = abs(1.0 - fid[0])
    zne_error = abs(1.0 - res["fidelity_zne"])
    assert zne_error < base_error, (
        f"ZNE should move the estimate closer to ideal (F=1) than the raw "
        f"base-noise sample: base_error={base_error:.4f}, zne_error={zne_error:.4f}"
    )


def test_build_panel_mitigation():
    mitigation_res = dc.run_mitigation_sweep(
        "Libreria Built-in", BELL_CIRCUIT, "", "depolarizing", 0.05, 256, 42,
    )
    base_res = dc.run_simulation(
        "Libreria Built-in", BELL_CIRCUIT, "", "depolarizing", 0.05, 256, 42,
    )
    fig = dc.build_panel_mitigation(mitigation_res, base_res)
    assert fig is not None
    assert type(fig).__name__ == "Figure"


def test_build_panel_mitigation_with_healing():
    mitigation_res = dc.run_mitigation_sweep(
        "Libreria Built-in", BELL_CIRCUIT, "", "depolarizing", 0.05, 256, 42,
        healing_enabled=True,
    )
    base_res = dc.run_simulation(
        "Libreria Built-in", BELL_CIRCUIT, "", "depolarizing", 0.05, 256, 42,
    )
    fig = dc.build_panel_mitigation(mitigation_res, base_res)
    assert fig is not None
