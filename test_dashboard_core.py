"""
Smoke tests for dashboard_core.py — the compute/panel-builder layer behind
app_dashboard.py's Quantum Simulator tab. Mirrors the checks performed
manually while building the module: correct shapes/keys, no exceptions,
every panel builder returns a usable figure (including the empty-data
placeholder paths).
"""

import json

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

import dashboard_core as dc


BELL_CIRCUIT = "Bell |Φ+⟩"          # no parametric gates -> mock VQE path
VQE_CIRCUIT = "VQE ansatz H₂"       # has parametric gates -> real VQE path


@pytest.fixture(scope="module")
def bell_res():
    return dc.run_simulation("Libreria Built-in", BELL_CIRCUIT, "", "ideal", 0.0, 128, 42, use_float32=True)


@pytest.fixture(scope="module")
def noisy_res():
    return dc.run_simulation("Libreria Built-in", VQE_CIRCUIT, "", "depolarizing", 0.03, 128, 42, use_float32=True)


@pytest.fixture(scope="module")
def df_vqe_mock(bell_res):
    return dc.run_vqe_telemetry(
        bell_res["sim"], bell_res["parser"], dc.QASM_LIBRARY[BELL_CIRCUIT],
        BELL_CIRCUIT, bell_res["n_qubits"], True, epochs=6, lr=0.05, beta1=0.9, beta2=0.999, seed=42,
    )


@pytest.fixture(scope="module")
def df_vqe_real(noisy_res):
    return dc.run_vqe_telemetry(
        noisy_res["sim"], noisy_res["parser"], dc.QASM_LIBRARY[VQE_CIRCUIT],
        VQE_CIRCUIT, noisy_res["n_qubits"], True, epochs=4, lr=0.05, beta1=0.9, beta2=0.999, seed=42,
    )


@pytest.fixture(scope="module")
def md_telemetry():
    return dc.run_md_telemetry(md_steps=20, md_temp=300)


# ── run_simulation ──────────────────────────────────────────────────────

def test_run_simulation_ideal_keys(bell_res):
    expected_keys = {
        "prob", "prob_ideal", "noise_factor", "fidelity", "n_qubits", "entropy",
        "idx_max", "stato_dominante", "tempo", "ram", "nome", "porte_count",
        "shots_data", "sim", "parser",
    }
    assert expected_keys <= set(bell_res.keys())
    assert bell_res["n_qubits"] == 2
    assert not np.isnan(bell_res["prob"]).any()
    assert np.isclose(bell_res["prob"].sum(), 1.0, atol=1e-4)


def test_run_simulation_with_noise(noisy_res):
    assert not np.isnan(noisy_res["prob"]).any()
    assert 0.0 <= noisy_res["fidelity"] <= 1.0 + 1e-6


def test_run_simulation_heavy_circuit_15q():
    res = dc.run_simulation("Libreria Built-in", "Error Mitigation (Real-Stress)", "", "bitflip", 0.05, 64, 42, use_float32=True)
    assert res["n_qubits"] == 15


# ── run_vqe_telemetry ───────────────────────────────────────────────────

def test_vqe_telemetry_mock_path(df_vqe_mock):
    assert list(df_vqe_mock.columns) == ["VQE_Energy", "Entropy", "Purity", "Gradient", "Noise_Factor", "Theta_Correction"]
    assert len(df_vqe_mock) == 6


def test_vqe_telemetry_real_path(df_vqe_real):
    assert list(df_vqe_real.columns) == ["VQE_Energy", "Entropy", "Purity", "Gradient", "Noise_Factor", "Theta_Correction"]
    assert len(df_vqe_real) == 4


def test_vqe_telemetry_heavy_qubit_guard_is_caller_responsibility():
    # dashboard_core itself doesn't refuse heavy circuits (the UI layer
    # gates this via QM_MM_HEAVY_QUBIT_THRESHOLD) — just confirm the
    # threshold constant it's built around is exported and sane.
    assert dc.QM_MM_HEAVY_QUBIT_THRESHOLD == 12


# ── run_md_telemetry ────────────────────────────────────────────────────

def test_md_telemetry_shape(md_telemetry):
    df_md, corr_matrix = md_telemetry
    assert len(df_md) == 20
    assert "Energia_VQE_Ha" in df_md.columns
    assert corr_matrix.shape == (len(df_md.columns), len(df_md.columns))


# ── compute_overview_metrics ────────────────────────────────────────────

def test_compute_overview_metrics_keys(bell_res):
    metrics = dc.compute_overview_metrics(bell_res, "ideal", 0.0)
    expected_labels = {
        "Qubits", "Hilbert Dim", "Gates", "Entropy", "Concurrence", "Purity",
        "Spectral σ", "Top State", "P(top)", "RAM", "Time", "Noise", "Noise p",
    }
    assert {m["label"] for m in metrics} == expected_labels
    assert all(isinstance(m["value"], str) for m in metrics)
    # short values must never overflow a narrow st.metric tile — cap at a sane length
    assert all(len(m["value"]) <= 20 for m in metrics)


def test_compute_overview_metrics_short_values_even_for_heavy_circuit():
    # 15-qubit dominant-state bitstring used to blow past any reasonable
    # tile width ("|000000000000000⟩") — must now stay short via the #idx form
    res = dc.run_simulation("Libreria Built-in", "Error Mitigation (Real-Stress)", "", "ideal", 0.0, 64, 42, use_float32=True)
    metrics = dc.compute_overview_metrics(res, "ideal", 0.0)
    top_state = next(m for m in metrics if m["label"] == "Top State")
    assert len(top_state["value"]) <= 10
    assert "⟩" in top_state["help"]


# ── panel builders ──────────────────────────────────────────────────────

def test_build_panel_overview(bell_res, df_vqe_mock, md_telemetry):
    df_md, corr_matrix = md_telemetry
    fig = dc.build_panel_overview(bell_res, df_vqe_mock, corr_matrix, "ideal", 0.0)
    assert fig is not None


def test_build_panel_fisica(bell_res):
    fig = dc.build_panel_fisica(bell_res, seed=42)
    assert fig is not None


def test_build_panel_mosaico(bell_res):
    fig = dc.build_panel_mosaico(bell_res)
    assert fig is not None


def test_build_panel_vqe_results_with_data(df_vqe_mock):
    fig = dc.build_panel_vqe_results(df_vqe_mock)
    assert fig is not None


def test_build_panel_vqe_results_empty():
    fig = dc.build_panel_vqe_results(pd.DataFrame())
    assert fig is not None


def test_build_panel_md_results_with_data(md_telemetry):
    df_md, corr_matrix = md_telemetry
    fig = dc.build_panel_md_results(df_md, corr_matrix)
    assert fig is not None


def test_build_panel_md_results_empty():
    fig = dc.build_panel_md_results(pd.DataFrame(), pd.DataFrame())
    assert fig is not None


def test_build_panel_performance_with_history(bell_res):
    history = [{"nome": bell_res["nome"], "n_qubits": bell_res["n_qubits"],
                "tempo": bell_res["tempo"], "ram": bell_res["ram"]}]
    fig = dc.build_panel_performance(bell_res, history)
    assert fig is not None


def test_build_panel_performance_empty_history(bell_res):
    fig = dc.build_panel_performance(bell_res, [])
    assert fig is not None


def test_build_3d_helix_patch(bell_res):
    fig = dc.build_3d_helix_patch(bell_res["n_qubits"], bell_res["prob"])
    assert fig is not None


# ── benchmark + provenance export ───────────────────────────────────────

def test_run_benchmark_scan_small_range():
    df = dc.run_benchmark_scan(range(2, 6, 2))
    assert list(df.columns) == ["Qubits", "Hilbert_Dim", "Time_s", "RAM_Sim_MB", "Delta_RAM_RSS_MB"]
    assert len(df) == 2


def test_build_provenance_json_roundtrip(bell_res):
    history = [{"nome": bell_res["nome"], "n_qubits": bell_res["n_qubits"],
                "tempo": bell_res["tempo"], "ram": bell_res["ram"],
                "prob_sample": bell_res["prob"]}]
    payload = dc.build_provenance_json(history)
    parsed = json.loads(payload)
    integrity_hash = parsed["metadata"]["integrity_sha256"]
    assert len(integrity_hash) == 64 and all(c in "0123456789abcdef" for c in integrity_hash)
    assert len(parsed["records"]) == 1
    assert isinstance(parsed["records"][0]["prob_sample"], list)


# ── Hamiltonian library ─────────────────────────────────────────────────

def test_infer_qubit_count_from_qasm():
    assert dc.infer_qubit_count_from_qasm(dc.QASM_LIBRARY[VQE_CIRCUIT]) == 2
    assert dc.infer_qubit_count_from_qasm(dc.QASM_LIBRARY["Error Mitigation (Real-Stress)"]) == 15
    assert dc.infer_qubit_count_from_qasm("") is None
    assert dc.infer_qubit_count_from_qasm("garbage, no qreg here") is None


def test_get_compatible_hamiltonians():
    compat_2q = dc.get_compatible_hamiltonians(2)
    assert len(compat_2q) > 0
    assert all(len(v) == 4 for v in compat_2q.values())
    # the None-valued "Spettro Uniforme Classico" entry must never appear (matches dash.py's own filter)
    assert all(v is not None for v in compat_2q.values())
    assert dc.get_compatible_hamiltonians(0) == {}
    assert dc.get_compatible_hamiltonians(None) == {}


def test_save_custom_hamiltonian_valid_and_duplicate():
    library = dict(dc.LIBRERIA_HAMILTONIANE)
    ok, _ = dc.save_custom_hamiltonian(library, "Test H", "[1.0, -1.0, 0.5, -0.5]")
    assert ok and "Test H" in library
    ok2, msg2 = dc.save_custom_hamiltonian(library, "Test H", "[1.0, -1.0, 0.5, -0.5]")
    assert not ok2 and "esiste già" in msg2


def test_save_custom_hamiltonian_invalid_inputs():
    library = dict(dc.LIBRERIA_HAMILTONIANE)
    ok, _ = dc.save_custom_hamiltonian(library, "", "[1.0]")
    assert not ok
    ok, _ = dc.save_custom_hamiltonian(library, "Bad JSON", "not json")
    assert not ok
    ok, _ = dc.save_custom_hamiltonian(library, "Not a list", '{"a": 1}')
    assert not ok


def test_build_panel_hamiltonian_with_and_without_data():
    fig = dc.build_panel_hamiltonian([-1.13, -0.45, 0.12, 0.64], "H2 test")
    assert fig is not None
    fig_empty = dc.build_panel_hamiltonian(None, "none")
    assert fig_empty is not None


def test_vqe_telemetry_with_custom_hamiltonian(noisy_res):
    compat = dc.get_compatible_hamiltonians(noisy_res["n_qubits"])
    values = next(iter(compat.values()))
    df = dc.run_vqe_telemetry(
        noisy_res["sim"], noisy_res["parser"], dc.QASM_LIBRARY[VQE_CIRCUIT],
        VQE_CIRCUIT, noisy_res["n_qubits"], True, epochs=3, lr=0.05, beta1=0.9, beta2=0.999, seed=42,
        hamiltonian_values=values,
    )
    assert len(df) == 3


# ── on_epoch callback hook (purely additive) ────────────────────────────

def test_on_epoch_callback_mock_path(bell_res):
    calls = []
    df = dc.run_vqe_telemetry(
        bell_res["sim"], bell_res["parser"], dc.QASM_LIBRARY[BELL_CIRCUIT],
        BELL_CIRCUIT, bell_res["n_qubits"], True, epochs=5, lr=0.05, beta1=0.9, beta2=0.999, seed=42,
        on_epoch=lambda e, t, row: calls.append((e, t, row)),
    )
    assert len(calls) == 5
    assert calls[0][0] == 0 and calls[0][1] == 5
    assert calls[-1][0] == 4
    assert set(calls[0][2].keys()) >= {"VQE_Energy", "Entropy", "Purity", "Gradient", "Noise_Factor", "Theta_Correction"}
    assert len(df) == 5


def test_on_epoch_callback_real_path(noisy_res):
    calls = []
    df = dc.run_vqe_telemetry(
        noisy_res["sim"], noisy_res["parser"], dc.QASM_LIBRARY[VQE_CIRCUIT],
        VQE_CIRCUIT, noisy_res["n_qubits"], True, epochs=4, lr=0.05, beta1=0.9, beta2=0.999, seed=42,
        on_epoch=lambda e, t, row: calls.append((e, t, row)),
    )
    assert len(calls) == 4
    assert calls[0][1] == 4
    assert len(df) == 4


def test_run_vqe_telemetry_without_on_epoch_still_works(bell_res):
    # regression guard: on_epoch defaults to None, existing callers unaffected
    df = dc.run_vqe_telemetry(
        bell_res["sim"], bell_res["parser"], dc.QASM_LIBRARY[BELL_CIRCUIT],
        BELL_CIRCUIT, bell_res["n_qubits"], True, epochs=3, lr=0.05, beta1=0.9, beta2=0.999, seed=42,
    )
    assert len(df) == 3
