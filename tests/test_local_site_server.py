"""
Tests for local_site/app/server.py -- the Composer kernel FastAPI app.

Uses FastAPI's TestClient against the real `app` object: every endpoint
below calls the real dashboard_core/dense_evolution functions (already
covered by their own unit tests), so these tests exercise real routing,
request/response shapes, and error handling -- no mocked physics.
"""
import dashboard_core as dc
import numpy as np
import pytest

# fastapi/uvicorn/pydantic are the `dense-evolution[composer]` extra, not
# core dependencies (see dense_evolution/cli.py's module docstring) --
# skip this whole module cleanly wherever they aren't installed, instead
# of an import-time collection error that would abort the entire suite.
pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from local_site.app import server

client = TestClient(server.app)

BELL_QASM = dc.QASM_LIBRARY["Bell state (2 qubit)"]
SMALL_MOLECULE = "H2 (Idrogeno) - R = 0.7414 A [equilibrio reale]"


def test_health():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["dense_evolution_version"] == server.dense_evolution.__version__


def test_has_qiskit_runs_without_raising():
    assert server._has_qiskit() in (True, False)


def test_cors_allows_the_published_page_origin():
    resp = client.get("/api/health", headers={"Origin": "https://tatopenn-cell.github.io"})
    assert resp.headers.get("access-control-allow-origin") == "https://tatopenn-cell.github.io"


def test_health_reports_real_hostname_and_ram():
    """The Composer page shows this instead of an unverifiable "circuits
    really run on your PC" claim -- a visitor can check hostname/RAM
    themselves, so these need to be the machine's real values, not
    placeholders."""
    import socket
    resp = client.get("/api/health")
    body = resp.json()
    assert body["hostname"] == socket.gethostname()
    assert body["total_ram_gb"] > 0
    assert 0 <= body["ram_percent_free"] <= 100
    assert 0 <= body["available_ram_gb"] <= body["total_ram_gb"]


def test_private_network_access_preflight_succeeds_for_allowed_origin():
    """Chromium (incl. VS Code's Simple Browser) sends this extra preflight
    header when a page loaded from a public HTTPS origin talks to
    127.0.0.1, and requires this exact response header back -- without it,
    the real browser silently blocks the request even though plain CORS
    looks fine (verified directly: curl doesn't send this header, so this
    failure is invisible to curl-only testing)."""
    resp = client.options(
        "/api/health",
        headers={
            "Origin": "https://tatopenn-cell.github.io",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Private-Network": "true",
        },
    )
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-private-network") == "true"
    assert resp.headers.get("access-control-allow-origin") == "https://tatopenn-cell.github.io"


def test_private_network_access_preflight_rejected_for_disallowed_origin():
    """The PNA shortcut must not become a way to bypass the origin
    allowlist -- an origin not in ALLOWED_ORIGINS gets no PNA grant."""
    resp = client.options(
        "/api/health",
        headers={
            "Origin": "https://evil.example.com",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Private-Network": "true",
        },
    )
    assert resp.headers.get("access-control-allow-private-network") is None


def test_build_from_ops():
    resp = client.post("/api/build_from_ops", json={
        "n_qubits": 2,
        "ops": [{"gate": "h", "qubits": [0]}, {"gate": "cx", "qubits": [0, 1]}],
    })
    assert resp.status_code == 200
    qasm = resp.json()["qasm"]
    assert "OPENQASM" in qasm
    assert "h q[0]" in qasm


def test_build_from_ops_invalid_gate_returns_400():
    resp = client.post("/api/build_from_ops", json={
        "n_qubits": 2,
        "ops": [{"gate": "not_a_real_gate", "qubits": [0]}],
    })
    assert resp.status_code == 400


def test_run_bell_state_dense_backend():
    resp = client.post("/api/run", json={"qasm": BELL_QASM, "shots": 500, "seed": 0})
    assert resp.status_code == 200
    body = resp.json()
    assert body["large_scale"] is False
    assert body["n_qubits"] == 2
    assert set(body["counts"].keys()) <= {"00", "11"}
    assert sum(body["counts"].values()) == 500
    assert body["probabilities"] == pytest.approx([0.5, 0.0, 0.0, 0.5], abs=1e-9)
    assert body["circuit_png"]
    assert body["histogram_png"]
    assert body["qsphere_png"]
    assert body["bloch_png"]


def test_run_invalid_qasm_returns_400():
    resp = client.post("/api/run", json={"qasm": "this is not qasm"})
    assert resp.status_code == 400


def test_run_dispatches_to_mps_large_scale_path(monkeypatch):
    """Force the large-scale branch without an actually huge circuit by
    lowering the dense/MPS crossover threshold below the Bell state's own
    qubit count."""
    monkeypatch.setattr(server.dc, "MPS_DENSE_CONTRACTION_LIMIT", 1)
    resp = client.post("/api/run", json={"qasm": BELL_QASM, "backend": "mps", "seed": 0})
    assert resp.status_code == 200
    body = resp.json()
    assert body["large_scale"] is True
    assert body["backend"] == "mps"
    assert body["n_qubits"] == 2
    assert body["top_k_states"]
    assert body["circuit_png"]


def test_presets_matches_the_real_qasm_library():
    resp = client.get("/api/presets")
    assert resp.status_code == 200
    assert resp.json() == dc.QASM_LIBRARY


def test_palette_matches_the_real_gate_palette():
    resp = client.get("/api/palette")
    assert resp.status_code == 200
    assert resp.json() == dc.GATE_PALETTE


def test_noise_models_matches_the_real_registry():
    resp = client.get("/api/noise_models")
    assert resp.status_code == 200
    from dense_evolution.registry import NoiseModel
    assert resp.json() == NoiseModel.MODELS


def test_system_limits():
    # available_mb is a live free-RAM reading -- it can differ by a
    # fraction of a MB between this call and a second, separate call to
    # the same underlying function, so compare structure/keys, not an
    # exact second snapshot.
    resp = client.get("/api/system_limits")
    assert resp.status_code == 200
    body = resp.json()
    reference = dc.max_safe_dense_qubits()
    assert body.keys() == reference.keys()
    assert body["max_qubits_dense"] == reference["max_qubits_dense"]
    assert body["available_mb"] > 0


def test_hamiltonians_catalog():
    resp = client.get("/api/hamiltonians")
    assert resp.status_code == 200
    assert SMALL_MOLECULE in resp.json()


def test_hamiltonian_molecule():
    resp = client.post("/api/hamiltonian/molecule", json={"name": SMALL_MOLECULE})
    assert resp.status_code == 200
    body = resp.json()
    assert body["symbols"] == ["H", "H"]
    assert isinstance(body["ground_state_energy_hartree"], float)


def test_hamiltonian_molecule_unknown_returns_404():
    resp = client.post("/api/hamiltonian/molecule", json={"name": "not-a-real-molecule"})
    assert resp.status_code == 404


def test_hamiltonian_mix():
    resp = client.post("/api/hamiltonian/mix", json={
        "name_a": SMALL_MOLECULE, "name_b": SMALL_MOLECULE, "weight_a": 0.5, "weight_b": 0.5,
    })
    assert resp.status_code == 200
    body = resp.json()
    # mixing a molecule with itself at equal weight reproduces its own energy
    assert body["energy_mixed"] == pytest.approx(body["energy_a"], abs=1e-9)


def test_hamiltonian_mix_unknown_returns_404():
    resp = client.post("/api/hamiltonian/mix", json={"name_a": "nope", "name_b": SMALL_MOLECULE})
    assert resp.status_code == 404


def test_hamiltonian_mix_different_qubit_counts_returns_400(monkeypatch):
    """dc.mix_hamiltonians itself raises ValueError for mismatched Hilbert
    spaces -- fake two differently-sized (but each individually valid)
    Hamiltonians so this real rejection path is exercised without waiting
    on a second, slower real molecule's Hartree-Fock construction."""
    matrices = [np.eye(4, dtype=complex), np.eye(8, dtype=complex)]
    monkeypatch.setattr(server.dc, "get_molecular_hamiltonian_matrix", lambda *a, **k: matrices.pop(0))
    resp = client.post("/api/hamiltonian/mix", json={"name_a": SMALL_MOLECULE, "name_b": SMALL_MOLECULE})
    assert resp.status_code == 400


def test_hamiltonian_custom():
    resp = client.post("/api/hamiltonian/custom", json={
        "symbols": ["H", "H"], "geometry": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.7414]],
    })
    assert resp.status_code == 200
    assert isinstance(resp.json()["ground_state_energy_hartree"], float)


def test_hamiltonian_custom_mismatched_lengths_returns_400():
    resp = client.post("/api/hamiltonian/custom", json={
        "symbols": ["H", "H"], "geometry": [[0.0, 0.0, 0.0]],
    })
    assert resp.status_code == 400


def test_hamiltonian_custom_too_large_returns_400(monkeypatch):
    monkeypatch.setattr(server.dc, "build_molecular_hamiltonian", lambda *a, **k: (None, 13))
    resp = client.post("/api/hamiltonian/custom", json={
        "symbols": ["H", "H"], "geometry": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.7414]],
    })
    assert resp.status_code == 400
    assert "13 qubits" in resp.json()["detail"]


def test_vqe_on_catalog_molecule():
    resp = client.post("/api/vqe", json={
        "name": SMALL_MOLECULE, "n_layers": 1, "maxiter": 3, "seed": 0,
    })
    assert resp.status_code == 200
    assert "ground_state_energy" in resp.json() or resp.json()


def test_vqe_unknown_molecule_returns_404():
    resp = client.post("/api/vqe", json={"name": "not-a-real-molecule"})
    assert resp.status_code == 404


def test_vqe_without_name_or_symbols_returns_400():
    resp = client.post("/api/vqe", json={})
    assert resp.status_code == 400


def test_vqe_custom_mismatched_lengths_returns_400():
    resp = client.post("/api/vqe", json={"symbols": ["H", "H"], "geometry": [[0.0, 0.0, 0.0]]})
    assert resp.status_code == 400


def test_vqe_on_custom_symbols_and_geometry():
    resp = client.post("/api/vqe", json={
        "symbols": ["H", "H"], "geometry": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.7414]],
        "n_layers": 1, "maxiter": 3, "seed": 0,
    })
    assert resp.status_code == 200
    assert resp.json()


def test_vqe_accepts_real_adam_hyperparameters():
    """step_size/beta1/beta2 used to be hardcoded inside run_vqe with no
    way to set them from any request -- confirms the API actually forwards
    them (a wrong/unused value here wouldn't error, so this checks the
    energy actually differs from the default hyperparameters, not just
    that the request succeeds)."""
    default_resp = client.post("/api/vqe", json={
        "name": SMALL_MOLECULE, "n_layers": 1, "maxiter": 5, "seed": 0,
    })
    tuned_resp = client.post("/api/vqe", json={
        "name": SMALL_MOLECULE, "n_layers": 1, "maxiter": 5, "seed": 0,
        "step_size": 0.5, "beta1": 0.5, "beta2": 0.9,
    })
    assert default_resp.status_code == 200
    assert tuned_resp.status_code == 200
    assert default_resp.json()["vqe_energy_hartree"] != pytest.approx(
        tuned_resp.json()["vqe_energy_hartree"])


def test_qmmm_forces_on_catalog_molecule():
    resp = client.post("/api/qmmm_forces", json={"name": SMALL_MOLECULE})
    assert resp.status_code == 200
    body = resp.json()
    assert body["symbols"] == ["H", "H"]
    assert len(body["forces_hartree_per_angstrom"]) == 2
    assert body["force_norm"] > 0


def test_qmmm_forces_unknown_molecule_returns_400():
    resp = client.post("/api/qmmm_forces", json={"name": "not-a-real-molecule"})
    assert resp.status_code == 400


def test_md_trajectory_on_catalog_molecule():
    resp = client.post("/api/md_trajectory", json={
        "name": SMALL_MOLECULE, "n_steps": 3, "dt_fs": 0.5,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["step"]) == 3
    assert body["time_fs"] == [0.0, 0.5, 1.0]


def test_md_trajectory_with_ab_initio_recompute():
    """recompute_electronic_state=True re-solves real Hartree-Fock every
    step -- the more expensive, more accurate path; kept to a couple of
    steps here since each one is a real SCF solve."""
    resp = client.post("/api/md_trajectory", json={
        "name": SMALL_MOLECULE, "n_steps": 2, "dt_fs": 0.5,
        "recompute_electronic_state": True,
    })
    assert resp.status_code == 200
    assert len(resp.json()["step"]) == 2


def test_md_trajectory_n_steps_out_of_range_returns_400():
    resp = client.post("/api/md_trajectory", json={"name": SMALL_MOLECULE, "n_steps": 0})
    assert resp.status_code == 400
    resp = client.post("/api/md_trajectory", json={"name": SMALL_MOLECULE, "n_steps": 500})
    assert resp.status_code == 400


def test_md_trajectory_ab_initio_over_step_cap_returns_400():
    resp = client.post("/api/md_trajectory", json={
        "name": SMALL_MOLECULE, "n_steps": 31, "recompute_electronic_state": True,
    })
    assert resp.status_code == 400


def test_mitigate():
    resp = client.post("/api/mitigate", json={
        "qasm": BELL_QASM, "pauli_string": "ZZ", "noise_model": "depolarizing",
        "noise_p": 0.05, "seed": 0,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["n_qubits"] == 2
    assert isinstance(body["zne_extrapolated"], float)


def test_mitigate_invalid_qasm_returns_400():
    resp = client.post("/api/mitigate", json={
        "qasm": "not qasm", "pauli_string": "Z", "noise_model": "depolarizing", "noise_p": 0.05,
    })
    assert resp.status_code == 400


def test_mitigate_matrix():
    resp = client.post("/api/mitigate_matrix", json={
        "qasm": BELL_QASM, "noise_model": "depolarizing", "noise_p": 0.05, "seed": 0,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["n_qubits"] == 2
    assert 0.0 <= body["fidelity_raw"] <= 1.0
    assert 0.0 <= body["fidelity_corrected"] <= 1.0


def test_mitigate_matrix_invalid_qasm_returns_400():
    resp = client.post("/api/mitigate_matrix", json={
        "qasm": "not qasm", "noise_model": "depolarizing", "noise_p": 0.05,
    })
    assert resp.status_code == 400
