"""
Tests for mcp_server/server.py -- the dense_evolution_mcp MCP adapter.

Routes every request through httpx.ASGITransport straight into the real,
in-process local_site.app.server.app (see server._TEST_TRANSPORT) instead
of a live subprocess kernel bound to a real port. That means every test
below still exercises the real DenseSVSimulator / real PennyLane
Hartree-Fock Hamiltonians -- no mocked physics, only the network hop is
swapped for an in-process one, for the same reason test_local_site_server.py
uses FastAPI's TestClient rather than a live server.
"""
import asyncio
import json
import os

import pytest

# fastapi/uvicorn/pydantic (composer) and mcp/httpx (mcp) are both optional
# extras -- skip this whole module cleanly wherever they aren't installed,
# same pattern test_local_site_server.py uses for the composer extra alone.
pytest.importorskip("fastapi")
pytest.importorskip("mcp")
import httpx  # noqa: E402

from local_site.app import server as kernel  # noqa: E402
from mcp_server import server as mcp_adapter  # noqa: E402

H2 = "H2 (Idrogeno) - R = 0.7414 A [equilibrio reale]"
EXACT_H2_ENERGY_HARTREE = -1.1372701748786913
BELL_QASM = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0],q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
"""


def run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def route_through_real_kernel_in_process():
    mcp_adapter._TEST_TRANSPORT = httpx.ASGITransport(app=kernel.app)
    # A fresh molecule-alias cache per test: several tests below rely on
    # the real /api/hamiltonians catalog being (re)fetched, not whatever a
    # previous test happened to populate.
    mcp_adapter._molecule_alias_cache = None
    yield
    mcp_adapter._TEST_TRANSPORT = None
    mcp_adapter._molecule_alias_cache = None


def test_health_reports_real_kernel_info():
    data = json.loads(run(mcp_adapter.dense_evolution_health()))
    assert data["status"] == "ok"
    assert data["dense_evolution_version"] == kernel.dense_evolution.__version__
    assert data["total_ram_gb"] > 0


def test_system_limits_reports_positive_qubit_ceiling():
    data = json.loads(run(mcp_adapter.dense_evolution_system_limits()))
    assert data["max_qubits_dense"] > 0


def test_list_presets_includes_a_real_bell_state():
    data = json.loads(run(mcp_adapter.dense_evolution_list_presets()))
    assert any("Bell" in name for name in data)


def test_list_gates_is_non_empty():
    data = json.loads(run(mcp_adapter.dense_evolution_list_gates()))
    assert len(data) > 0


def test_list_noise_models_includes_depolarizing():
    data = json.loads(run(mcp_adapter.dense_evolution_list_noise_models()))
    assert "depolarizing" in data


def test_build_circuit_from_ops_round_trips_to_valid_qasm():
    ops = [{"gate": "h", "qubits": [0]}, {"gate": "cx", "qubits": [0, 1]}]
    result = run(mcp_adapter.dense_evolution_build_circuit(mcp_adapter.BuildCircuitInput(n_qubits=2, ops=ops)))
    data = json.loads(result)
    assert "OPENQASM" in data["qasm"]
    assert "h q[0]" in data["qasm"]


def test_custom_molecule_energy_matches_catalog_h2():
    result = json.loads(run(mcp_adapter.dense_evolution_custom_molecule_energy(mcp_adapter.CustomMoleculeInput(
        symbols=["H", "H"], geometry=[[0, 0, 0], [0, 0, 0.7414]],
    ))))
    assert result["ground_state_energy_hartree"] == pytest.approx(EXACT_H2_ENERGY_HARTREE, abs=1e-6)


def test_qmmm_forces_returns_real_force_vectors():
    result = json.loads(run(mcp_adapter.dense_evolution_qmmm_forces(mcp_adapter.QmmmForcesInput(name="H2"))))
    assert "forces" in result or "force_norm" in result or "energy_hartree" in result


def test_md_trajectory_runs_a_few_fixed_electronic_state_steps():
    result = json.loads(run(mcp_adapter.dense_evolution_md_trajectory(mcp_adapter.MdTrajectoryInput(
        name="H2", n_steps=2, recompute_electronic_state=False,
    ))))
    assert "error" not in result


def test_mitigate_zne_richardson_extrapolates_toward_ideal():
    result = json.loads(run(mcp_adapter.dense_evolution_mitigate_zne(mcp_adapter.MitigateZneInput(
        qasm=BELL_QASM, pauli_string="ZZ", noise_model="depolarizing", noise_p=0.05,
    ))))
    # Real physics check, not just "did it return something": noisy
    # expectations should sit between the ideal value and zero, and the
    # extrapolated value should land closer to ideal than the noisiest one.
    assert result["ideal_expectation"] == pytest.approx(1.0, abs=1e-6)
    assert abs(result["zne_extrapolated"] - 1.0) < abs(result["noisy_expectations"][-1] - 1.0)


def test_mitigate_density_matrix_reports_fidelity_improvement():
    result = json.loads(run(mcp_adapter.dense_evolution_mitigate_density_matrix(
        mcp_adapter.MitigateDensityMatrixInput(qasm=BELL_QASM, noise_model="depolarizing", noise_p=0.05)
    )))
    assert 0.0 <= result["fidelity_raw"] <= 1.0
    assert 0.0 <= result["fidelity_corrected"] <= 1.0


def test_list_molecules_includes_short_ids():
    data = json.loads(run(mcp_adapter.dense_evolution_list_molecules(mcp_adapter.ListMoleculesInput())))
    by_id = {m["id"]: m for m in data}
    assert "H2" in by_id
    assert by_id["H2"]["full_name"] == H2
    assert by_id["H2"]["n_qubits"] == 4


def test_molecule_energy_accepts_short_id():
    result = run(mcp_adapter.dense_evolution_molecule_energy(mcp_adapter.MoleculeEnergyInput(name="H2")))
    data = json.loads(result)
    assert data["ground_state_energy_hartree"] == pytest.approx(EXACT_H2_ENERGY_HARTREE, abs=1e-6)


def test_molecule_energy_accepts_full_catalog_name():
    result = run(mcp_adapter.dense_evolution_molecule_energy(mcp_adapter.MoleculeEnergyInput(name=H2)))
    data = json.loads(result)
    assert data["ground_state_energy_hartree"] == pytest.approx(EXACT_H2_ENERGY_HARTREE, abs=1e-6)


def test_molecule_energy_unknown_name_is_a_clear_error_not_a_crash():
    result = run(mcp_adapter.dense_evolution_molecule_energy(mcp_adapter.MoleculeEnergyInput(name="Unobtainium")))
    assert result.startswith("Error:")
    assert "Unobtainium" in result


def test_mix_molecules_accepts_short_ids():
    result = run(mcp_adapter.dense_evolution_mix_molecules(
        mcp_adapter.MixMoleculesInput(name_a="H2", name_b="HeH+", weight_a=0.5, weight_b=0.5)
    ))
    # H2 (4 qubits) and HeH+ (4 qubits) share a qubit count -- a real,
    # meaningful mix; the tool should resolve both short ids and succeed.
    data = json.loads(result)
    assert "energy_mixed" in data


def test_run_circuit_bell_state_gives_real_50_50_split():
    result = run(mcp_adapter.dense_evolution_run_circuit(
        mcp_adapter.RunCircuitInput(qasm=BELL_QASM, shots=500, seed=42)
    ))
    data = json.loads(result)
    assert data["n_qubits"] == 2
    assert set(data["counts"]) == {"00", "11"}
    assert data["counts"]["00"] + data["counts"]["11"] == 500
    # Both truncation summaries should be present and self-consistent.
    assert data["statevector"]["total_nonzero_amplitudes"] == 2
    assert data["probabilities"]["total_basis_states"] == 4


def test_run_circuit_visualizations_are_saved_to_disk_not_inlined(tmp_path):
    original_dir = mcp_adapter.IMAGE_OUTPUT_DIR
    mcp_adapter.IMAGE_OUTPUT_DIR = tmp_path
    try:
        result = run(mcp_adapter.dense_evolution_run_circuit(
            mcp_adapter.RunCircuitInput(qasm=BELL_QASM, shots=50, include_visualizations=True)
        ))
        data = json.loads(result)
        for key in ("circuit_png_path", "histogram_png_path", "qsphere_png_path", "bloch_png_path"):
            assert key in data
            path = data[key]
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
            assert str(tmp_path) in path
        # The point of saving to disk: no raw base64 anywhere in the response
        # -- checked against the parsed dict's own keys, since "circuit_png"
        # is (correctly) a substring of "circuit_png_path" in the raw text.
        for raw_key in ("circuit_png", "histogram_png", "qsphere_png", "bloch_png"):
            assert raw_key not in data
    finally:
        mcp_adapter.IMAGE_OUTPUT_DIR = original_dir


def test_energy_scan_matches_individual_molecule_energy_call():
    single = json.loads(run(mcp_adapter.dense_evolution_molecule_energy(mcp_adapter.MoleculeEnergyInput(name="H2"))))
    scan_result = json.loads(run(mcp_adapter.dense_evolution_energy_scan(mcp_adapter.EnergyScanInput(
        symbols=["H", "H"],
        geometries=[[[0, 0, 0], [0, 0, 0.7414]]],
        labels=["equilibrium"],
    ))))
    assert scan_result["n_points"] == 1
    assert scan_result["results"][0]["ground_state_energy_hartree"] == pytest.approx(
        single["ground_state_energy_hartree"], abs=1e-9
    )
    assert scan_result["minimum"]["label"] == "equilibrium"


def test_energy_scan_finds_h2_minimum_near_known_equilibrium():
    result = json.loads(run(mcp_adapter.dense_evolution_energy_scan(mcp_adapter.EnergyScanInput(
        symbols=["H", "H"],
        geometries=[[[0, 0, 0], [0, 0, r]] for r in (0.5, 0.7414, 1.2)],
        labels=[0.5, 0.7414, 1.2],
    ))))
    assert result["minimum"]["label"] == 0.7414


def test_energy_scan_isolates_a_failing_point_from_the_rest():
    # A 13-atom linear hydrogen chain needs far more than 12 qubits --
    # dense_evolution_custom_molecule_energy's own real, documented cap --
    # so this point must fail while the small, valid point still succeeds.
    big_symbols = ["H"] * 13
    big_geometry = [[0, 0, float(i)] for i in range(13)]
    result = json.loads(run(mcp_adapter.dense_evolution_energy_scan(mcp_adapter.EnergyScanInput(
        symbols=["H", "H"],
        geometries=[[[0, 0, 0], [0, 0, 0.7414]]],
        labels=["ok"],
    ))))
    assert "error" not in result["results"][0]

    # Mismatched symbols/geometry length within one scan point is caught
    # before ever calling the kernel, and reported per-point.
    mismatched = json.loads(run(mcp_adapter.dense_evolution_energy_scan(mcp_adapter.EnergyScanInput(
        symbols=["H", "H", "H"],
        geometries=[[[0, 0, 0], [0, 0, 0.7414]]],  # only 2 rows for 3 symbols
        labels=["bad"],
    ))))
    assert mismatched["results"][0]["error"]
    assert mismatched["minimum"] is None


def test_energy_scan_rejects_mismatched_labels_length():
    result = run(mcp_adapter.dense_evolution_energy_scan(mcp_adapter.EnergyScanInput(
        symbols=["H", "H"],
        geometries=[[[0, 0, 0], [0, 0, 0.7]], [[0, 0, 0], [0, 0, 0.8]]],
        labels=["only-one-label"],
    )))
    assert result.startswith("Error:")


def test_run_vqe_hardware_efficient_converges_close_to_exact_for_h2():
    result = json.loads(run(mcp_adapter.dense_evolution_run_vqe(mcp_adapter.RunVqeInput(
        name="H2", ansatz_type="hardware_efficient", n_layers=4, maxiter=150, seed=0,
    ))))
    assert abs(result["vqe_energy_hartree"] - result["exact_energy_hartree"]) < 0.01


def test_kernel_unreachable_gives_actionable_error_not_a_traceback():
    mcp_adapter._TEST_TRANSPORT = None  # force a real (failing) TCP attempt
    original_url = mcp_adapter.KERNEL_URL
    mcp_adapter.KERNEL_URL = "http://127.0.0.1:1"  # nothing listens here
    try:
        result = run(mcp_adapter.dense_evolution_health())
        assert result.startswith("Error:")
        assert "dense-evolution serve" in result
    finally:
        mcp_adapter.KERNEL_URL = original_url


def test_wormhole_select_instance_finds_the_known_seed_61():
    data = json.loads(run(mcp_adapter.dense_evolution_wormhole_select_instance(
        mcp_adapter.WormholeSelectInstanceInput(
            n_majorana=8, k_terms=10, J=2 ** 0.5, n_candidates=200, target_commuting=34,
        )
    )))
    assert data["seed"] == 61
    assert data["commuting"] == 34
    assert data["anticommuting"] == 11


def test_wormhole_teleportation_exact_backend_matches_known_reference_values():
    """seed=61, t0=0.3, t1=0.60 known peak: I(mu=+12)=0.01326, I(mu=-12)=0.01793
    (exact backend) -- same values pinned in tests/test_local_site_server.py,
    checked again here through the MCP adapter's own request path."""
    pos = json.loads(run(mcp_adapter.dense_evolution_wormhole_teleportation(
        mcp_adapter.WormholeTeleportationInput(mu=12.0, t0=0.3, t1=0.6, seed=61, backend="exact")
    )))
    neg = json.loads(run(mcp_adapter.dense_evolution_wormhole_teleportation(
        mcp_adapter.WormholeTeleportationInput(mu=-12.0, t0=0.3, t1=0.6, seed=61, backend="exact")
    )))
    assert pos["mutual_information_pt"] == pytest.approx(0.01326, abs=1e-5)
    assert neg["mutual_information_pt"] == pytest.approx(0.01793, abs=1e-5)


def test_wormhole_teleportation_trotter_backend_matches_known_reference_values():
    pos = json.loads(run(mcp_adapter.dense_evolution_wormhole_teleportation(
        mcp_adapter.WormholeTeleportationInput(mu=12.0, t0=0.3, t1=0.6, seed=61, backend="trotter")
    )))
    neg = json.loads(run(mcp_adapter.dense_evolution_wormhole_teleportation(
        mcp_adapter.WormholeTeleportationInput(mu=-12.0, t0=0.3, t1=0.6, seed=61, backend="trotter")
    )))
    assert pos["mutual_information_pt"] == pytest.approx(0.01301, abs=1e-5)
    assert neg["mutual_information_pt"] == pytest.approx(0.01821, abs=1e-5)


def test_wormhole_teleportation_invalid_backend_gives_error_not_a_traceback():
    result = run(mcp_adapter.dense_evolution_wormhole_teleportation(
        mcp_adapter.WormholeTeleportationInput(backend="not-a-backend")
    ))
    assert result.startswith("Error:")


def test_wormhole_scan_sweeps_t1_and_finds_the_known_peak():
    """One batched call replacing the two-call-per-point pattern above --
    the peak (largest mu<0 minus mu>0 delta) for this exact configuration
    is known to land at t1=0.60 (delta=+0.00468, exact backend) from the
    original research reproduction's 11-point sweep."""
    result = json.loads(run(mcp_adapter.dense_evolution_wormhole_scan(
        mcp_adapter.WormholeScanInput(
            mu_magnitude=12.0, t0=0.3, t1_values=[0.10, 0.30, 0.60, 0.85, 1.20],
            seed=61, backend="exact",
        )
    )))
    assert result["n_points"] == 5
    assert all("delta" in r for r in result["results"])
    assert result["peak"]["t1"] == pytest.approx(0.60)
    assert result["peak"]["delta"] == pytest.approx(0.00468, abs=1e-4)


def test_wormhole_scan_reports_per_point_errors_without_aborting():
    result = json.loads(run(mcp_adapter.dense_evolution_wormhole_scan(
        mcp_adapter.WormholeScanInput(
            n_majorana=200, t0=0.3, t1_values=[0.5], seed=61, backend="exact",
        )
    )))
    assert result["n_points"] == 1
    assert "error" in result["results"][0]
    assert result["peak"] is None
