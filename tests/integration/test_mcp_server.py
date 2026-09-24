"""
Tests for mcp_server/server.py -- the dense_evolution_mcp MCP adapter.

Routes every request through httpx.ASGITransport straight into the real,
in-process local_site.app.server.app (client._TEST_TRANSPORT, swapped via
monkeypatch.setattr in the autouse fixture below -- not a raw manual
global mutation, for automatic restoration even if a test fails partway
through) instead of a live subprocess kernel bound to a real port. That
means every test below still exercises the real DenseSVSimulator / real PennyLane
Hartree-Fock Hamiltonians -- no mocked physics, only the network hop is
swapped for an in-process one, for the same reason test_local_site_server.py
uses FastAPI's TestClient rather than a live server.
"""
import asyncio
import base64
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
from mcp_server import client as mcp_client  # noqa: E402
from mcp_server.utils import images as mcp_images  # noqa: E402
from mcp_server import models as mcp_models  # noqa: E402
from mcp_server import molecules as mcp_molecules  # noqa: E402

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


# prog.txt test-suite audit (issue #269 point 7): run() used to be
# `asyncio.run(coro)` -- a fresh event loop per call, immediately closed
# afterward. client.py::_get_client() caches the shared httpx.AsyncClient
# across calls, so the second call's asyncio.run() would hand it a
# NEW loop while its transport/connection-pool internals may still be
# bound to the FIRST (now-closed) one -- test_request_reuses_the_same_client_across_calls
# only ever passed because httpx.ASGITransport (the test-only transport
# used here) doesn't hold any loop-bound state the way a real TCP
# transport's connection pool does; that test was verifying something
# never actually exercised the way production (one long-lived stdio
# loop, see server.py's main()) uses this client. A single loop for the
# whole module, closed once at teardown, makes the caching this suite
# already asserts on genuinely tested against a stable loop instead.
_shared_loop = asyncio.new_event_loop()


def run(coro):
    return _shared_loop.run_until_complete(coro)


@pytest.fixture(scope="module", autouse=True)
def _close_shared_loop_at_module_end():
    yield
    _shared_loop.close()


@pytest.fixture(autouse=True)
def route_through_real_kernel_in_process(monkeypatch):
    monkeypatch.setattr(mcp_client, "_TEST_TRANSPORT", httpx.ASGITransport(app=kernel.app))
    # A fresh molecule-alias cache per test: several tests below rely on
    # the real /api/hamiltonians catalog being (re)fetched, not whatever a
    # previous test happened to populate.
    mcp_molecules._molecule_catalog_cache.invalidate()
    yield
    mcp_molecules._molecule_catalog_cache.invalidate()


def test_health_reports_real_kernel_info():
    data = json.loads(run(mcp_adapter.dense_evolution_health()))
    assert data["status"] == "ok"
    assert data["dense_evolution_version"] == kernel.dense_evolution.__version__
    assert data["total_ram_gb"] > 0


def test_system_limits_reports_positive_qubit_ceiling():
    data = json.loads(run(mcp_adapter.dense_evolution_system_limits()))
    assert data["max_qubits_dense"] > 0


def test_kernel_status_reports_adapter_local_state():
    data = json.loads(run(mcp_adapter.dense_evolution_kernel_status()))
    assert data["kernel_reachable"] is True
    assert data["kernel_url"] == mcp_client.KERNEL_URL
    assert data["molecule_cache_entries"] >= 0
    assert "image_output_dir" in data


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
    result = run(mcp_adapter.dense_evolution_build_circuit(mcp_models.BuildCircuitInput(n_qubits=2, ops=ops)))
    data = json.loads(result)
    assert "OPENQASM" in data["qasm"]
    assert "h q[0]" in data["qasm"]


def test_custom_molecule_energy_matches_catalog_h2():
    result = json.loads(run(mcp_adapter.dense_evolution_custom_molecule_energy(mcp_models.CustomMoleculeInput(
        symbols=["H", "H"], geometry=[[0, 0, 0], [0, 0, 0.7414]],
    ))))
    assert result["ground_state_energy_hartree"] == pytest.approx(EXACT_H2_ENERGY_HARTREE, abs=1e-6)


def test_qmmm_forces_returns_real_force_vectors():
    result = json.loads(run(mcp_adapter.dense_evolution_qmmm_forces(mcp_models.QmmmForcesInput(name="H2"))))
    assert "forces" in result or "force_norm" in result or "energy_hartree" in result


def test_md_trajectory_runs_a_few_fixed_electronic_state_steps():
    result = json.loads(run(mcp_adapter.dense_evolution_md_trajectory(mcp_models.MdTrajectoryInput(
        name="H2", n_steps=2, recompute_electronic_state=False,
    ))))
    assert "error" not in result


def test_mitigate_zne_richardson_extrapolates_toward_ideal():
    # Single-seed, single-comparison assertions here are genuinely
    # statistical, not deterministic: NoiseModel.apply_to_sv's depolarizing
    # channel is a real stochastic single-qubit-per-shot Kraus draw (fixed
    # 2026-08-11, see registry.py's changelog entry -- the channel used to
    # decide fire/no-fire independently per computational-basis amplitude
    # pair instead of once per qubit per shot, which understated true
    # per-shot variance on entangled states). At the default n_trials=200
    # per noise scale, a single fixed seed can land on the unlucky side of
    # the distribution (verified directly: 42's own default seed passes
    # only ~70% of nearby seeds at n_trials=200). Average the comparison
    # over several seeds instead of trusting one arbitrary seed's outcome
    # -- this is the statistically honest form of the same physics check.
    ideal_ok = True
    zne_gaps = []
    noisiest_gaps = []
    for seed in range(5):
        result = json.loads(run(mcp_adapter.dense_evolution_mitigate_zne(mcp_models.MitigateZneInput(
            qasm=BELL_QASM, pauli_string="ZZ", noise_model="depolarizing", noise_p=0.05, seed=seed,
        ))))
        ideal_ok = ideal_ok and (result["ideal_expectation"] == pytest.approx(1.0, abs=1e-6))
        zne_gaps.append(abs(result["zne_extrapolated"] - 1.0))
        noisiest_gaps.append(abs(result["noisy_expectations"][-1] - 1.0))
    assert ideal_ok
    # Real physics check, not just "did it return something": averaged
    # over several seeds, ZNE should land closer to ideal than the
    # noisiest single measurement.
    assert sum(zne_gaps) / len(zne_gaps) < sum(noisiest_gaps) / len(noisiest_gaps)


def test_mitigate_density_matrix_reports_fidelity_improvement():
    result = json.loads(run(mcp_adapter.dense_evolution_mitigate_density_matrix(
        mcp_models.MitigateDensityMatrixInput(qasm=BELL_QASM, noise_model="depolarizing", noise_p=0.05)
    )))
    assert 0.0 <= result["fidelity_raw"] <= 1.0
    assert 0.0 <= result["fidelity_corrected"] <= 1.0


def test_vector_healing_replaces_an_outlier_with_the_local_median():
    vectors = [[1.0, 2.0], [1.1, 2.1], [1.05, 2.05], [50.0, -30.0], [1.08, 2.08], [1.02, 2.02]]
    result = json.loads(run(mcp_adapter.dense_evolution_vector_healing(
        mcp_models.VectorHealingInput(vectors=vectors)
    )))
    assert len(result["healed_vectors"]) == 6
    assert result["healed_vectors"][3] != vectors[3]
    assert result["reconstruction_error"] > 0.0


def test_list_molecules_includes_short_ids():
    data = json.loads(run(mcp_adapter.dense_evolution_list_molecules(mcp_models.ListMoleculesInput())))
    by_id = {m["id"]: m for m in data}
    assert "H2" in by_id
    assert by_id["H2"]["full_name"] == H2
    assert by_id["H2"]["n_qubits"] == 4


def test_molecule_energy_accepts_short_id():
    result = run(mcp_adapter.dense_evolution_molecule_energy(mcp_models.MoleculeEnergyInput(name="H2")))
    data = json.loads(result)
    assert data["ground_state_energy_hartree"] == pytest.approx(EXACT_H2_ENERGY_HARTREE, abs=1e-6)


def test_molecule_energy_accepts_full_catalog_name():
    result = run(mcp_adapter.dense_evolution_molecule_energy(mcp_models.MoleculeEnergyInput(name=H2)))
    data = json.loads(result)
    assert data["ground_state_energy_hartree"] == pytest.approx(EXACT_H2_ENERGY_HARTREE, abs=1e-6)


def test_molecule_energy_unknown_name_is_a_clear_error_not_a_crash():
    result = run(mcp_adapter.dense_evolution_molecule_energy(mcp_models.MoleculeEnergyInput(name="Unobtainium")))
    assert result.startswith("Error:")
    assert "Unobtainium" in result


def test_mix_molecules_accepts_short_ids():
    result = run(mcp_adapter.dense_evolution_mix_molecules(
        mcp_models.MixMoleculesInput(name_a="H2", name_b="HeH+", weight_a=0.5, weight_b=0.5)
    ))
    # H2 (4 qubits) and HeH+ (4 qubits) share a qubit count -- a real,
    # meaningful mix; the tool should resolve both short ids and succeed.
    data = json.loads(result)
    assert "energy_mixed" in data


def test_run_circuit_bell_state_gives_real_50_50_split():
    result = run(mcp_adapter.dense_evolution_run_circuit(
        mcp_models.RunCircuitInput(qasm=BELL_QASM, shots=500, seed=42)
    ))
    data = json.loads(result)
    assert data["n_qubits"] == 2
    assert set(data["counts"]) == {"00", "11"}
    assert data["counts"]["00"] + data["counts"]["11"] == 500
    # Both truncation summaries should be present and self-consistent.
    assert data["statevector"]["total_nonzero_amplitudes"] == 2
    assert data["probabilities"]["total_basis_states"] == 4


def test_run_circuit_top_k_is_configurable():
    # BUG FIX: top_k used to be hardcoded at 25 inside the truncation
    # helpers with no caller-facing way to change it. A 3-qubit GHZ-ish
    # circuit only has 2 nonzero amplitudes/basis states, so top_k=1 is
    # the meaningful case to check truncation actually engages.
    ghz_qasm = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];
h q[0];
cx q[0],q[1];
cx q[1],q[2];
"""
    result = run(mcp_adapter.dense_evolution_run_circuit(
        mcp_models.RunCircuitInput(qasm=ghz_qasm, shots=50, top_k=1)
    ))
    data = json.loads(result)
    assert data["statevector"]["total_nonzero_amplitudes"] == 2
    assert data["statevector"]["shown"] == 1
    assert len(data["statevector"]["top_amplitudes_by_magnitude"]) == 1
    assert data["probabilities"]["shown"] == 1
    assert len(data["probabilities"]["top_states_by_probability"]) == 1


def test_run_circuit_visualizations_are_saved_to_disk_not_inlined(tmp_path):
    original_dir = mcp_images.IMAGE_OUTPUT_DIR
    mcp_images.IMAGE_OUTPUT_DIR = tmp_path
    try:
        result = run(mcp_adapter.dense_evolution_run_circuit(
            mcp_models.RunCircuitInput(qasm=BELL_QASM, shots=50, include_visualizations=True)
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
        mcp_images.IMAGE_OUTPUT_DIR = original_dir


def test_run_circuit_image_metadata_sidecar_is_written(tmp_path):
    # BUG FIX: a saved image's filename (circuit_<timestamp>.png) used to
    # carry no way to trace it back to the circuit/tool call that produced
    # it. Every PNG dense_evolution_run_circuit saves should now have a
    # same-stem .json sidecar identifying the source qasm/seed/etc.
    original_dir = mcp_images.IMAGE_OUTPUT_DIR
    mcp_images.IMAGE_OUTPUT_DIR = tmp_path
    try:
        result = run(mcp_adapter.dense_evolution_run_circuit(
            mcp_models.RunCircuitInput(qasm=BELL_QASM, shots=50, seed=7, include_visualizations=True)
        ))
        data = json.loads(result)
        import pathlib
        png_path = pathlib.Path(data["circuit_png_path"])
        json_path = png_path.with_suffix(".json")
        assert json_path.exists()
        meta = json.loads(json_path.read_text())
        assert meta["tool"] == "dense_evolution_run_circuit"
        assert meta["qasm"] == BELL_QASM
        assert meta["seed"] == 7
    finally:
        mcp_images.IMAGE_OUTPUT_DIR = original_dir


def test_prune_old_images_keeps_only_the_most_recent_max_files(tmp_path):
    # BUG FIX: nothing ever cleaned up IMAGE_OUTPUT_DIR -- a long-running
    # MCP session grows it without bound. Uses synthetic files with
    # explicit mtimes (not _save_png's real-clock filenames) so the
    # "oldest" ordering is deterministic instead of racing real time.
    original_dir = mcp_images.IMAGE_OUTPUT_DIR
    original_max = mcp_images.IMAGE_MAX_FILES
    mcp_images.IMAGE_OUTPUT_DIR = tmp_path
    mcp_images.IMAGE_MAX_FILES = 3
    try:
        for i in range(5):
            p = tmp_path / f"img{i}.png"
            p.write_bytes(b"x")
            os.utime(p, (i, i))

        mcp_images._prune_old_images()

        remaining = {p.name for p in tmp_path.glob("*.png")}
        assert remaining == {"img2.png", "img3.png", "img4.png"}
    finally:
        mcp_images.IMAGE_OUTPUT_DIR = original_dir
        mcp_images.IMAGE_MAX_FILES = original_max


def test_prune_old_images_disabled_when_max_files_non_positive(tmp_path):
    original_dir = mcp_images.IMAGE_OUTPUT_DIR
    original_max = mcp_images.IMAGE_MAX_FILES
    mcp_images.IMAGE_OUTPUT_DIR = tmp_path
    mcp_images.IMAGE_MAX_FILES = 0
    try:
        for i in range(5):
            (tmp_path / f"img{i}.png").write_bytes(b"x")

        mcp_images._prune_old_images()

        assert len(list(tmp_path.glob("*.png"))) == 5
    finally:
        mcp_images.IMAGE_OUTPUT_DIR = original_dir
        mcp_images.IMAGE_MAX_FILES = original_max


def test_save_png_calls_prune_so_the_directory_stays_bounded(tmp_path):
    # End-to-end wiring check: _save_png itself must trigger pruning, not
    # just the helper in isolation. Real-clock filenames make exact
    # oldest-file identity unreliable at test speed, so this only checks
    # the invariant that actually matters: the directory never exceeds
    # the configured cap.
    original_dir = mcp_images.IMAGE_OUTPUT_DIR
    original_max = mcp_images.IMAGE_MAX_FILES
    mcp_images.IMAGE_OUTPUT_DIR = tmp_path
    mcp_images.IMAGE_MAX_FILES = 3
    try:
        tiny_png_b64 = base64.b64encode(b"not a real png but bytes are bytes").decode()
        for i in range(6):
            mcp_images._save_png(tiny_png_b64, f"shot{i}")
        assert len(list(tmp_path.glob("*.png"))) <= 3
    finally:
        mcp_images.IMAGE_OUTPUT_DIR = original_dir
        mcp_images.IMAGE_MAX_FILES = original_max


def test_molecule_catalog_second_call_hits_the_cache_not_the_network():
    # BUG FIX target: the pre-Phase-2 code never cached across calls in
    # production (only test code ever reset the old plain-dict cache) --
    # this checks the actual cache-HIT code path fires: a second call for
    # the same mapping must NOT reach _request again.
    first = run(mcp_adapter.dense_evolution_list_molecules(mcp_models.ListMoleculesInput()))

    async def _fail_if_called(*args, **kwargs):
        raise AssertionError("second call should have hit the cache, not _request")

    original_request = mcp_molecules._request
    mcp_molecules._request = _fail_if_called
    try:
        second = run(mcp_adapter.dense_evolution_list_molecules(mcp_models.ListMoleculesInput()))
    finally:
        mcp_molecules._request = original_request
    assert first == second


def test_molecule_catalog_fetch_failure_is_cached_and_reraised(monkeypatch):
    mcp_molecules._molecule_catalog_cache.invalidate()

    class _BrokenClient:
        async def request(self, method, path, **kwargs):
            raise httpx.ConnectError("simulated kernel down")

    monkeypatch.setattr(mcp_client, "_get_client", lambda: _BrokenClient())
    result = run(mcp_adapter.dense_evolution_list_molecules(mcp_models.ListMoleculesInput()))
    assert result.startswith("Error:")
    # Second call within the failure TTL must reuse the cached failure
    # (not attempt a fresh connection) -- proven by NOT patching _get_client
    # back yet and getting the same error shape again, fast.
    result_again = run(mcp_adapter.dense_evolution_list_molecules(mcp_models.ListMoleculesInput()))
    assert result_again.startswith("Error:")
    mcp_molecules._molecule_catalog_cache.invalidate()


def test_run_circuit_large_scale_response_includes_image_metadata(monkeypatch, tmp_path):
    # Mocks the kernel's own large_scale response shape (same technique
    # test_kernel_timeout_gives_actionable_error_not_a_traceback uses)
    # rather than driving a real >24-qubit MPS run, which would be slow
    # and depend on exact MPS internals unrelated to what this test checks:
    # that the large_scale branch's saved image also gets a metadata
    # sidecar, same as the normal branch already does.
    original_dir = mcp_images.IMAGE_OUTPUT_DIR
    mcp_images.IMAGE_OUTPUT_DIR = tmp_path
    tiny_png_b64 = base64.b64encode(b"not a real png but bytes are bytes").decode()

    class _FakeResponse:
        status_code = 200

        def json(self):
            return {
                "large_scale": True, "n_qubits": 30, "k_requested": 32,
                "top_k_states": [], "circuit_png": tiny_png_b64,
            }

    class _FakeClient:
        async def request(self, method, path, **kwargs):
            return _FakeResponse()

    monkeypatch.setattr(mcp_client, "_get_client", lambda: _FakeClient())
    try:
        result = run(mcp_adapter.dense_evolution_run_circuit(
            mcp_models.RunCircuitInput(qasm=BELL_QASM, include_visualizations=True)
        ))
        data = json.loads(result)
        assert data["large_scale"] is True
        assert "circuit_png" not in data
        import pathlib
        png_path = pathlib.Path(data["circuit_png_path"])
        assert png_path.with_suffix(".json").exists()
    finally:
        mcp_images.IMAGE_OUTPUT_DIR = original_dir


def test_save_png_returns_none_for_falsy_input():
    assert mcp_images._save_png(None, "whatever") is None
    assert mcp_images._save_png("", "whatever") is None


def test_energy_scan_matches_individual_molecule_energy_call():
    single = json.loads(run(mcp_adapter.dense_evolution_molecule_energy(mcp_models.MoleculeEnergyInput(name="H2"))))
    scan_result = json.loads(run(mcp_adapter.dense_evolution_energy_scan(mcp_models.EnergyScanInput(
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
    result = json.loads(run(mcp_adapter.dense_evolution_energy_scan(mcp_models.EnergyScanInput(
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
    result = json.loads(run(mcp_adapter.dense_evolution_energy_scan(mcp_models.EnergyScanInput(
        symbols=["H", "H"],
        geometries=[[[0, 0, 0], [0, 0, 0.7414]]],
        labels=["ok"],
    ))))
    assert "error" not in result["results"][0]

    # Mismatched symbols/geometry length within one scan point is caught
    # before ever calling the kernel, and reported per-point.
    mismatched = json.loads(run(mcp_adapter.dense_evolution_energy_scan(mcp_models.EnergyScanInput(
        symbols=["H", "H", "H"],
        geometries=[[[0, 0, 0], [0, 0, 0.7414]]],  # only 2 rows for 3 symbols
        labels=["bad"],
    ))))
    assert mismatched["results"][0]["error"]
    assert mismatched["minimum"] is None


def test_energy_scan_isolates_a_real_kernel_error_from_the_rest():
    # Distinct from test_energy_scan_isolates_a_failing_point_from_the_rest,
    # which only exercises the LOCAL symbols/geometry length check (never
    # reaches the kernel at all). This one needs a point where that local
    # check passes but the kernel itself rejects the request (13 H atoms
    # needs far more than the real, documented 12-qubit exact-diagonalization
    # cap) -- the try/except around the real _request call inside _one_point.
    big_symbols = ["H"] * 13
    big_geometry = [[0, 0, float(i)] for i in range(13)]
    result = json.loads(run(mcp_adapter.dense_evolution_energy_scan(mcp_models.EnergyScanInput(
        symbols=big_symbols,
        geometries=[big_geometry],
        labels=["too_big"],
    ))))
    assert "error" in result["results"][0]
    assert result["minimum"] is None


def test_energy_scan_rejects_mismatched_labels_length():
    result = run(mcp_adapter.dense_evolution_energy_scan(mcp_models.EnergyScanInput(
        symbols=["H", "H"],
        geometries=[[[0, 0, 0], [0, 0, 0.7]], [[0, 0, 0], [0, 0, 0.8]]],
        labels=["only-one-label"],
    )))
    assert result.startswith("Error:")


def test_run_vqe_hardware_efficient_converges_close_to_exact_for_h2():
    result = json.loads(run(mcp_adapter.dense_evolution_run_vqe(mcp_models.RunVqeInput(
        name="H2", ansatz_type="hardware_efficient", n_layers=4, maxiter=150, seed=0,
    ))))
    assert abs(result["vqe_energy_hartree"] - result["exact_energy_hartree"]) < 0.01


def test_kernel_unreachable_gives_actionable_error_not_a_traceback(monkeypatch):
    monkeypatch.setattr(mcp_client, "_TEST_TRANSPORT", None)  # force a real (failing) TCP attempt
    monkeypatch.setattr(mcp_client, "KERNEL_URL", "http://127.0.0.1:1")  # nothing listens here
    result = run(mcp_adapter.dense_evolution_health())
    assert result.startswith("Error:")
    assert "dense-evolution serve" in result


def test_kernel_status_reports_unreachable_kernel_without_raising(monkeypatch):
    monkeypatch.setattr(mcp_client, "_TEST_TRANSPORT", None)  # force a real (failing) TCP attempt
    monkeypatch.setattr(mcp_client, "KERNEL_URL", "http://127.0.0.1:1")  # nothing listens here
    data = json.loads(run(mcp_adapter.dense_evolution_kernel_status()))
    assert data["kernel_reachable"] is False
    assert data["kernel_url"] == "http://127.0.0.1:1"


def test_kernel_timeout_gives_actionable_error_not_a_traceback(monkeypatch):
    class _TimeoutClient:
        async def request(self, method, path, **kwargs):
            raise httpx.TimeoutException("simulated timeout")

    monkeypatch.setattr(mcp_client, "_get_client", lambda: _TimeoutClient())
    result = run(mcp_adapter.dense_evolution_health())
    assert result.startswith("Error:")
    assert "timed out" in result


def test_kernel_error_response_with_non_json_body_falls_back_to_raw_text(monkeypatch):
    # _request's status_code>=400 branch tries resp.json().get("detail", ...)
    # first and falls back to resp.text if the body isn't valid JSON (e.g. a
    # raw HTML 502 from a proxy in front of the kernel, not the kernel's own
    # structured FastAPI error response).
    class _FakeResponse:
        status_code = 502
        text = "<html>Bad Gateway</html>"

        def json(self):
            raise ValueError("not valid json")

    class _ErrorClient:
        async def request(self, method, path, **kwargs):
            return _FakeResponse()

    monkeypatch.setattr(mcp_client, "_get_client", lambda: _ErrorClient())
    result = run(mcp_adapter.dense_evolution_health())
    assert result.startswith("Error:")
    assert "<html>Bad Gateway</html>" in result


def test_wormhole_select_instance_finds_the_known_seed_61():
    data = json.loads(run(mcp_adapter.dense_evolution_wormhole_select_instance(
        mcp_models.WormholeSelectInstanceInput(
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
        mcp_models.WormholeTeleportationInput(mu=12.0, t0=0.3, t1=0.6, seed=61, backend="exact")
    )))
    neg = json.loads(run(mcp_adapter.dense_evolution_wormhole_teleportation(
        mcp_models.WormholeTeleportationInput(mu=-12.0, t0=0.3, t1=0.6, seed=61, backend="exact")
    )))
    assert pos["mutual_information_pt"] == pytest.approx(0.01326, abs=1e-5)
    assert neg["mutual_information_pt"] == pytest.approx(0.01793, abs=1e-5)


def test_wormhole_teleportation_trotter_backend_matches_known_reference_values():
    pos = json.loads(run(mcp_adapter.dense_evolution_wormhole_teleportation(
        mcp_models.WormholeTeleportationInput(mu=12.0, t0=0.3, t1=0.6, seed=61, backend="trotter")
    )))
    neg = json.loads(run(mcp_adapter.dense_evolution_wormhole_teleportation(
        mcp_models.WormholeTeleportationInput(mu=-12.0, t0=0.3, t1=0.6, seed=61, backend="trotter")
    )))
    assert pos["mutual_information_pt"] == pytest.approx(0.01301, abs=1e-5)
    assert neg["mutual_information_pt"] == pytest.approx(0.01821, abs=1e-5)


def test_wormhole_teleportation_invalid_backend_gives_error_not_a_traceback():
    result = run(mcp_adapter.dense_evolution_wormhole_teleportation(
        mcp_models.WormholeTeleportationInput(backend="not-a-backend")
    ))
    assert result.startswith("Error:")


def test_wormhole_scan_sweeps_t1_and_finds_the_known_peak():
    """One batched call replacing the two-call-per-point pattern above --
    the peak (largest mu<0 minus mu>0 delta) for this exact configuration
    is known to land at t1=0.60 (delta=+0.00468, exact backend) from the
    original research reproduction's 11-point sweep."""
    result = json.loads(run(mcp_adapter.dense_evolution_wormhole_scan(
        mcp_models.WormholeScanInput(
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
        mcp_models.WormholeScanInput(
            n_majorana=200, t0=0.3, t1_values=[0.5], seed=61, backend="exact",
        )
    )))
    assert result["n_points"] == 1
    assert "error" in result["results"][0]
    assert result["peak"] is None


def test_request_reuses_the_same_client_across_calls():
    # BUG FIX (perf): _request used to open a fresh httpx.AsyncClient
    # (TCP handshake) per call -- _get_client now caches and reuses one
    # as long as (_TEST_TRANSPORT, KERNEL_URL) haven't changed, which
    # they don't within a single test (the autouse fixture only swaps
    # _TEST_TRANSPORT between tests, not within one).
    run(mcp_adapter.dense_evolution_health())
    client_after_first_call = mcp_client._shared_client
    run(mcp_adapter.dense_evolution_health())
    assert mcp_client._shared_client is client_after_first_call


def test_run_helper_reuses_one_event_loop_not_a_fresh_one_per_call():
    # prog.txt test-suite audit (issue #269 point 7): this file's run()
    # used to be asyncio.run(coro) -- a new loop, closed immediately
    # after each call -- which the test above's cached-client assertion
    # never actually exercised realistically (only true because
    # ASGITransport holds no loop-bound state). Confirms run() now
    # executes on the same, still-open loop across calls, matching
    # production's single long-lived stdio loop (server.py's main()).
    async def _current_loop():
        return asyncio.get_running_loop()

    loop_a = run(_current_loop())
    loop_b = run(_current_loop())
    assert loop_a is loop_b
    assert not loop_a.is_closed()


def test_request_rebuilds_client_when_transport_or_url_changes(monkeypatch):
    run(mcp_adapter.dense_evolution_health())
    stale_client = mcp_client._shared_client

    monkeypatch.setattr(mcp_client, "_TEST_TRANSPORT", None)
    monkeypatch.setattr(mcp_client, "KERNEL_URL", "http://127.0.0.1:1")
    result = run(mcp_adapter.dense_evolution_health())
    assert result.startswith("Error:")  # confirms it actually hit the new (broken) target
    assert mcp_client._shared_client is not stale_client


def test_per_tool_timeouts_reach_the_real_http_call(monkeypatch):
    """Each tool used to share one flat 180s timeout regardless of real
    cost -- a fast health check waited as long as a slow VQE run would
    need, and a genuinely long VQE/MD run could get cut off by the same
    180s ceiling. Verifies the actual timeout= kwarg httpx.AsyncClient.request
    receives differs per tool and matches each endpoint's real expected
    cost, not that the call merely succeeds (which wouldn't catch a timeout
    value silently being ignored or defaulted)."""
    seen_timeouts = {}
    real_request = httpx.AsyncClient.request

    async def _spy_request(self, method, url, *args, **kwargs):
        seen_timeouts[url] = kwargs.get("timeout")
        return await real_request(self, method, url, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "request", _spy_request)

    run(mcp_adapter.dense_evolution_health())
    run(mcp_adapter.dense_evolution_run_vqe(mcp_models.RunVqeInput(
        name="H2", ansatz_type="hardware_efficient", n_layers=1, maxiter=1,
    )))

    assert seen_timeouts["/api/health"] == 5.0
    assert seen_timeouts["/api/vqe"] == 600.0
    assert seen_timeouts["/api/health"] != seen_timeouts["/api/vqe"]


def test_registered_tool_count_matches_documented_count():
    # prog.txt (dense_evolution_mcp audit, transversal note): server.py's
    # own module docstring used to claim "22 tools" while the real count
    # (counted directly from the imports at the top of that file: 7
    # system + 2 circuit + 7 chemistry + 3 mitigation + 3 wormhole + 3
    # noise) was 25 -- undetected drift, not a functional bug, but the
    # exact kind a trivial len()==N test catches for free going forward.
    # Now 32: +1 chemistry (native_hf_diagnostics), +1 chemistry
    # (mass_decomposition), +1 mitigation (coherence), +3 crypto
    # (bb84/di_qkd_ghz/dicka), +1 retrieval (rag_search).
    assert len(mcp_adapter.mcp._tool_manager._tools) == 32


def test_kernel_error_response_kind_is_classified_and_prefix_preserved():
    # Issue #258 point 10: catch_errors' output stays "Error: ..." (every
    # existing result.startswith("Error:") check keeps working) but now
    # carries a stable "Error: <kind>: ..." token a client can branch on.
    result = run(mcp_adapter.dense_evolution_energy_scan(mcp_models.EnergyScanInput(
        symbols=["H", "H"],
        geometries=[[[0, 0, 0], [0, 0, 0.7]], [[0, 0, 0], [0, 0, 0.8]]],
        labels=["only-one-label"],
    )))
    assert result.startswith("Error: invalid_input:")


def test_kernel_unreachable_error_is_classified_as_unreachable(monkeypatch):
    monkeypatch.setattr(mcp_client, "_TEST_TRANSPORT", None)
    monkeypatch.setattr(mcp_client, "KERNEL_URL", "http://127.0.0.1:1")
    result = run(mcp_adapter.dense_evolution_health())
    assert result.startswith("Error: unreachable:")


def test_kernel_remote_protocol_error_gives_actionable_blas_hint(monkeypatch):
    class _ProtocolErrorClient:
        async def request(self, method, path, **kwargs):
            raise httpx.RemoteProtocolError("simulated malformed response")

    monkeypatch.setattr(mcp_client, "_get_client", lambda: _ProtocolErrorClient())
    result = run(mcp_adapter.dense_evolution_health())
    assert result.startswith("Error: protocol_error:")
    assert "BLAS" in result


def test_kernel_read_error_gives_actionable_error(monkeypatch):
    class _ReadErrorClient:
        async def request(self, method, path, **kwargs):
            raise httpx.ReadError("simulated connection closed mid-response")

    monkeypatch.setattr(mcp_client, "_get_client", lambda: _ReadErrorClient())
    result = run(mcp_adapter.dense_evolution_health())
    assert result.startswith("Error: connection_closed:")


def test_generic_httpx_error_falls_back_to_kernel_error_kind(monkeypatch):
    # Covers _request's generic `except httpx.HTTPError` catch-all --
    # anything in httpx's exception hierarchy not specific enough to get
    # its own KernelError subclass above (WriteError here, arbitrarily).
    class _WriteErrorClient:
        async def request(self, method, path, **kwargs):
            raise httpx.WriteError("simulated write failure")

    monkeypatch.setattr(mcp_client, "_get_client", lambda: _WriteErrorClient())
    result = run(mcp_adapter.dense_evolution_health())
    assert result.startswith("Error: kernel_error:")


def test_handle_error_classifies_a_non_kernel_non_value_error_as_internal():
    # Covers _handle_error's final "internal" branch -- an exception that
    # is neither a KernelError (kernel-communication failure) nor a
    # ValueError (input validation).
    assert mcp_client._handle_error(KeyError("boom")) == "Error: internal: 'boom'"


def test_close_shared_client_closes_and_resets_the_cached_client():
    # Issue #258 point 6: close_shared_client() is the actual shutdown
    # hook (called from server.py's main() try/finally) -- verify it both
    # closes the real client's connection pool and resets the module
    # cache so a later call rebuilds a fresh one instead of reusing a
    # closed client.
    run(mcp_adapter.dense_evolution_health())
    client = mcp_client._shared_client
    assert client is not None and not client.is_closed

    run(mcp_client.close_shared_client())
    assert client.is_closed
    assert mcp_client._shared_client is None

    run(mcp_adapter.dense_evolution_health())
    assert mcp_client._shared_client is not None
    assert mcp_client._shared_client is not client


async def _noop_ensure_kernel_running(*args, **kwargs):
    pass


def test_main_closes_the_shared_client_after_mcp_run_returns(monkeypatch):
    from mcp_server.server import main

    # main() now calls ensure_kernel_running() before mcp.run() -- these
    # three tests are about shutdown/close behavior, not startup, so it's
    # stubbed out rather than exercised for real (it would work fine
    # against the in-process kernel via _TEST_TRANSPORT, but coupling
    # these tests to that adds an unrelated real async call on a fresh
    # event loop -- see ensure_kernel_running's own test below instead).
    monkeypatch.setattr(mcp_client, "ensure_kernel_running", _noop_ensure_kernel_running)
    monkeypatch.setattr(mcp_adapter.mcp, "run", lambda: None)
    closed = []

    async def _fake_close():
        closed.append(True)

    monkeypatch.setattr(mcp_client, "close_shared_client", _fake_close)
    main()
    assert closed == [True]


def test_main_still_closes_the_shared_client_if_mcp_run_raises(monkeypatch):
    from mcp_server.server import main

    monkeypatch.setattr(mcp_client, "ensure_kernel_running", _noop_ensure_kernel_running)

    def _raising_run():
        raise ValueError("simulated stdio transport crash")

    monkeypatch.setattr(mcp_adapter.mcp, "run", _raising_run)
    closed = []

    async def _fake_close():
        closed.append(True)

    monkeypatch.setattr(mcp_client, "close_shared_client", _fake_close)
    with pytest.raises(ValueError, match="simulated stdio transport crash"):
        main()
    assert closed == [True]


def test_main_swallows_runtime_error_from_close_shared_client(monkeypatch):
    from mcp_server.server import main

    monkeypatch.setattr(mcp_client, "ensure_kernel_running", _noop_ensure_kernel_running)
    monkeypatch.setattr(mcp_adapter.mcp, "run", lambda: None)

    async def _raising_close():
        raise RuntimeError("simulated event-loop-already-closed at shutdown")

    monkeypatch.setattr(mcp_client, "close_shared_client", _raising_close)
    main()  # must not raise -- the RuntimeError is swallowed


def test_version_mismatch_message_none_when_versions_match():
    import dense_evolution
    assert mcp_client.version_mismatch_message({"dense_evolution_version": dense_evolution.__version__}) is None


def test_version_mismatch_message_none_when_key_missing():
    assert mcp_client.version_mismatch_message({}) is None


def test_version_mismatch_message_describes_the_drift():
    message = mcp_client.version_mismatch_message({"dense_evolution_version": "0.0.1-fake"})
    assert message is not None
    assert "0.0.1-fake" in message
    assert "dense-evolution serve" in message


def test_dense_evolution_health_surfaces_version_mismatch(monkeypatch):
    from mcp_server.tools import system_tools

    async def _fake_request(*args, **kwargs):
        return {"status": "ok", "dense_evolution_version": "0.0.1-fake", "hostname": "test", "total_ram_gb": 1, "available_ram_gb": 1, "ram_percent_free": 50}

    monkeypatch.setattr(system_tools, "_request", _fake_request)
    result = json.loads(run(mcp_adapter.dense_evolution_health()))
    assert "version_mismatch" in result
    assert "0.0.1-fake" in result["version_mismatch"]


def test_dense_evolution_health_has_no_mismatch_field_when_versions_match():
    result = json.loads(run(mcp_adapter.dense_evolution_health()))
    assert "version_mismatch" not in result


def test_ensure_kernel_running_skips_when_kernel_url_overridden(monkeypatch):
    monkeypatch.setattr(mcp_client, "_KERNEL_URL_IS_DEFAULT", False)
    calls = []
    monkeypatch.setattr(mcp_client.subprocess, "Popen", lambda *a, **k: calls.append((a, k)))
    run(mcp_client.ensure_kernel_running())
    assert calls == []


def test_ensure_kernel_running_does_not_spawn_when_kernel_already_up(monkeypatch):
    # Under the autouse fixture, the real in-process kernel is already
    # reachable via _TEST_TRANSPORT -- ensure_kernel_running should see
    # that and never attempt to spawn a subprocess.
    monkeypatch.setattr(mcp_client, "_KERNEL_URL_IS_DEFAULT", True)
    calls = []
    monkeypatch.setattr(mcp_client.subprocess, "Popen", lambda *a, **k: calls.append((a, k)))
    run(mcp_client.ensure_kernel_running())
    assert calls == []


def test_mitigate_coherence_reports_fidelity_improvement():
    result = json.loads(run(mcp_adapter.dense_evolution_mitigate_coherence(
        mcp_models.MitigateDensityMatrixInput(qasm=BELL_QASM, noise_model="phaseflip", noise_p=0.1)
    )))
    assert 0.0 <= result["fidelity_raw"] <= 1.0
    assert 0.0 <= result["fidelity_corrected"] <= 1.0


def test_crypto_bb84_perfect_channel_has_zero_qber():
    result = json.loads(run(mcp_adapter.dense_evolution_crypto_bb84(
        mcp_models.Bb84Input(n_rounds=500, p_channel=0.0, seed=0)
    )))
    assert result["qber"] == pytest.approx(0.0, abs=1e-9)
    assert result["sifted_key_length"] > 0


def test_crypto_di_qkd_ghz_near_quantum_max_on_ideal_channel():
    result = json.loads(run(mcp_adapter.dense_evolution_crypto_di_qkd_ghz(
        mcp_models.DiQkdGhzInput(n_rounds=500, p_dep=0.0, seed=0)
    )))
    assert result["win_rate"] == pytest.approx(0.8535533905932737, abs=0.02)
    assert result["win_rate"] > 0.75  # clears the classical bound


def test_crypto_dicka_returns_full_shape():
    result = json.loads(run(mcp_adapter.dense_evolution_crypto_dicka(
        mcp_models.DickaInput(n_rounds=200, gamma=0.1, beta=0.8, p_dep=0.0, seed=0)
    )))
    assert set(result.keys()) == {"n_rounds", "n_test", "n_key", "p_hat", "beta", "aborted", "qber_b1", "qber_b2"}


def test_mass_decomposition_glucose_water_loss():
    result = json.loads(run(mcp_adapter.dense_evolution_mass_decomposition(
        mcp_models.MassDecompositionInput(formula="C6H12O6", target_mass=18.0106)
    )))
    assert result["nearest_reachable_mass"] == pytest.approx(18.010565, abs=1e-4)


def test_native_hf_diagnostics_on_h2():
    pytest.importorskip("dense_armor")
    result = json.loads(run(mcp_adapter.dense_evolution_native_hf_diagnostics(
        mcp_models.NativeHfDiagnosticsInput(symbols=["H", "H"], geometry=[[0, 0, 0], [0, 0, 0.74]])
    )))
    assert result["converged"] is True
    assert result["total_energy_hartree"] == pytest.approx(-1.116759307489317, abs=1e-6)


def test_rag_search_exact_finds_a_known_phrase():
    pytest.importorskip("sklearn")
    result = json.loads(run(mcp_adapter.dense_evolution_rag_search(
        mcp_models.RagSearchInput(
            documents=[["The traversable wormhole construction couples two boundaries.", "gao.pdf"]],
            query="two boundaries", exact=True,
        )
    )))
    assert result["results"][0]["source"] == "gao.pdf"
    assert result["results"][0]["match"] == "two boundaries"
