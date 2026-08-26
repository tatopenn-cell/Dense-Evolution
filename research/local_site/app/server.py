"""
The real local compute kernel behind the published Composer page
(docs/composer.md, part of the main site -- not a separate docs project
anymore). This process is the only thing that runs on your own machine:
`pip install dense-evolution[composer]` then `dense-evolution serve` (or
directly `python -m local_site.app.server`), and the public page's
JavaScript talks to it at http://127.0.0.1:8800 as its execution backend.
It serves no HTML of its own -- the UI lives in the real published docs
site, this is API-only.

Every endpoint below is a thin wrapper around the real, already-tested
dashboard_core functions (engine.run_circuit_from_qasm, visuals.*,
graphical_builder.ops_to_native_tuples) -- dense_evolution's actual
DenseSVSimulator does the computation; the circuit diagram is drawn
natively (dashboard_core.circuit_diagram, no Qiskit), the other three
panels by Qiskit's own visualization functions. No mock data anywhere
here.
"""

import base64
import io
import sys
from pathlib import Path
from typing import List, Optional

import matplotlib
# FastAPI/Starlette runs sync route handlers in a worker thread, not the
# main thread -- matplotlib's default backend on Windows tries to open an
# interactive GUI window there, which either warns ("Starting a
# Matplotlib GUI outside of the main thread will likely fail") or hangs
# outright (observed directly: /api/run stuck mid-request with no
# response). Agg is the headless, thread-safe raster backend -- must be
# set before any other matplotlib/qiskit.visualization import.
matplotlib.use("Agg")

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# dashboard_core / dense_evolution live at the repo root, two levels up
# from this file (local_site/app/server.py -> local_site/ -> repo root).
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import dashboard_core as dc  # noqa: E402
import dense_evolution  # noqa: E402

app = FastAPI(title="Dense-Evolution Composer Kernel")

# The Composer page (docs/composer.md, part of the main site, published on
# GitHub Pages at tatopenn-cell.github.io) calls this local server as its
# compute kernel: the page itself is served from GitHub's origin (or opened
# straight from a downloaded copy on disk, origin "null"), this server from
# 127.0.0.1, so the browser treats every /api/* call as cross-origin and
# blocks it without this. localhost:8000 is `mkdocs serve`'s own dev server,
# included so the docs site can be developed against a locally-running
# kernel too, not just the published one. "null" is what browsers send as
# Origin for a page opened via file:// (the offline-downloaded copy) -- not
# a wildcard: this API executes real code (arbitrary OpenQASM), so only
# these known, intended origins are allowed rather than any page on the
# internet that happens to guess the port.
ALLOWED_ORIGINS = [
    "https://tatopenn-cell.github.io",
    "http://127.0.0.1:8000",
    "http://localhost:8000",
    "null",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class PrivateNetworkAccessMiddleware:
    """Chromium (and anything built on it, including VS Code's Simple
    Browser) treats a page loaded from a public HTTPS origin talking to
    127.0.0.1 as a "Private Network Access" request: on the CORS preflight
    it adds `Access-Control-Request-Private-Network: true` and requires
    the response to answer `Access-Control-Allow-Private-Network: true`,
    on top of ordinary CORS -- Starlette's CORSMiddleware doesn't know
    about this (newer, Chromium-specific) header at all and instead
    rejects the preflight outright with 400 "Disallowed CORS
    private-network" (verified directly: curl'd the exact preflight the
    real browser sends and got that 400). Without this, the Composer
    page's "detect the local kernel" banner never turns green in Chrome/
    Edge/VS Code, even though the kernel is up and every plain curl
    request succeeds -- curl never sends that header, so it doesn't hit
    this path, which is why the failure wasn't visible from the command
    line alone.

    Installed as ASGI middleware (added after CORSMiddleware, so it runs
    first on the request / last on the response) rather than routed
    through CORSMiddleware itself: only intercepts the specific preflight
    shape this needs (OPTIONS + that header, from an already-allowed
    origin), so it can't accidentally relax CORS for anything else --
    every other request still goes through CORSMiddleware unchanged."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        is_preflight = scope["method"] == "OPTIONS"
        wants_private_network = headers.get(b"access-control-request-private-network") == b"true"
        origin = headers.get(b"origin", b"").decode("latin-1")

        if is_preflight and wants_private_network and origin in ALLOWED_ORIGINS:
            response_headers = [
                (b"access-control-allow-origin", origin.encode("latin-1")),
                (b"access-control-allow-private-network", b"true"),
                (b"access-control-allow-methods", b"GET, POST"),
                (b"access-control-allow-headers", b"*"),
                (b"vary", b"Origin"),
                (b"content-length", b"0"),
            ]
            await send({"type": "http.response.start", "status": 200, "headers": response_headers})
            await send({"type": "http.response.body", "body": b""})
            return

        await self.app(scope, receive, send)


app.add_middleware(PrivateNetworkAccessMiddleware)


def _has_qiskit() -> bool:
    """qiskit is an optional dependency of this kernel (see pyproject.toml's
    composer extra comment for why: it's the same library independently
    described as destabilizing the process on macOS CI runners, so it's
    opt-in, not required just to start). Histogram/Q-sphere/Bloch are the
    only three panels that need it (the circuit diagram is native) -- this
    lets /api/run skip exactly those three with an honest reason instead of
    the whole endpoint failing with a raw 500 when it isn't installed."""
    try:
        import qiskit  # noqa: F401
        return True
    except ImportError:
        return False


@app.get("/api/health")
def health():
    """Presence probe for the published Composer page: if this responds,
    a real local kernel is installed and running on this machine, so the
    page can unlock live execution instead of showing install instructions.
    version is dense_evolution's own, not this API's -- lets the page warn
    if the installed kernel is old enough to be missing an endpoint it needs.

    hostname/total_ram_gb are returned so the page can show concrete,
    checkable proof of which real machine answered ("connected to
    DESKTOP-ABC123, 16.0 GB RAM") instead of an unverifiable claim like
    "circuits really run on your PC" -- a visitor can open Task Manager/
    a terminal and confirm this hostname and RAM figure are their own
    machine's, which a generic sentence gives them no way to check."""
    import socket
    import psutil
    mem = psutil.virtual_memory()
    return {
        "status": "ok",
        "dense_evolution_version": dense_evolution.__version__,
        "hostname": socket.gethostname(),
        "total_ram_gb": round(mem.total / 1e9, 1),
        "available_ram_gb": round(mem.available / 1e9, 1),
        "ram_percent_free": round(100 - mem.percent, 1),
    }


def _figure_to_base64_png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


class RunRequest(BaseModel):
    qasm: str
    shots: int = 1000
    seed: int = 42
    noise_model: str = "ideal"
    noise_p: float = 0.0
    backend: str = "dense"


class BuildRequest(BaseModel):
    n_qubits: int
    ops: list


@app.post("/api/build_from_ops")
def build_from_ops(req: BuildRequest):
    """Convert the graphical builder's op list into real OpenQASM text."""
    try:
        native_ops = dc.ops_to_native_tuples(req.n_qubits, req.ops)
        return {"qasm": dc.gate_tuples_to_qasm(native_ops, req.n_qubits)}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/run")
def run(req: RunRequest):
    """Run real OpenQASM on dense_evolution's DenseSVSimulator and return
    every quantity the page displays -- statevector, probabilities,
    shot counts, and the three real Qiskit-rendered figures.

    Above dc.MPS_DENSE_CONTRACTION_LIMIT qubits with the MPS backend, no
    dense (2**n,) array can exist at all -- dispatches to
    dc.run_large_circuit_mps instead (real top-k most-probable-states via
    greedy beam search, no sampling) and returns a distinctly-shaped
    "large_scale" response instead of pretending a full statevector/
    histogram/Q-sphere exist at that size."""
    try:
        # Never QuantumCircuit.from_qasm_str -- qiskit.circuit.QuantumCircuit
        # itself segfaults on macOS regardless of how it's built (see
        # dashboard_core/engine.py's module docstring), and this endpoint's
        # repeated-same-process shape is exactly what triggers it. The
        # qubit count is read via dense_evolution's own QASMParser instead,
        # never through qiskit at all.
        import dense_evolution as _de
        req_n_qubits = _de.QASMParser().parse(req.qasm).n_qubits
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if req.backend == "mps" and req_n_qubits > dc.MPS_DENSE_CONTRACTION_LIMIT:
        try:
            large_result = dc.run_large_circuit_mps(req.qasm, k=32, seed=req.seed)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return {
            "large_scale": True,
            "n_qubits": large_result.n_qubits,
            "k_requested": large_result.k_requested,
            "top_k_states": [{"state": s, "probability": p} for s, p in large_result.top_k_states],
            "circuit_png": _figure_to_base64_png(dc.draw_circuit_figure(large_result.ops, large_result.n_qubits)),
            "backend": "mps",
            "mps_max_bond_used": large_result.mps_max_bond_used,
            "mps_memory_mb": large_result.mps_memory_mb,
            "mps_avg_jsd": large_result.mps_avg_jsd,
        }

    try:
        result = dc.run_circuit_from_qasm(
            req.qasm, n_shots=req.shots, seed=req.seed,
            noise_model=req.noise_model, noise_p=req.noise_p,
            backend=req.backend,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    statevector_rows = [
        {
            "state": format(i, f"0{result.n_qubits}b"),
            "re": float(amp.real),
            "im": float(amp.imag),
            "abs": float(abs(amp)),
            "phase": float(np.angle(amp)),
        }
        for i, amp in enumerate(result.statevector)
        if abs(amp) > 1e-10
    ]

    # Both native (dashboard_core.state_visuals, no Qiskit): measured
    # directly at the dense backend's own practical qubit ceiling (24,
    # MPS_DENSE_CONTRACTION_LIMIT), Q-sphere takes ~1.4s and Bloch ~7s --
    # slow at the extreme edge but nowhere near the qiskit.visualization
    # bottleneck this replaced (~37s/~252s at just 12 qubits), so no skip
    # threshold is needed at all; every panel works at any qubit count the
    # engine itself can produce a statevector for.
    qsphere_png = _figure_to_base64_png(dc.qsphere_figure(result.statevector))
    qsphere_skipped_reason = None
    bloch_png = _figure_to_base64_png(dc.bloch_multivector_figure(result.statevector))
    bloch_skipped_reason = None

    return {
        "large_scale": False,
        "n_qubits": result.n_qubits,
        "counts": result.counts,
        "probabilities": result.probabilities.tolist(),
        "statevector": statevector_rows,
        "circuit_png": _figure_to_base64_png(dc.draw_circuit_figure(result.ops, result.n_qubits)),
        "histogram_png": _figure_to_base64_png(dc.histogram_figure(result.counts)),
        "qsphere_png": qsphere_png,
        "qsphere_skipped_reason": qsphere_skipped_reason,
        "bloch_png": bloch_png,
        "bloch_skipped_reason": bloch_skipped_reason,
        "fidelity_vs_ideal": result.fidelity_vs_ideal,
        "backend": result.backend,
        "mps_max_bond_used": result.mps_max_bond_used,
        "mps_memory_mb": result.mps_memory_mb,
        "mps_avg_jsd": result.mps_avg_jsd,
    }


@app.get("/api/presets")
def presets():
    return dc.QASM_LIBRARY


@app.get("/api/palette")
def palette():
    return dc.GATE_PALETTE


@app.get("/api/noise_models")
def noise_models():
    """Real Kraus-channel noise models (dense_evolution.NoiseModel)."""
    import dense_evolution as _de
    return _de.NoiseModel.MODELS


@app.get("/api/system_limits")
def system_limits():
    """Real, per-machine max qubit count for a dense statevector, based on
    actual free RAM right now (dense_evolution.chunk.SafeMemoryGuard) --
    not a fixed number picked for one dev machine."""
    return dc.max_safe_dense_qubits()


@app.get("/api/hamiltonians")
def hamiltonians(mapping: str = "jordan_wigner"):
    """Every real molecular Hamiltonian in the catalog (PennyLane qchem
    Hartree-Fock, real fermion-to-qubit mapping), each annotated with its
    real qubit count -- unfiltered, so the whole catalog is always
    visible and picking a molecule is what sets the circuit's qubit
    count, not the reverse. mapping is jordan_wigner or bravyi_kitaev --
    both represent the identical physical Hamiltonian (same spectrum),
    just in a different qubit basis."""
    return dc.get_all_molecules(mapping=mapping)


class MoleculeRequest(BaseModel):
    name: str
    mapping: str = "jordan_wigner"


@app.post("/api/hamiltonian/molecule")
def hamiltonian_molecule(req: MoleculeRequest):
    """Ground-state energy of a catalog molecule -- real Hartree-Fock +
    Jordan-Wigner/Bravyi-Kitaev Hamiltonian, exact dense diagonalization."""
    try:
        H = dc.get_molecular_hamiltonian_matrix(req.name, mapping=req.mapping)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown molecule: {req.name!r}")
    spec = dc.MOLECULE_CATALOG[req.name]
    geometry = spec["geometry"]() if callable(spec["geometry"]) else spec["geometry"]
    return {
        "n_qubits": int(np.log2(H.shape[0])),
        "symbols": spec["symbols"],
        "geometry": np.asarray(geometry).tolist(),
        "charge": spec["charge"],
        "ground_state_energy_hartree": dc.ground_state_energy(H),
    }


class MixRequest(BaseModel):
    name_a: str
    name_b: str
    weight_a: float = 0.5
    weight_b: float = 0.5
    mapping: str = "jordan_wigner"


@app.post("/api/hamiltonian/mix")
def hamiltonian_mix(req: MixRequest):
    """Real weighted combination of two catalog Hamiltonians that share
    the same qubit count (same electron space) -- H_mix = weight_a*H_a +
    weight_b*H_b, a real Hermitian operator, exact-diagonalized just like
    any single-molecule entry. Mixing molecules with different qubit
    counts is physically meaningless (different Hilbert spaces), so
    that's rejected with a clear error, not silently coerced."""
    try:
        H_a = dc.get_molecular_hamiltonian_matrix(req.name_a, mapping=req.mapping)
        H_b = dc.get_molecular_hamiltonian_matrix(req.name_b, mapping=req.mapping)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"unknown molecule: {exc}")
    try:
        H_mix = dc.mix_hamiltonians(H_a, H_b, req.weight_a, req.weight_b)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {
        "n_qubits": int(np.log2(H_mix.shape[0])),
        "energy_a": dc.ground_state_energy(H_a),
        "energy_b": dc.ground_state_energy(H_b),
        "energy_mixed": dc.ground_state_energy(H_mix),
    }


class CustomMoleculeRequest(BaseModel):
    symbols: list
    geometry: list  # [[x, y, z], ...] in Angstrom
    charge: int = 0
    mapping: str = "jordan_wigner"


@app.post("/api/hamiltonian/custom")
def hamiltonian_custom(req: CustomMoleculeRequest):
    """Real Hamiltonian for an arbitrary molecule specified on the spot --
    same PennyLane Hartree-Fock pipeline as the catalog, just built from
    whatever symbols/geometry the caller provides instead of a fixed
    preset. Small molecules only (exact dense diagonalization -- this
    simulator's real limit, not an arbitrary cap): reject before
    PennyLane if the electron/orbital count would need too many qubits.
    """
    if len(req.symbols) != len(req.geometry):
        raise HTTPException(
            status_code=400,
            detail=f"{len(req.symbols)} symbols but {len(req.geometry)} geometry rows",
        )
    try:
        H, n_qubits = dc.build_molecular_hamiltonian(req.symbols, req.geometry, req.charge, mapping=req.mapping)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if n_qubits > 12:
        raise HTTPException(
            status_code=400,
            detail=f"{n_qubits} qubits needed -- too large for exact dense diagonalization here",
        )
    return {
        "n_qubits": n_qubits,
        "ground_state_energy_hartree": dc.ground_state_energy(H),
    }


class VqeRequest(BaseModel):
    # Either name (catalog molecule) or symbols+geometry (custom) must be given.
    name: str | None = None
    symbols: list | None = None
    geometry: list | None = None
    charge: int = 0
    ansatz_type: str = "hardware_efficient"  # or "uccsd"
    n_layers: int = 8
    maxiter: int = 200
    step_size: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.999
    seed: int = 0


@app.post("/api/vqe")
def vqe(req: VqeRequest):
    """Real VQE, optimized with a real Adam gradient descent
    (lightning.qubit + adjoint differentiation -- see dashboard_core.vqe)
    against the molecule's real Jordan-Wigner Hamiltonian. Not a
    fixed/precomputed circuit -- every call re-runs the optimization from
    a fresh random start. ansatz_type "hardware_efficient" is a generic
    n_layers-deep RY+CNOT template; "uccsd" is the real chemically-
    motivated ansatz built from the molecule's own fermionic single/
    double excitation operators (qml.qchem.excitations) -- fewer
    parameters, converges faster per iteration, but each iteration is
    much more expensive (the decomposed excitation circuits are far
    deeper), so it needs fewer maxiter but more wall-clock time per
    molecule at 10+ qubits."""
    active_electrons = active_orbitals = None
    if req.name:
        try:
            spec = dc.MOLECULE_CATALOG[req.name]
        except KeyError:
            raise HTTPException(status_code=404, detail=f"unknown molecule: {req.name!r}")
        symbols = spec["symbols"]
        geometry = spec["geometry"]() if callable(spec["geometry"]) else spec["geometry"]
        charge = spec["charge"]
        active_electrons = spec.get("active_electrons")
        active_orbitals = spec.get("active_orbitals")
    elif req.symbols and req.geometry:
        if len(req.symbols) != len(req.geometry):
            raise HTTPException(
                status_code=400,
                detail=f"{len(req.symbols)} symbols but {len(req.geometry)} geometry rows",
            )
        symbols, geometry, charge = req.symbols, req.geometry, req.charge
    else:
        raise HTTPException(status_code=400, detail="provide either 'name' or 'symbols'+'geometry'")

    try:
        result = dc.run_vqe(
            symbols, geometry, charge=charge, ansatz_type=req.ansatz_type,
            n_layers=req.n_layers, maxiter=req.maxiter,
            step_size=req.step_size, beta1=req.beta1, beta2=req.beta2,
            active_electrons=active_electrons, active_orbitals=active_orbitals, seed=req.seed,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return result


class QmmmForcesRequest(BaseModel):
    name: str
    mapping: str = "jordan_wigner"


@app.post("/api/qmmm_forces")
def qmmm_forces(req: QmmmForcesRequest):
    """Real Hellmann-Feynman nuclear forces (dashboard_core.qmmm) on the
    catalog molecule's real Hartree-Fock ground state -- see qmmm.py's
    module docstring for the real differentiable-Hamiltonian derivation
    (F = -d<psi|H(R)|psi>/dR via PennyLane's own autodiff, not a
    finite-difference approximation or a fabricated classical-embedding
    perturbation)."""
    try:
        result = dc.compute_hellmann_feynman_forces(req.name, mapping=req.mapping)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return result


class MdTrajectoryRequest(BaseModel):
    name: str
    n_steps: int = 20
    dt_fs: float = 0.5
    mapping: str = "jordan_wigner"
    recompute_electronic_state: bool = False


@app.post("/api/md_trajectory")
def md_trajectory(req: MdTrajectoryRequest):
    """Real Velocity-Verlet MD trajectory (dashboard_core.qmmm) driven by
    real Hellmann-Feynman forces at every step. recompute_electronic_state
    re-solves real Hartree-Fock at each step's new geometry (true ab-initio
    MD, substantially more expensive per step) instead of holding the
    initial electronic state fixed throughout (the default, real but only
    accurate close to the starting geometry -- see qmmm.run_md_trajectory's
    docstring). n_steps is capped to keep a single request bounded on
    this kernel's real, often RAM/CPU-constrained host machine."""
    if req.n_steps < 1 or req.n_steps > 200:
        raise HTTPException(status_code=400, detail="n_steps must be between 1 and 200")
    if req.recompute_electronic_state and req.n_steps > 30:
        raise HTTPException(
            status_code=400,
            detail="recompute_electronic_state re-solves real Hartree-Fock every step -- "
                   "capped at 30 steps to keep a single request's real cost bounded",
        )
    try:
        result = dc.run_md_trajectory(
            req.name, req.n_steps, dt_fs=req.dt_fs, mapping=req.mapping,
            recompute_electronic_state=req.recompute_electronic_state,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return result


class MitigateRequest(BaseModel):
    qasm: str
    pauli_string: str
    noise_model: str
    noise_p: float
    seed: int = 42
    extrapolation_method: str = "richardson"


@app.post("/api/mitigate")
def mitigate(req: MitigateRequest):
    """Real Zero-Noise Extrapolation: <pauli_string> measured on the ideal
    state and on the real noise channel at each noise scale (each an
    ensemble average over many stochastic Kraus draws), extrapolated back
    to zero noise -- Richardson (exact, through 1x/2x/3x noise_p) or a
    degree-2 least-squares polynomial fit through 5 scales (1x..5x, trades
    a little interpolation bias for averaging down statistical noise)."""
    try:
        result = dc.run_zne_mitigation(
            req.qasm, req.pauli_string, req.noise_model, req.noise_p, seed=req.seed,
            extrapolation_method=req.extrapolation_method,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {
        "n_qubits": result.n_qubits,
        "pauli_string": result.pauli_string,
        "ideal_expectation": result.ideal_expectation,
        "noise_factors": result.noise_factors,
        "noisy_expectations": result.noisy_expectations,
        "zne_extrapolated": result.zne_extrapolated,
        "extrapolation_method": result.extrapolation_method,
    }


class MitigateMatrixRequest(BaseModel):
    qasm: str
    noise_model: str
    noise_p: float
    seed: int = 42


@app.post("/api/mitigate_matrix")
def mitigate_matrix(req: MitigateMatrixRequest):
    """Real density-matrix ZNE (dense_evolution.zne_density_matrix):
    Monte-Carlo density-matrix estimate at 1x/2x/3x noise_p,
    extrapolated + projected onto the nearest physical state, graded by
    real Uhlmann fidelity against the true ideal state."""
    try:
        result = dc.run_density_matrix_zne(req.qasm, req.noise_model, req.noise_p, seed=req.seed)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {
        "n_qubits": result.n_qubits,
        "noise_factors": result.noise_factors,
        "fidelity_raw": result.fidelity_raw,
        "fidelity_corrected": result.fidelity_corrected,
    }


class CosmicRayBurstRequest(BaseModel):
    baseline_gamma: float
    times_us: Optional[List[float]] = None


@app.post("/api/cosmic_ray_burst")
def cosmic_ray_burst(req: CosmicRayBurstRequest):
    """Real cosmic-ray/gamma-ray-induced quasiparticle burst profile,
    reproducing a real measured event from arXiv:2104.05219."""
    try:
        result = dc.run_cosmic_ray_burst(req.baseline_gamma, times_us=req.times_us)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {
        "times_us": result.times_us,
        "baseline_gamma": result.baseline_gamma,
        "decay_probabilities": result.decay_probabilities,
        "peak_ratio": result.peak_ratio,
    }


class OscillatingNoiseRequest(BaseModel):
    base_p: float
    freq: float
    amp: float
    factors: Optional[List[float]] = None


@app.post("/api/oscillating_noise")
def oscillating_noise(req: OscillatingNoiseRequest):
    """Noise strength that oscillates instead of scaling smoothly with a
    ZNE-style scale factor -- for stress-testing a mitigation technique's
    smoothness assumption."""
    try:
        result = dc.run_oscillating_noise(req.base_p, req.freq, req.amp, factors=req.factors)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {
        "base_p": result.base_p,
        "freq": result.freq,
        "amp": result.amp,
        "factors": result.factors,
        "p_eff": result.p_eff,
    }


class DensityMatrixChannelRequest(BaseModel):
    qasm: str
    channel: str
    param: float


@app.post("/api/density_matrix_channel")
def density_matrix_channel(req: DensityMatrixChannelRequest):
    """Apply a density-matrix-level noise channel ('global_depolarizing'
    or 'amplitude_damping') to the ideal density matrix of a QASM
    circuit -- distinct from NoiseModel's per-qubit statevector channels."""
    try:
        result = dc.run_density_matrix_channel(req.qasm, req.channel, req.param)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {
        "n_qubits": result.n_qubits,
        "channel": result.channel,
        "param": result.param,
        "ideal_diagonal": result.ideal_diagonal,
        "noisy_diagonal": result.noisy_diagonal,
        "trace": result.trace,
    }


class VectorHealingRequest(BaseModel):
    vectors: list       # (n_steps, dim) -- list of equal-length rows
    radius_baseline: int | None = None


@app.post("/api/vector_healing")
def vector_healing(req: VectorHealingRequest):
    """Real predictive-healing pass over a noisy vector sequence
    (dashboard_core.run_vector_healing / ia_utils.vector_healing): per
    step, a Phi-Trigger decides whether the change from a local baseline
    looks like genuine dynamics (kept) or static noise (replaced by the
    local median). NaN/Inf entries are sanitized first regardless."""
    try:
        result = dc.run_vector_healing(req.vectors, radius_baseline=req.radius_baseline)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {
        "healed_vectors": result.healed_vectors,
        "fallback_triggered": result.fallback_triggered,
        "adaptive_radius_used": result.adaptive_radius_used,
        "reconstruction_error": result.reconstruction_error,
    }


class WormholeSelectInstanceRequest(BaseModel):
    n_majorana: int = 8
    k_terms: int = 10
    J: float = float(np.sqrt(2))
    n_candidates: int = 200
    target_commuting: int = 34


@app.post("/api/wormhole_select_instance")
def wormhole_select_instance(req: WormholeSelectInstanceRequest):
    """Screen `n_candidates` random binary-sparse-SYK realizations (see
    dashboard_core.wormhole) by their exact commuting/anticommuting term-
    pair count and return the seed closest to `target_commuting` --
    arXiv:2604.10090's own instance-selection criterion. A uniformly-
    random seed does not reliably show the protocol's sign-dependent
    teleportation signal; this is the real, verified fix (not a random
    seed dressed up), see /api/wormhole_teleportation."""
    if req.n_majorana < 4 or req.n_majorana % 2 != 0:
        raise HTTPException(status_code=400, detail="n_majorana must be even and >= 4")
    if req.k_terms < 1 or req.k_terms > (req.n_majorana * (req.n_majorana - 1) *
                                          (req.n_majorana - 2) * (req.n_majorana - 3)) // 24:
        raise HTTPException(status_code=400, detail=f"k_terms must be in [1, C({req.n_majorana},4)]")
    try:
        seed = dc.select_good_instance(
            req.n_majorana, req.k_terms, req.J,
            n_candidates=req.n_candidates, target_commuting=req.target_commuting,
        )
        n_qubits, terms = dc.build_sparse_syk_terms(req.n_majorana, req.k_terms, req.J, seed)
        commuting, anticommuting = dc.commuting_pair_count(terms, n_qubits)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {
        "seed": seed,
        "n_majorana": req.n_majorana,
        "k_terms": req.k_terms,
        "commuting": commuting,
        "anticommuting": anticommuting,
        "target_commuting": req.target_commuting,
    }


class WormholeTeleportationRequest(BaseModel):
    n_majorana: int = 8
    k_terms: int = 10
    J: float = float(np.sqrt(2))
    mu: float = 12.0
    t0: float = 0.3
    t1: float = 0.6
    seed: int = 61
    with_message: bool = True
    backend: str = "exact"  # or "trotter"
    n_steps_evolution: int = 8
    n_steps_coupling: int = 16


@app.post("/api/wormhole_teleportation")
def wormhole_teleportation(req: WormholeTeleportationRequest):
    """Real traversable-wormhole-inspired quantum teleportation
    (Gao-Jafferis-Wall theory, arXiv:2604.10090) on a binary sparse SYK
    model -- see dashboard_core.wormhole's module docstring for the full
    derivation. Returns the mutual information I(P:R[0]) between the
    injected-message reference qubit and a qubit read out from the R
    (right) SYK register after the full TFD-setup / evolve(t0) /
    couple(mu) / evolve(t1) protocol; the qualitative result to look for
    is that this value differs between mu>0 and mu<0 (call twice with
    opposite-sign mu and compare -- see /api/wormhole_scan-adjacent MCP
    tool for a batch version of exactly that sweep).

    backend="exact" evolves via exact matrix exponentiation
    (eigendecomposition, cheap and exact); backend="trotter" evolves via
    a real Trotterized gate circuit (dense_evolution.trotter_evolve_ops)
    run through DenseSVSimulator -- closer to what actual hardware would
    execute, verified to reproduce the exact backend's result closely at
    the known signal peak (seed 61, t0=0.3, t1=0.60)."""
    if req.n_majorana < 4 or req.n_majorana % 2 != 0:
        raise HTTPException(status_code=400, detail="n_majorana must be even and >= 4")
    n_full = req.n_majorana + 2
    limits = dc.max_safe_dense_qubits()
    if n_full > limits["max_qubits_dense"]:
        raise HTTPException(
            status_code=400,
            detail=f"n_majorana={req.n_majorana} needs {n_full} qubits total (L+R+P+Q), "
                   f"too large for exact dense simulation here (max {limits['max_qubits_dense']})",
        )
    if req.backend not in ("exact", "trotter"):
        raise HTTPException(status_code=400, detail="backend must be 'exact' or 'trotter'")

    try:
        if req.backend == "exact":
            mi = dc.run_wormhole_protocol(
                req.n_majorana, req.k_terms, req.J, req.mu, req.t0, req.t1,
                req.seed, with_message=req.with_message,
            )
        else:
            mi = dc.run_wormhole_protocol_trotter(
                req.n_majorana, req.k_terms, req.J, req.mu, req.t0, req.t1,
                req.seed, with_message=req.with_message,
                n_steps_evolution=req.n_steps_evolution, n_steps_coupling=req.n_steps_coupling,
            )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {
        "mutual_information_pt": mi,
        "backend": req.backend,
        "n_majorana": req.n_majorana,
        "k_terms": req.k_terms,
        "mu": req.mu,
        "t0": req.t0,
        "t1": req.t1,
        "seed": req.seed,
        "with_message": req.with_message,
    }


def main():
    """Console-script entry point (`dense-evolution serve`, see
    dense_evolution/cli.py) -- identical to running this file directly."""
    uvicorn.run(app, host="127.0.0.1", port=8800)


if __name__ == "__main__":
    main()
