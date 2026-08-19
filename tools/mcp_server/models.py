"""Pydantic input schemas for every dense_evolution_mcp tool, extracted
from server.py so tool logic and schema definitions can change
independently. Field descriptions stay here (not moved into tool
docstrings) since MCP clients read them directly off the schema to build
their own UI/validation -- the docstrings still carry the longer
explanation for a human or agent reading the tool's full documentation.
"""
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class ListMoleculesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mapping: str = Field(
        default="jordan_wigner",
        description="Fermion-to-qubit mapping: 'jordan_wigner' or 'bravyi_kitaev'. Both represent "
        "the identical physical Hamiltonian (same spectrum) in a different qubit basis.",
    )


class BuildCircuitInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    n_qubits: int = Field(..., ge=1, description="Number of qubits in the circuit.")
    ops: list = Field(
        ..., description="List of graphical-builder gate operations to convert into OpenQASM. "
        "Get valid gate names from dense_evolution_list_gates first."
    )


class RunCircuitInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    qasm: str = Field(..., min_length=1, description="OpenQASM 2.0 circuit source to simulate.")
    shots: int = Field(default=1000, ge=1, le=1_000_000, description="Number of measurement shots for the counts histogram.")
    seed: int = Field(default=42, description="Random seed for sampling/noise reproducibility.")
    noise_model: str = Field(default="ideal", description="Noise model name from dense_evolution_list_noise_models, or 'ideal'.")
    noise_p: float = Field(default=0.0, ge=0.0, le=1.0, description="Noise channel error probability (ignored if noise_model='ideal').")
    backend: str = Field(default="dense", description="'dense' (exact statevector, up to the safe qubit ceiling) or 'mps' "
                         "(matrix-product-state, approximate top-k states, for larger circuits).")
    include_visualizations: bool = Field(
        default=False,
        description="If true, also render and save circuit/histogram/Q-sphere/Bloch PNGs to disk and "
        "return their file paths. Leave false unless you actually need to view an image -- "
        "rendering costs extra time and the paths are not useful without viewing them.",
    )


class MoleculeEnergyInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(
        ..., description="Catalog molecule -- short id (e.g. 'H2', 'LiH', 'HeH+') or the full catalog "
        "name. See dense_evolution_list_molecules."
    )
    mapping: str = Field(default="jordan_wigner", description="'jordan_wigner' or 'bravyi_kitaev'.")


class MixMoleculesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name_a: str = Field(..., description="First catalog molecule -- short id or full catalog name.")
    name_b: str = Field(..., description="Second catalog molecule -- short id or full catalog name. "
                         "Must have the same qubit count as name_a.")
    weight_a: float = Field(default=0.5, description="Weight of the first Hamiltonian in the mix.")
    weight_b: float = Field(default=0.5, description="Weight of the second Hamiltonian in the mix.")
    mapping: str = Field(default="jordan_wigner", description="'jordan_wigner' or 'bravyi_kitaev'.")


class CustomMoleculeInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    symbols: list = Field(..., min_length=1, description="Atomic symbols, e.g. ['H', 'H', 'O'].")
    geometry: list = Field(..., min_length=1, description="[[x, y, z], ...] coordinates in Angstrom, one row per symbol.")
    charge: int = Field(default=0, description="Total molecular charge.")
    mapping: str = Field(default="jordan_wigner", description="'jordan_wigner' or 'bravyi_kitaev'.")


class EnergyScanInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    symbols: list = Field(..., min_length=1, description="Atomic symbols shared by every point in the scan, e.g. ['H', 'H'].")
    geometries: list = Field(
        ..., min_length=1, max_length=50,
        description="List of [[x,y,z], ...] geometries (Angstrom) to evaluate, one per scan point -- "
        "e.g. a bond-length or angle sweep. Each geometry must have the same number of rows as `symbols`. "
        "Capped at 50 points per call to keep one request's kernel load bounded.",
    )
    charge: int = Field(default=0, description="Molecular charge, shared by every point.")
    mapping: str = Field(default="jordan_wigner", description="'jordan_wigner' or 'bravyi_kitaev'.")
    labels: Optional[list] = Field(
        default=None,
        description="Optional label per point (e.g. bond lengths in Angstrom: [0.4, 0.5, ...]) shown "
        "alongside each result. Defaults to the point's index (0, 1, 2, ...) if omitted. Must be the "
        "same length as `geometries` if given.",
    )


class RunVqeInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Optional[str] = Field(default=None, description="Catalog molecule -- short id or full catalog name. "
                                 "Provide this OR symbols+geometry, not both.")
    symbols: Optional[list] = Field(default=None, description="Atomic symbols for a custom molecule.")
    geometry: Optional[list] = Field(default=None, description="[[x, y, z], ...] in Angstrom for a custom molecule.")
    charge: int = Field(default=0, description="Molecular charge (custom molecule only).")
    ansatz_type: str = Field(
        default="hardware_efficient",
        description="'hardware_efficient' (generic n_layers-deep RY+CNOT template) or 'uccsd' "
        "(chemically-motivated, fewer parameters but much deeper circuits per iteration).",
    )
    n_layers: int = Field(default=8, ge=1, description="Ansatz depth (hardware_efficient only).")
    maxiter: int = Field(default=200, ge=1, le=5000, description="Maximum Adam optimizer iterations.")
    step_size: float = Field(default=0.1, gt=0, description="Adam optimizer learning rate.")
    beta1: float = Field(default=0.9, description="Adam optimizer beta1.")
    beta2: float = Field(default=0.999, description="Adam optimizer beta2.")
    seed: int = Field(default=0, description="Random seed for the initial variational parameters.")


class QmmmForcesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(..., description="Catalog molecule -- short id or full catalog name. See dense_evolution_list_molecules.")
    mapping: str = Field(default="jordan_wigner", description="'jordan_wigner' or 'bravyi_kitaev'.")


class MdTrajectoryInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(..., description="Catalog molecule -- short id or full catalog name. See dense_evolution_list_molecules.")
    n_steps: int = Field(default=20, ge=1, le=200, description="Number of MD steps (capped at 200 to bound request cost).")
    dt_fs: float = Field(default=0.5, gt=0, description="Timestep in femtoseconds.")
    mapping: str = Field(default="jordan_wigner", description="'jordan_wigner' or 'bravyi_kitaev'.")
    recompute_electronic_state: bool = Field(
        default=False,
        description="If true, re-solves real Hartree-Fock at every step (true ab-initio MD, much more "
        "expensive; n_steps capped at 30 in that case). If false, holds the initial "
        "electronic state fixed throughout (accurate only close to the starting geometry).",
    )


class MitigateZneInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    qasm: str = Field(..., min_length=1, description="OpenQASM circuit to run under noise and mitigate.")
    pauli_string: str = Field(..., description="Pauli observable to measure and mitigate, e.g. 'ZZI'.")
    noise_model: str = Field(..., description="Noise model name -- see dense_evolution_list_noise_models.")
    noise_p: float = Field(..., ge=0.0, le=1.0, description="Base noise channel error probability.")
    seed: int = Field(default=42, description="Random seed for the stochastic Kraus draws.")
    extrapolation_method: str = Field(
        default="richardson",
        description="'richardson' (exact, through 1x/2x/3x noise_p) or 'polynomial' "
        "(degree-2 least-squares fit through 5 noise scales).",
    )


class MitigateDensityMatrixInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    qasm: str = Field(..., min_length=1, description="OpenQASM circuit to run under noise and mitigate.")
    noise_model: str = Field(..., description="Noise model name -- see dense_evolution_list_noise_models.")
    noise_p: float = Field(..., ge=0.0, le=1.0, description="Base noise channel error probability.")
    seed: int = Field(default=42, description="Random seed for the Monte-Carlo density-matrix estimate.")


class VectorHealingInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    vectors: list = Field(..., description="(n_steps, dim) sequence of equal-length numeric rows to heal -- "
                           "e.g. a VQE parameter/energy trajectory, MD telemetry, or any other noisy vector "
                           "sequence. NaN/Inf entries are sanitized automatically.")
    radius_baseline: int | None = Field(
        default=None,
        description="Fixed radius for the local baseline window used to judge each step. If omitted, "
        "computed adaptively as min(20, max(3, n_steps // 3)).",
    )


class WormholeSelectInstanceInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    n_majorana: int = Field(default=8, ge=4, description="Number of Majorana modes per side (must be even). "
                             "8 matches arXiv:2604.10090's own experiment (4 qubits/side).")
    k_terms: int = Field(default=10, ge=1, description="Number of sparse 4-Majorana coupling terms to keep, "
                          "out of C(n_majorana,4) possible -- 10 matches the paper's K=10.")
    J: float = Field(default=1.4142135623730951, description="SYK coupling strength (paper's J=sqrt(2)).")
    n_candidates: int = Field(default=200, ge=1, le=2000, description="Number of random seeds to screen.")
    target_commuting: int = Field(default=34, description="Target commuting-pair count out of C(k_terms,2) pairs "
                                   "-- 34 (out of 45) is the paper's own selected instance's ratio for K=10.")


class WormholeTeleportationInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    n_majorana: int = Field(default=8, ge=4, description="Number of Majorana modes per side (must be even).")
    k_terms: int = Field(default=10, ge=1, description="Number of sparse 4-Majorana coupling terms per side.")
    J: float = Field(default=1.4142135623730951, description="SYK coupling strength (paper's J=sqrt(2)).")
    mu: float = Field(default=12.0, description="L-R coupling strength exp(i*mu*V). Sign matters -- the "
                       "protocol's signature is the difference in teleported mutual information between "
                       "+mu and -mu (call twice, or use dense_evolution_wormhole_scan for a batch sweep).")
    t0: float = Field(default=0.3, description="Pre-coupling evolution time under H_L+H_R.")
    t1: float = Field(default=0.6, description="Post-coupling evolution time under H_L+H_R. The known signal "
                       "peak for seed=61/n_majorana=8/k_terms=10 is at t1=0.60.")
    seed: int = Field(default=61, description="SYK instance seed -- use dense_evolution_wormhole_select_instance "
                       "to find a good one for other (n_majorana, k_terms) combinations; 61 is the verified "
                       "match for the defaults.")
    with_message: bool = Field(default=True, description="Inject the reference qubit's message into the L "
                                "register (swap). False = no message injected, a baseline/control run.")
    backend: str = Field(default="exact", description="'exact' (matrix exponentiation, cheap, exact) or "
                          "'trotter' (real Trotterized gate circuit, closer to actual hardware execution, "
                          "verified to reproduce the exact backend's result closely).")
    n_steps_evolution: int = Field(default=8, ge=1, description="Trotter steps for the t0/t1 evolution (trotter backend only).")
    n_steps_coupling: int = Field(default=16, ge=1, description="Trotter steps for the mu coupling (trotter backend only).")


class WormholeScanInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    n_majorana: int = Field(default=8, ge=4, description="Number of Majorana modes per side (must be even).")
    k_terms: int = Field(default=10, ge=1, description="Number of sparse 4-Majorana coupling terms per side.")
    J: float = Field(default=1.4142135623730951, description="SYK coupling strength (paper's J=sqrt(2)).")
    mu_magnitude: float = Field(default=12.0, ge=0, description="Both +mu_magnitude and -mu_magnitude are "
                                 "run at every t1 point -- this is the wormhole signal's defining comparison.")
    t0: float = Field(default=0.3, description="Pre-coupling evolution time under H_L+H_R, shared by every point.")
    t1_values: list = Field(..., min_length=1, max_length=20, description="List of post-coupling evolution "
                             "times to sweep. Capped at 20 points (each point runs backend twice, +mu and "
                             "-mu, sequentially -- unlike dense_evolution_energy_scan this is NOT cheap: "
                             "each single teleportation call takes several seconds, so a full sweep can "
                             "take a few minutes; see the tool's own docstring).")
    seed: int = Field(default=61, description="SYK instance seed -- use dense_evolution_wormhole_select_instance first.")
    with_message: bool = Field(default=True, description="Inject the reference qubit's message into L.")
    backend: str = Field(default="exact", description="'exact' or 'trotter' -- see dense_evolution_wormhole_teleportation.")
    n_steps_evolution: int = Field(default=8, ge=1, description="Trotter steps for t0/t1 evolution (trotter backend only).")
    n_steps_coupling: int = Field(default=16, ge=1, description="Trotter steps for the mu coupling (trotter backend only).")
