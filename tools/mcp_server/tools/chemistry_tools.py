"""Tools: molecular Hamiltonians, VQE, QM/MM forces, and molecular
dynamics."""
import asyncio
import json

from ..client import _request, catch_errors
from ..config import COMPUTE
from ..models import (
    CustomMoleculeInput, EnergyScanInput, MdTrajectoryInput, MixMoleculesInput,
    MoleculeEnergyInput, QmmmForcesInput, RunVqeInput,
)
from ..molecules import _resolve_molecule_name
from ..server import mcp


@mcp.tool(name="dense_evolution_molecule_energy", annotations={"title": "Get catalog molecule ground-state energy", **COMPUTE})
@catch_errors
async def dense_evolution_molecule_energy(params: MoleculeEnergyInput) -> str:
    """Compute the exact ground-state energy of a catalog molecule via real
    Hartree-Fock + Jordan-Wigner/Bravyi-Kitaev Hamiltonian construction and
    exact dense diagonalization.

    Args:
        params (MoleculeEnergyInput): name (short id or full catalog name), mapping.

    Returns:
        str: JSON with n_qubits, symbols, geometry, charge, ground_state_energy_hartree.
        "Error: ..." with a 404-style message if `name` is not in the catalog
        (call dense_evolution_list_molecules to see valid names/ids).
    """
    resolved = await _resolve_molecule_name(params.name)
    payload = {**params.model_dump(), "name": resolved}
    return json.dumps(await _request("POST", "/api/hamiltonian/molecule", timeout=60.0, json=payload), indent=2)


@mcp.tool(name="dense_evolution_mix_molecules", annotations={"title": "Mix two catalog Hamiltonians", **COMPUTE})
@catch_errors
async def dense_evolution_mix_molecules(params: MixMoleculesInput) -> str:
    """Compute H_mix = weight_a*H_a + weight_b*H_b for two catalog molecules
    that share the same qubit count (same electron space), and diagonalize
    all three (H_a, H_b, H_mix) for their ground-state energies. Mixing
    molecules with different qubit counts is physically meaningless and is
    rejected with a clear error.

    Args:
        params (MixMoleculesInput): name_a, name_b (short id or full catalog name), weight_a, weight_b, mapping.

    Returns:
        str: JSON with n_qubits, energy_a, energy_b, energy_mixed (all in Hartree).
    """
    name_a = await _resolve_molecule_name(params.name_a)
    name_b = await _resolve_molecule_name(params.name_b)
    payload = {**params.model_dump(), "name_a": name_a, "name_b": name_b}
    return json.dumps(await _request("POST", "/api/hamiltonian/mix", timeout=60.0, json=payload), indent=2)


@mcp.tool(name="dense_evolution_custom_molecule_energy", annotations={"title": "Get custom molecule ground-state energy", **COMPUTE})
@catch_errors
async def dense_evolution_custom_molecule_energy(params: CustomMoleculeInput) -> str:
    """Compute the ground-state energy of an arbitrary molecule (not in the
    catalog) from its atomic symbols and geometry, via the same Hartree-Fock
    pipeline as the catalog. Small molecules only -- exact dense
    diagonalization caps out at 12 qubits, rejected before PennyLane runs if
    the electron/orbital count would exceed that.

    Args:
        params (CustomMoleculeInput): symbols, geometry, charge, mapping.
            len(symbols) must equal len(geometry).

    Returns:
        str: JSON with n_qubits, ground_state_energy_hartree, or "Error: ..."
        if the molecule needs more than 12 qubits.
    """
    return json.dumps(await _request("POST", "/api/hamiltonian/custom", timeout=60.0, json=params.model_dump()), indent=2)


@mcp.tool(name="dense_evolution_energy_scan", annotations={"title": "Scan ground-state energy over several geometries", **COMPUTE})
@catch_errors
async def dense_evolution_energy_scan(params: EnergyScanInput) -> str:
    """Compute the ground-state energy at each of several geometries in one
    call -- e.g. a bond-length dissociation curve or a bond-angle sweep --
    instead of calling dense_evolution_custom_molecule_energy once per
    point. Points are evaluated concurrently against the kernel. A point
    that fails (e.g. too many qubits for exact diagonalization) is reported
    with its own error and does not abort the rest of the scan.

    Args:
        params (EnergyScanInput): symbols, geometries (list of points, max 50),
            charge, mapping, labels (optional, one per point).

    Returns:
        str: JSON with:
        {
            "n_points": int,
            "results": [{"label": ..., "n_qubits": int, "ground_state_energy_hartree": float} | {"label": ..., "error": str}, ...],
            "minimum": {"label": ..., "ground_state_energy_hartree": float} | null  # over successful points only
        }
    """
    if params.labels is not None and len(params.labels) != len(params.geometries):
        raise ValueError(f"{len(params.labels)} labels but {len(params.geometries)} geometries -- must match.")
    labels = params.labels if params.labels is not None else list(range(len(params.geometries)))

    async def _one_point(label, geometry):
        if len(params.symbols) != len(geometry):
            return {"label": label, "error": f"{len(params.symbols)} symbols but {len(geometry)} geometry rows"}
        try:
            data = await _request(
                "POST", "/api/hamiltonian/custom", timeout=60.0,
                json={"symbols": params.symbols, "geometry": geometry, "charge": params.charge, "mapping": params.mapping},
            )
            return {"label": label, "n_qubits": data["n_qubits"], "ground_state_energy_hartree": data["ground_state_energy_hartree"]}
        except Exception as e:
            return {"label": label, "error": str(e)}

    results = await asyncio.gather(*(_one_point(l, g) for l, g in zip(labels, params.geometries)))
    successful = [r for r in results if "ground_state_energy_hartree" in r]
    minimum = min(successful, key=lambda r: r["ground_state_energy_hartree"]) if successful else None
    return json.dumps({"n_points": len(results), "results": results, "minimum": minimum}, indent=2)


@mcp.tool(name="dense_evolution_run_vqe", annotations={"title": "Run VQE ground-state optimization", **COMPUTE})
@catch_errors
async def dense_evolution_run_vqe(params: RunVqeInput) -> str:
    """Run real VQE (Adam gradient descent with adjoint differentiation)
    against a molecule's Jordan-Wigner Hamiltonian, from a fresh random
    start every call -- not a cached/precomputed result. Can take a while
    for 'uccsd' ansatz or high maxiter; consider dense_evolution_health's
    RAM figures and start with a small maxiter/n_layers to gauge cost first.

    Args:
        params (RunVqeInput): either `name` (short id or full catalog name)
            or `symbols`+`geometry` (custom) must be given, plus
            ansatz_type, n_layers, maxiter, step_size, beta1, beta2, seed.

    Returns:
        str: JSON with the optimized energy, convergence history, and
        optimized parameters. "Error: ..." if neither name nor
        symbols+geometry is provided, or the molecule is unknown.
    """
    payload = params.model_dump()
    if params.name:
        payload["name"] = await _resolve_molecule_name(params.name)
    return json.dumps(await _request("POST", "/api/vqe", timeout=600.0, json=payload), indent=2)


@mcp.tool(name="dense_evolution_qmmm_forces", annotations={"title": "Compute Hellmann-Feynman nuclear forces", **COMPUTE})
@catch_errors
async def dense_evolution_qmmm_forces(params: QmmmForcesInput) -> str:
    """Compute real Hellmann-Feynman nuclear forces (F = -d<psi|H(R)|psi>/dR
    via PennyLane autodiff, not finite differences) on a catalog molecule's
    real Hartree-Fock ground state.

    Args:
        params (QmmmForcesInput): name (short id or full catalog name), mapping.

    Returns:
        str: JSON with per-atom force vectors and related energetics.
    """
    resolved = await _resolve_molecule_name(params.name)
    payload = {**params.model_dump(), "name": resolved}
    return json.dumps(await _request("POST", "/api/qmmm_forces", timeout=120.0, json=payload), indent=2)


@mcp.tool(name="dense_evolution_md_trajectory", annotations={"title": "Run a molecular dynamics trajectory", **COMPUTE})
@catch_errors
async def dense_evolution_md_trajectory(params: MdTrajectoryInput) -> str:
    """Run a real Velocity-Verlet MD trajectory driven by Hellmann-Feynman
    forces at every step, for a catalog molecule.

    Args:
        params (MdTrajectoryInput): name (short id or full catalog name),
            n_steps, dt_fs, mapping, recompute_electronic_state (true = true
            ab-initio MD, capped at 30 steps; false = fixed electronic
            state, capped at 200 steps).

    Returns:
        str: JSON with the trajectory (positions/energies per step).
        "Error: ..." if n_steps is out of range for the chosen mode.
    """
    resolved = await _resolve_molecule_name(params.name)
    payload = {**params.model_dump(), "name": resolved}
    return json.dumps(await _request("POST", "/api/md_trajectory", timeout=600.0, json=payload), indent=2)
