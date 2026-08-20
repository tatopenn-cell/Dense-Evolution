"""Tools: traversable-wormhole-inspired quantum teleportation (SYK
model)."""
import json

from ..client import _request, catch_errors
from ..config import COMPUTE
from ..models import WormholeScanInput, WormholeSelectInstanceInput, WormholeTeleportationInput
from ..server import mcp


@mcp.tool(name="dense_evolution_wormhole_select_instance",
          annotations={"title": "Select a good SYK instance for wormhole teleportation", **COMPUTE})
@catch_errors
async def dense_evolution_wormhole_select_instance(params: WormholeSelectInstanceInput) -> str:
    """Find a binary-sparse-SYK random-instance seed suitable for the
    traversable-wormhole-inspired teleportation protocol
    (`dense_evolution_wormhole_teleportation`).

    A uniformly-random seed does NOT reliably show the protocol's
    sign-dependent teleportation signal -- verified directly across many
    seeds (some give a clean peak, some the wrong sign for most of a
    sweep, some are flat noise). arXiv:2604.10090 didn't use an arbitrary
    instance either: they picked one "selected for favorable commutation
    properties" among their chosen terms. This tool reproduces that same
    selection criterion -- screening `n_candidates` seeds by their exact
    commuting/anticommuting term-pair count and returning the one closest
    to `target_commuting` -- rather than trusting a random seed.

    Always call this before `dense_evolution_wormhole_teleportation` /
    `dense_evolution_wormhole_scan` unless you already have a known-good
    seed (e.g. 61, for the defaults n_majorana=8/k_terms=10/target=34,
    the exact match found and used throughout this project's own
    verification -- see research/wormhole_syk.md).

    Args:
        params (WormholeSelectInstanceInput): n_majorana, k_terms, J,
            n_candidates, target_commuting.

    Returns:
        str: JSON {seed, n_majorana, k_terms, commuting, anticommuting,
        target_commuting} -- pass `seed` straight into
        dense_evolution_wormhole_teleportation / _wormhole_scan.
    """
    return json.dumps(await _request("POST", "/api/wormhole_select_instance", timeout=60.0, json=params.model_dump()), indent=2)


@mcp.tool(name="dense_evolution_wormhole_teleportation",
          annotations={"title": "Run traversable-wormhole-inspired quantum teleportation", **COMPUTE})
@catch_errors
async def dense_evolution_wormhole_teleportation(params: WormholeTeleportationInput) -> str:
    """Run one point of the real traversable-wormhole-inspired quantum
    teleportation protocol (Gao-Jafferis-Wall theory, arXiv:2604.10090)
    on a binary sparse SYK model: two coupled chaotic Hamiltonians (L,R),
    a message injected into L via a reference-qubit pair (P,Q), a real
    bilinear L-R coupling exp(i*mu*V), and a readout that is NOT a
    single-qubit expectation value (which the no-signaling theorem
    forbids from ever showing this signal) but the mutual information
    between the reference qubit P and a qubit read out from R.

    Returns a single mutual-information value for the given mu -- the
    physically meaningful result is the *difference* between a positive-
    and negative-mu run at the same (n_majorana, k_terms, seed, t0, t1).
    Call this tool twice with opposite-sign mu, or use
    dense_evolution_wormhole_scan to sweep many (t1, mu) combinations in
    one batched call.

    Requires a well-selected seed (see
    dense_evolution_wormhole_select_instance) -- an arbitrary random seed
    will likely not show a clean signal.

    Args:
        params (WormholeTeleportationInput): n_majorana, k_terms, J, mu,
            t0, t1, seed, with_message, backend, n_steps_evolution,
            n_steps_coupling.

    Returns:
        str: JSON {mutual_information_pt, backend, n_majorana, k_terms,
        mu, t0, t1, seed, with_message}.
    """
    return json.dumps(await _request("POST", "/api/wormhole_teleportation", timeout=120.0, json=params.model_dump()), indent=2)


@mcp.tool(name="dense_evolution_wormhole_scan",
          annotations={"title": "Sweep the wormhole teleportation signal over t1", **COMPUTE})
@catch_errors
async def dense_evolution_wormhole_scan(params: WormholeScanInput) -> str:
    """Sweep `t1_values` for the traversable-wormhole-inspired
    teleportation protocol, running both +mu_magnitude and -mu_magnitude
    at every point -- one call instead of 2*len(t1_values) separate
    dense_evolution_wormhole_teleportation calls. Returns each point's
    mutual information for both signs plus their difference (mu<0 minus
    mu>0 in this project's own convention), the standard readout for the
    protocol's qualitative signature: a smooth peak in that difference
    across the sweep (known peak for seed=61/n_majorana=8/k_terms=10/
    t0=0.3: around t1≈0.6-0.7).

    Unlike dense_evolution_energy_scan, points run sequentially, not
    concurrently: each single teleportation call does real exact
    diagonalization (or Trotterized circuit execution) of the full joint
    L+R+P+Q system and takes several seconds on its own (verified:
    concurrent calls to this specific endpoint crashed the kernel process
    outright -- a real BLAS/eigh thread-safety issue under this protocol's
    heavier-than-usual concurrent linear algebra, not present in the
    lighter Hamiltonian-diagonalization calls energy_scan batches). A
    full 20-point sweep can take several minutes; start with fewer points
    to gauge cost.

    A point that fails does not abort the rest of the sweep; its error is
    reported alongside the successful points.

    Args:
        params (WormholeScanInput): n_majorana, k_terms, J, mu_magnitude,
            t0, t1_values (list, max 20 points), seed, with_message,
            backend, n_steps_evolution, n_steps_coupling.

    Returns:
        str: JSON with:
        {
            "n_points": int,
            "results": [{"t1": float, "mu_positive": float, "mu_negative": float, "delta": float} |
                        {"t1": float, "error": str}, ...],
            "peak": {"t1": ..., "delta": ...} | null  # point with the largest delta, successful points only
        }
    """
    async def _one_point(t1):
        base = dict(
            n_majorana=params.n_majorana, k_terms=params.k_terms, J=params.J, t0=params.t0, t1=t1,
            seed=params.seed, with_message=params.with_message, backend=params.backend,
            n_steps_evolution=params.n_steps_evolution, n_steps_coupling=params.n_steps_coupling,
        )
        try:
            pos = await _request("POST", "/api/wormhole_teleportation", timeout=120.0, json={**base, "mu": params.mu_magnitude})
            neg = await _request("POST", "/api/wormhole_teleportation", timeout=120.0, json={**base, "mu": -params.mu_magnitude})
            i_pos, i_neg = pos["mutual_information_pt"], neg["mutual_information_pt"]
            return {"t1": t1, "mu_positive": i_pos, "mu_negative": i_neg, "delta": i_neg - i_pos}
        except Exception as e:
            return {"t1": t1, "error": str(e)}

    results = [await _one_point(t1) for t1 in params.t1_values]
    successful = [r for r in results if "delta" in r]
    peak = max(successful, key=lambda r: r["delta"]) if successful else None
    return json.dumps({"n_points": len(results), "results": results, "peak": peak}, indent=2)
