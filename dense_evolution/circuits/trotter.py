"""
Real-time Hamiltonian evolution as an actual gate circuit (Trotterization)
-- did not exist anywhere in this package before. Every existing piece of
"evolution" machinery here is either gate-based-and-fixed (a hand-written
or VQE-optimized circuit template) or exact-and-not-a-circuit
(dashboard_core.hamiltonians.ground_state_energy's dense diagonalization).
Nothing composed exp(-i*H*t) for an arbitrary Hamiltonian into gates a
real quantum computer could run.

Originated in research/wormhole_syk.py, where it closed an explicit,
previously-open follow-on: reproducing a traversable-wormhole-teleportation
signal (arXiv:2604.10090) first via exact matrix exponentiation (cheap,
but not what real hardware executes), then via this module's Trotterized
gate circuit -- verified the signal wasn't an artifact of the exact-
evolution shortcut, it survives with real gates too. Neither function
here is specific to that experiment or to SYK physics; both drop
straight into any future feature needing exp(-i*H*t) as gates (a
Trotterized VQE-adjacent ansatz, quench dynamics, etc.).

pauli_rotation_ops is exact for a single Pauli-string term (fidelity
1.0 against scipy.linalg.expm, verified in tests/unit/test_trotter.py for
1-4 qubit mixed X/Y/Z strings, not just Z-strings); trotter_evolve_ops
composes many such terms via the first-order product formula by default,
which is an *approximation* whose error shrinks as n_steps grows (also
verified: infidelity drops roughly 4x per doubling of steps against a
real, non-trivial multi-qubit Hamiltonian, consistent with the expected
quadratic convergence of first-order Trotter error in state overlap).

order=2 selects the second-order (Strang/symmetric) product formula
instead -- each step applies the terms forward at half the angle, then
backward (reversed order) at half the angle again:
[prod_k exp(-i*c_k*P_k*dt/2)] * [prod_k(reversed) exp(-i*c_k*P_k*dt/2)],
which cancels the first-order formula's leading error term (verified in
tests/unit/test_trotter.py: infidelity drops roughly 16x per doubling of
steps, consistent with the expected quartic convergence of second-order
Trotter error in state overlap, vs. order=1's ~4x). Costs 2x the gates
of order=1 for the same n_steps -- the standard second-order tradeoff,
worth it when n_steps would otherwise need to be large for accuracy
(e.g. the noise-robustness experiments in wormhole_syk_teleportation.py,
where gate count directly limits how much depolarizing noise the
circuit accumulates).
"""

import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm

__all__ = ['pauli_rotation_ops', 'trotter_evolve_ops', 'continuous_pulse_evolve',
           'continuous_dissipative_evolve']


def pauli_rotation_ops(pauli_dict, angle):
    """Gate-tuple circuit for exp(-i*angle*P), P a Pauli string given as
    {qubit: 'X'/'Y'/'Z'} -- basis-change + CNOT-staircase + RZ + inverse,
    the same identity already used elsewhere in this codebase
    (dashboard_core.vqe's UCCSD/QAOA-style ZZ interactions) generalized
    here to arbitrary mixed X/Y/Z strings, not just Z-strings.

    This package's rz(theta) = exp(-i*theta/2*Z) (checked directly
    against scipy.linalg.expm when this was written, not assumed from
    convention) -- rz(2*angle) on the accumulator qubit therefore gives
    exactly exp(-i*angle*Z) on the accumulated parity.

    Parameters
    ----------
    pauli_dict : dict
        {qubit: 'X'|'Y'|'Z'}. An empty dict (identity term) returns [].
    angle : float

    Returns
    -------
    list[tuple]
        Gate tuples ready for DenseSVSimulator.run_circuit /
        QASMParser-compatible circuits.
    """
    qubits = sorted(pauli_dict.keys())
    if not qubits:
        return []
    ops = []
    for q in qubits:
        letter = pauli_dict[q]
        if letter == 'X':
            ops.append(('h', q))
        elif letter == 'Y':
            ops.append(('sdg', q))
            ops.append(('h', q))
    for i in range(len(qubits) - 1):
        ops.append(('cx', qubits[i], qubits[i + 1]))
    ops.append(('rz', qubits[-1], 2 * angle))
    for i in reversed(range(len(qubits) - 1)):
        ops.append(('cx', qubits[i], qubits[i + 1]))
    for q in qubits:
        letter = pauli_dict[q]
        if letter == 'X':
            ops.append(('h', q))
        elif letter == 'Y':
            ops.append(('h', q))
            ops.append(('s', q))
    return ops


def trotter_evolve_ops(terms, t, n_steps, order=1):
    """Trotter product formula for exp(-i*H*t), H = sum_k c_k*P_k.

    order=1 (default): [prod_k exp(-i*c_k*P_k*(t/n_steps))]^n_steps.
    Term order within one step follows `terms`' own order, identical
    every repetition (not re-randomized per step).

    order=2: Strang/symmetric splitting -- each step is a forward half-
    angle pass through `terms` followed by a backward half-angle pass
    through `terms` reversed, [prod_k exp(-i*c_k*P_k*dt/2)] *
    [prod_k(reversed) exp(-i*c_k*P_k*dt/2)], repeated n_steps times.
    Quadratically more accurate than order=1 for the same n_steps (see
    module docstring), at 2x the gate count per step.

    Parameters
    ----------
    terms : list[(float, dict)]
        (coefficient, pauli_dict) pairs, e.g. from
        dense_evolution.pauli_hamiltonian_to_matrix's own term format,
        or dense_evolution.majorana_pauli_terms products.
    t : float
        Total evolution time.
    n_steps : int
        Number of Trotter steps -- higher is more accurate and more
        gates, the standard Trotter accuracy/cost tradeoff.
    order : int
        1 (default) or 2 -- see above.

    Returns
    -------
    list[tuple]
        Gate tuples for the whole Trotterized evolution.
    """
    if order not in (1, 2):
        raise ValueError(f"order must be 1 or 2, got {order}")
    dt = t / n_steps
    if order == 1:
        step_ops = []
        for c, pdict in terms:
            step_ops.extend(pauli_rotation_ops(pdict, c * dt))
    else:
        half_dt = dt / 2
        step_ops = []
        for c, pdict in terms:
            step_ops.extend(pauli_rotation_ops(pdict, c * half_dt))
        for c, pdict in reversed(terms):
            step_ops.extend(pauli_rotation_ops(pdict, c * half_dt))
    return step_ops * n_steps


def continuous_pulse_evolve(psi0, hamiltonian_fn, coeffs_t, dt, observable_fn=None):
    """Evolve a statevector under a time-dependent Hamiltonian via
    jax.lax.scan, generalized out of a pattern first written ad hoc for a
    real time-dependent pulse (Dense-Evolution-Discovery's
    germanium_iswap_validation.py, exact_final_state/exact_final_state_general
    -- a 56ns raised-cosine baseband iSWAP pulse, arXiv:2608.16716). That
    script's own Trotterized-gate-circuit version of the same pulse
    (build_pulse_circuit) instead builds a plain Python list of gate tuples,
    one exp(-i*H*dt) per slice via pauli_rotation_ops -- fine for producing a
    circuit a discrete-gate simulator can run, but not what this function is
    for: this evolves the statevector directly, slice by slice, entirely
    inside JAX, with no Python-side list that grows with the number of
    slices (the O(1)-per-step scan carry is the whole point -- many slices
    for a finely-resolved pulse cost compile time, not accumulating Python
    memory).

    Not specific to any one Hamiltonian, qubit count, or pulse shape --
    `hamiltonian_fn` supplies the (possibly qubit-count-dependent) operator
    for a given instantaneous coefficient, and `coeffs_t` can be any sampled
    time-dependent profile (a smooth pulse envelope, a sudden burst, a
    constant array for a time-independent Hamiltonian, etc.).

    Parameters
    ----------
    psi0 : array_like
        Initial statevector, shape (2**n_qubits,).
    hamiltonian_fn : callable
        coeff -> Hamiltonian matrix, shape (2**n_qubits, 2**n_qubits), for
        that instant's coefficient. Called once per entry of `coeffs_t`
        under jax.lax.scan, so it must be JAX-traceable.
    coeffs_t : array_like
        Per-slice instantaneous coefficient, one entry per time slice
        (e.g. a peak amplitude times a sampled pulse envelope). The
        evolution applies exp(-i*hamiltonian_fn(coeff)*dt) for each entry,
        in order.
    dt : float
        Duration of one slice (coeffs_t is assumed sampled on a uniform
        grid of this spacing -- same convention as the germanium
        experiment's dt=0.05 ns midpoint/linspace sampling).
    observable_fn : callable, optional
        If given, applied to the statevector after each slice; the stacked
        per-slice results are returned as `trajectory` (mirrors the
        experiment's own step_record, used there to plot |01>/|10>
        occupation probability over the pulse). If omitted, `trajectory`
        is None and only the final state is computed.

    Returns
    -------
    final_psi : jnp.ndarray
    trajectory : jnp.ndarray or None
    """
    def step(psi, coeff):
        H_t = hamiltonian_fn(coeff)
        U_step = expm(-1j * H_t * dt)
        next_psi = jnp.dot(U_step, psi)
        y = observable_fn(next_psi) if observable_fn is not None else None
        return next_psi, y

    final_psi, trajectory = jax.lax.scan(step, jnp.asarray(psi0), jnp.asarray(coeffs_t))
    return final_psi, trajectory


def continuous_dissipative_evolve(rho0, channel_fn, params_t, observable_fn=None):
    """Evolve a density matrix through a time-dependent open-system (CPTP)
    channel via jax.lax.scan -- the dissipative counterpart of
    `continuous_pulse_evolve`, which only ever does unitary exp(-i*H*dt)
    steps on a pure state.

    Needed because not every real time-dependent physical event is coherent.
    E.g. a cosmic-ray/gamma impact on a superconducting qubit chip (real
    data: McEwen et al., arXiv:2104.05219) produces a burst of quasiparticles
    that transiently collapses the chip's effective T1 -- a rise (~10us to a
    first plateau, ~1ms to near-saturation) followed by a ~25-30ms
    exponential decay back to baseline, measured directly, not modeled as a
    static before/after depolarizing parameter. That is dissipation with a
    time-varying rate, which cannot be expressed as a coefficient inside a
    Hermitian Hamiltonian and passed to `continuous_pulse_evolve` -- it has
    to act on rho through an actual CPTP map at each instant.

    `channel_fn` supplies that per-slice CPTP map (e.g.
    `dense_evolution.global_depolarizing_channel`, or any other Kraus
    channel taking a time-varying parameter), so this function is not
    specific to any one noise mechanism, exactly like `continuous_pulse_evolve`
    is not specific to any one Hamiltonian.

    Parameters
    ----------
    rho0 : array_like
        Initial density matrix, shape (dim, dim).
    channel_fn : callable
        (rho, param) -> rho_next, a single-slice CPTP map. Called once per
        entry of `params_t` under jax.lax.scan, so it must be
        JAX-traceable.
    params_t : array_like
        Per-slice instantaneous channel parameter (e.g. a depolarizing/
        decay probability sampled on a time grid reproducing a measured
        event's rise-and-decay profile).
    observable_fn : callable, optional
        If given, applied to rho after each slice; the stacked per-slice
        results are returned as `trajectory`. If omitted, `trajectory` is
        None and only the final density matrix is computed.

    Returns
    -------
    final_rho : jnp.ndarray
    trajectory : jnp.ndarray or None
    """
    def step(rho, param):
        rho_next = channel_fn(rho, param)
        y = observable_fn(rho_next) if observable_fn is not None else None
        return rho_next, y

    final_rho, trajectory = jax.lax.scan(step, jnp.asarray(rho0), jnp.asarray(params_t))
    return final_rho, trajectory
