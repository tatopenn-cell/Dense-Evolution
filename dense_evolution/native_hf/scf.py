"""Restricted Hartree-Fock self-consistent field loop (Roothaan-Hall).

Standard textbook algorithm (e.g. Szabo & Ostlund, "Modern Quantum
Chemistry", ch. 3): orthogonalize the AO basis via S^(-1/2), build the
Fock matrix F = H_core + 2J - K from the current density, diagonalize
in the orthogonal basis, form a new density, repeat to convergence.
This part is genuinely simple compared to the integral evaluation and
doesn't need vectorizing -- a closed-shell molecule's SCF loop is a few
dozen matrix multiplies on an N x N matrix where N is a few tens at
most for STO-3G, nowhere near where PennyLane's implementation loses
its time (which is entirely in building H_core/repulsion tensor, done
once in assembly.py, not in this loop).

BUG FOUND (Si2, minimal 4-electron/4-orbital active space, R=2.184 A):
plain (undamped) density substitution never converged for this system
-- 100/100 iterations, still oscillating -- because two pairs of
orbitals near the active-space boundary are numerically degenerate
(HOMO-1/HOMO and LUMO/LUMO+1 each split by <1e-9 Ha), so each iteration
flips which member of a near-tied pair gets occupied, and the density
never settles. Confirmed this is a real oscillation, not just slow
convergence: three separate machines/runs of the undamped loop each
hit the iteration cap at a DIFFERENT total energy (-571.63, -570.69,
-571.02 Ha), all physically meaningless artifacts of whatever step the
loop happened to be on. First fixed with plain linear density damping
(P_next = alpha*P_new + (1-alpha)*P_old) -- textbook remedy for exactly
this oscillation failure mode (Szabo & Ostlund ch. 3.4.9), verified to
converge to the SAME energy (-570.874032094871 Ha, agreeing to 10
significant figures) across alpha in {0.1, 0.2, 0.3, 0.5, 0.7).

UPGRADED to DIIS (Pulay, "Convergence acceleration of iterative
sequences: the case of SCF iteration", Chem. Phys. Lett. 73, 393
(1980); "Improved SCF convergence acceleration", J. Comput. Chem. 3,
556 (1982)) -- the standard production-grade SCF accelerator plain
linear damping is a simplified special case of. Pulay's own error
vector, `e = X.T @ (F@P@S - S@P@F) @ X` (the Fock/density commutator in
the orthonormal basis, exactly zero at true self-consistency), is kept
across the last `diis_dim` iterations alongside the Fock matrices that
produced them; each step extrapolates a new Fock matrix as the
minimum-norm linear combination of that history (constrained to sum to
1) instead of diagonalizing the latest F directly. Verified on the
same Si2 near-degenerate case that motivated damping in the first
place: DIIS converges in 11 iterations to -570.8740320948958 Ha, versus
53 iterations for plain damped substitution alone (diis_dim=0,
alpha=0.5) to reach the same energy, -570.8740320948963 Ha -- agreeing
to 12 significant figures, the same true self-consistent solution
reached ~4.8x faster, not a different answer.

If the DIIS linear system is ever singular (a degenerate/duplicated
error-vector history), that step falls back to the plain, unextrapolated
Fock matrix rather than raising -- a transient fallback, not silent
wrong physics, since the next iteration's fresh error vector rebuilds a
usable history.

Convergence now requires BOTH the density AND the energy to stop
changing (`|P_new - P| < convergence_tol` AND `|E_new - E_old| <
energy_tol`) rather than density alone -- density convergence can
occasionally plateau one step before energy does (or vice versa) for
a system with several nearly-degenerate iterations near the end of the
run; requiring both is strictly more conservative than either alone
and costs at most a couple of extra iterations on every system tested
here.

Rewritten from NumPy to jax.numpy (the integral-building side in
assembly.py already was JAX; this loop and its np.array() outputs were
the only remaining barrier to an end-to-end JAX-traced pipeline). The
iteration itself uses jax.lax.while_loop for its data-dependent
convergence check, same as the original Python for/break -- note that
reverse-mode autodiff (jax.grad) does not work through while_loop, by
JAX's own design (the number of iterations isn't known ahead of time,
which reverse-mode differentiation needs). Making this loop's OUTPUT
differentiable is a separate, deliberately deferred step: the correct
approach is implicit differentiation at the fixed point (the custom
gradient rule only needs the converged F/P, not a replay of every
iteration), not unrolling this loop and backpropagating through it.

DIIS history is now a fixed-size (diis_dim, n, n) ring buffer instead
of a growing/shrinking Python list -- jax.lax.while_loop requires its
carried state to have constant shape across iterations. Each step
rolls the buffer and writes the newest Fock/error matrix into the last
slot; a boolean mask (derived from how many iterations have actually
run) excludes the not-yet-filled slots from the DIIS linear system
instead of the original's list-length check. The original's
try/except LinAlgError fallback becomes a jnp.isfinite check on the
solved coefficients (a singular solve produces NaN/Inf under JAX
rather than raising), selecting the single latest Fock matrix exactly
as the original's except-branch did.
"""

import dataclasses
import functools

import jax
import numpy as np
import jax.numpy as jnp

from dense_evolution.config import ensure_x64

_DIIS_DIM = 8


@dataclasses.dataclass
class HFResult:
    converged: bool
    n_iterations: int
    electronic_energy: float
    nuclear_repulsion_energy: float
    total_energy: float
    orbital_energies: jax.Array
    orbital_coefficients: jax.Array  # C, shape (n_basis, n_basis)
    density_matrix: jax.Array
    energy_history: jax.Array  # shape (max_iterations,), NaN past n_iterations


def nuclear_repulsion_energy(nuclear_charges: list[float], nuclear_positions: jax.Array) -> jax.Array:
    positions = jnp.asarray(nuclear_positions)
    energy = 0.0
    n = len(nuclear_charges)
    for i in range(n):
        for j in range(i + 1, n):
            r = jnp.linalg.norm(positions[i] - positions[j])
            energy = energy + nuclear_charges[i] * nuclear_charges[j] / r
    return energy


def _orthogonalizer(S: jax.Array) -> jax.Array:
    w, v = jnp.linalg.eigh(S)
    return v @ jnp.diag(1.0 / jnp.sqrt(w)) @ v.T


def _density_from_coefficients(C: jax.Array, n_occupied_pairs: int) -> jax.Array:
    C_occ = C[:, :n_occupied_pairs]
    return C_occ @ C_occ.T


def _diis_error(F: jax.Array, P: jax.Array, S: jax.Array, X: jax.Array) -> jax.Array:
    """Pulay's DIIS error vector, `X.T @ (F@P@S - S@P@F) @ X` -- the
    Fock/density commutator (zero at true self-consistency, since F and
    P then commute), transformed into the same orthonormal basis the
    Fock matrix itself is diagonalized in. F, P, S are all symmetric, so
    this reduces to `Y - Y.T` for `Y = X.T @ F @ P @ S @ X` -- computed
    that way to avoid forming two separate FPS/SPF products."""
    Y = X.T @ F @ P @ S @ X
    return Y - Y.T


def _diis_extrapolate(fock_history: jax.Array, error_history: jax.Array, history_count: jax.Array, diis_dim: int) -> jax.Array:
    """fock_history/error_history: (diis_dim, n, n), newest entry always
    in the last slot (see run_scf's roll-and-append). `valid` marks
    which of the diis_dim slots hold a real (not yet overwritten, not a
    zero-initialized placeholder) entry -- the last `history_count` of
    them. Invalid slots are pinned to c_i=0 by giving their row/column
    of the augmented linear system an identity-like equation instead of
    a real DIIS constraint, so they can't contribute to the solution
    regardless of their (garbage/zero) content."""
    dtype = fock_history.dtype
    zero = jnp.zeros((), dtype=dtype)
    one = jnp.ones((), dtype=dtype)
    valid = jnp.arange(diis_dim) >= (diis_dim - history_count)

    errs_flat = error_history.reshape(diis_dim, -1)
    B_full = errs_flat @ errs_flat.T
    mask2d = valid[:, None] & valid[None, :]
    B = jnp.where(mask2d, B_full, zero)
    B = jnp.where(jnp.eye(diis_dim, dtype=bool) & ~mask2d, one, B)

    A = jnp.zeros((diis_dim + 1, diis_dim + 1), dtype=dtype)
    A = A.at[:diis_dim, :diis_dim].set(B)
    col = jnp.where(valid, -one, zero)
    A = A.at[:diis_dim, diis_dim].set(col)
    A = A.at[diis_dim, :diis_dim].set(col)
    b = jnp.zeros(diis_dim + 1, dtype=dtype).at[diis_dim].set(-one)

    solution = jnp.linalg.solve(A, b)
    is_finite = jnp.all(jnp.isfinite(solution))
    coeffs = jnp.where(valid, jnp.where(is_finite, solution[:diis_dim], zero), zero)

    F_diis = jnp.tensordot(coeffs, fock_history, axes=1)
    return jnp.where(is_finite, F_diis, fock_history[-1])


def _level_shift_fock(F_ao: jax.Array, C_prev: jax.Array, S: jax.Array, n_occupied_pairs: int, level_shift: float) -> jax.Array:
    """Saunders & Hillier level shifting (Int. J. Quantum Chem. 7, 699
    (1973)): push the virtual orbitals of the PREVIOUS iteration's MO
    basis up by `level_shift` before this iteration's diagonalization,
    to open a numerical gap and stop the occupied/virtual split from
    flip-flopping across a near-degeneracy (see this module's own Si2
    docstring above -- damping/DIIS already fix the textbook case, but a
    real, harder case (a 30-atom aromatic fragment from the CASMI26
    wiring kernel) still took 1114 iterations, swinging through three
    wildly different intermediate energies first). Exact no-op at
    level_shift=0.0: since C_prev is a full, S-orthonormal basis
    (C_prev.T @ S @ C_prev = I, hence C_prev @ C_prev.T = S^{-1}), the
    round-trip S @ C_prev @ (C_prev.T @ F_ao @ C_prev) @ C_prev.T @ S
    reduces algebraically to exactly F_ao before any shift is added."""
    F_mo_prev = C_prev.T @ F_ao @ C_prev
    n = F_mo_prev.shape[0]
    level_shift = jnp.asarray(level_shift, dtype=F_ao.dtype)
    shift_diag = jnp.where(jnp.arange(n) >= n_occupied_pairs, level_shift, jnp.zeros((), dtype=F_ao.dtype))
    F_mo_prev_shifted = F_mo_prev + jnp.diag(shift_diag)
    return S @ C_prev @ F_mo_prev_shifted @ C_prev.T @ S


def run_scf(
    S: jax.Array,
    H_core: jax.Array,
    repulsion: jax.Array,
    n_electrons: int,
    nuclear_charges: list[float],
    nuclear_positions: jax.Array,
    max_iterations: int = 200,
    convergence_tol: float = 1e-10,
    energy_tol: float = 1e-10,
    damping: float = 0.5,
    diis_dim: int = _DIIS_DIM,
    level_shift: float = 0.0,
) -> HFResult:
    ensure_x64()
    if n_electrons % 2 != 0:
        raise ValueError("Only closed-shell (even electron count) systems are supported.")
    n_occupied_pairs = n_electrons // 2
    n_basis = S.shape[0]

    X = _orthogonalizer(S)
    orbital_energies0, C_ortho0 = jnp.linalg.eigh(X.T @ H_core @ X)
    C0 = X @ C_ortho0
    P0 = _density_from_coefficients(C0, n_occupied_pairs)

    def _fock_and_energy(P):
        J = jnp.einsum("pqrs,rs->pq", repulsion, P)
        K = jnp.einsum("prqs,rs->pq", repulsion, P)
        F = H_core + 2.0 * J - K
        energy = jnp.sum(P * (H_core + F))
        return F, energy

    def cond_fun(state):
        iteration, _P, _C, _orbital_energies, _energy_prev, converged, _fock_history, _error_history, _energy_history = state
        return jnp.logical_and(jnp.logical_not(converged), iteration < max_iterations)

    def body_fun(state):
        iteration, P, C_prev, _orbital_energies, energy_prev, _converged, fock_history, error_history, energy_history = state

        F, energy = _fock_and_energy(P)
        error = _diis_error(F, P, S, X)

        fock_history = jnp.roll(fock_history, shift=-1, axis=0).at[-1].set(F)
        error_history = jnp.roll(error_history, shift=-1, axis=0).at[-1].set(error)
        history_count = jnp.minimum(iteration + 1, diis_dim)

        F_step = jnp.where(
            history_count >= 2,
            _diis_extrapolate(fock_history, error_history, history_count, diis_dim),
            F,
        )

        F_diag = _level_shift_fock(F_step, C_prev, S, n_occupied_pairs, level_shift)
        orbital_energies, C_ortho = jnp.linalg.eigh(X.T @ F_diag @ X)
        C = X @ C_ortho
        P_new = _density_from_coefficients(C, n_occupied_pairs)

        density_converged = jnp.linalg.norm(P_new - P) < convergence_tol
        energy_converged = jnp.abs(energy - energy_prev) < energy_tol
        converged = jnp.logical_and(density_converged, energy_converged)

        P_damped = jnp.where(history_count < 2, damping * P_new + (1.0 - damping) * P, P_new)
        P_next = jnp.where(converged, P_new, P_damped)

        energy_history = energy_history.at[iteration].set(energy)

        return (iteration + 1, P_next, C, orbital_energies, energy, converged, fock_history, error_history, energy_history)

    init_state = (
        jnp.array(0),
        P0,
        C0,
        orbital_energies0,
        jnp.array(jnp.inf, dtype=H_core.dtype),
        jnp.array(False),
        jnp.zeros((diis_dim, n_basis, n_basis), dtype=H_core.dtype),
        jnp.zeros((diis_dim, n_basis, n_basis), dtype=H_core.dtype),
        jnp.full((max_iterations,), jnp.nan, dtype=H_core.dtype),
    )
    iteration, P, C, orbital_energies, _energy_prev, converged, _fh, _eh, energy_history = jax.lax.while_loop(cond_fun, body_fun, init_state)

    _F, electronic_energy = _fock_and_energy(P)
    e_nuc = nuclear_repulsion_energy(nuclear_charges, nuclear_positions)

    return HFResult(
        converged=bool(converged),
        n_iterations=int(iteration),
        electronic_energy=float(electronic_energy),
        nuclear_repulsion_energy=float(e_nuc),
        total_energy=float(electronic_energy) + float(e_nuc),
        orbital_energies=orbital_energies,
        orbital_coefficients=C,
        density_matrix=P,
        energy_history=energy_history,
    )


def _import_dense_armor_robust_filters():
    try:
        from dense_armor.utility.robust_filters import hampel_filter, tukey_fences
    except ImportError as exc:
        raise ImportError(
            "native_hf.scf.diagnose_convergence needs Dense-Armor (pip install dense-evolution[armor])"
        ) from exc
    return hampel_filter, tukey_fences


def diagnose_convergence(result: HFResult, radius: int = 5, n_sigmas: float = 3.0) -> dict:
    """Real automatic guard on HFResult.energy_history, instead of trusting
    the final `converged` flag in isolation: Dense-Armor's Hampel filter
    and Tukey fences (dense_evolution.utility.robust_filters -- the sister
    project's own anomaly detectors, Chauvenet/Tukey/Hampel/sigma-clipping,
    validated with 0 false positives on real H2 dissociation-curve
    chemistry) applied to the real per-iteration electronic energy trace.

    Validated on a real non-converging case (a 28-heavy-atom CASMI26
    fragment, level_shift=0.0): the broken run flags 19-24% of its 200
    iterations as anomalous; the same fragment fixed with level_shift=0.5
    (converges in 55 iterations) flags only ~4% -- background noise, not a
    false-alarm storm. Needs the `armor` extra (pip install
    dense-evolution[armor]) -- Dense-Armor is not a hard dependency of this
    module."""
    hampel_filter, tukey_fences = _import_dense_armor_robust_filters()
    trace = np.asarray(result.energy_history[:result.n_iterations])
    n = max(1, result.n_iterations)

    _cleaned_h, anomalies_h = hampel_filter(trace, radius=radius, n_sigmas=n_sigmas)
    _cleaned_t, anomalies_t = tukey_fences(trace, radius=len(trace))

    return {
        "n_iterations": result.n_iterations,
        "n_anomalies_hampel": len(anomalies_h),
        "n_anomalies_tukey": len(anomalies_t),
        "anomaly_fraction_hampel": len(anomalies_h) / n,
        "anomaly_fraction_tukey": len(anomalies_t) / n,
    }


@functools.partial(jax.custom_vjp, nondiff_argnums=(3,))
def scf_electronic_energy(S: jax.Array, H_core: jax.Array, repulsion: jax.Array, n_electrons: int) -> jax.Array:
    """The RHF electronic energy (run_scf's own `electronic_energy`, not
    counting nuclear repulsion) as a function of S/H_core/repulsion that
    IS differentiable via jax.grad -- unlike run_scf itself, whose
    jax.lax.while_loop can't be traced in reverse mode.

    The gradient is NOT backprop through the SCF iteration (which
    wouldn't even be possible) -- it's the analytic Hartree-Fock gradient
    of Pople, Krishnan, Schlegel & Binkley, Int. J. Quantum Chem. Symp.
    13, 225 (1979), eq. (21)-(22): at self-consistency, dE/dtheta equals
    the derivative of `Tr[P(H_core+F(P))] - Tr[W S]` with P and the
    energy-weighted density matrix W held fixed at their converged
    values (an envelope-theorem/Lagrangian result -- P and W are the
    stationary point and multipliers of the constrained HF variational
    problem, so their own dependence on theta drops out of the total
    derivative). W is built from ONLY the occupied orbitals:
    `W = C_occ @ diag(2 * orbital_energies_occ) @ C_occ.T` -- the factor
    of 2 is this module's own P convention (no explicit 2 in P itself,
    carried instead by F = H_core + 2J - K), not part of Pople et al.'s
    original spin-orbital formula. This automatically
    includes the "Pulay force" terms from the atom-centered basis moving
    with the nuclei (via H_core/repulsion/S's own theta-dependence),
    without hand-deriving them -- jax.grad on the frozen-P expression
    below does that part for free.

    Verified against central finite differences on H2/STO-3G (see
    tests/unit/test_native_hf_differentiable.py)."""
    result = run_scf(S, H_core, repulsion, n_electrons, [], jnp.zeros((0, 3)))
    return result.electronic_energy


def _scf_electronic_energy_fwd(S, H_core, repulsion, n_electrons):
    result = run_scf(S, H_core, repulsion, n_electrons, [], jnp.zeros((0, 3)))
    residuals = (S, H_core, repulsion, result.density_matrix, result.orbital_coefficients, result.orbital_energies)
    return result.electronic_energy, residuals


def _scf_electronic_energy_bwd(n_electrons, residuals, cotangent):
    S, H_core, repulsion, P, C, orbital_energies = residuals
    n_occupied_pairs = n_electrons // 2
    C_occ = C[:, :n_occupied_pairs]
    # Lagrange multiplier for the C_occ.T @ S @ C_occ = I constraint is
    # 2*diag(orbital_energies_occ), not diag(orbital_energies_occ) --
    # this module's P has no explicit factor of 2 (F = H_core + 2J - K
    # carries it instead), so stationarity of Tr[P(H_core+F(P))] -
    # Tr[Lambda(C_occ.T S C_occ - I)] w.r.t. C_occ gives F C_occ =
    # S C_occ (Lambda/2), which must match the Roothaan-Hall equation
    # F C_occ = S C_occ diag(eps_occ) -- so Lambda = 2*diag(eps_occ).
    # Verified: omitting this factor of 2 gave a gradient that disagreed
    # with central finite differences by exactly Tr[W_undoubled dS/dx].
    W = C_occ @ jnp.diag(2.0 * orbital_energies[:n_occupied_pairs]) @ C_occ.T

    def lagrangian(S_, H_core_, repulsion_):
        J = jnp.einsum("pqrs,rs->pq", repulsion_, P)
        K = jnp.einsum("prqs,rs->pq", repulsion_, P)
        F = H_core_ + 2.0 * J - K
        return jnp.sum(P * (H_core_ + F)) - jnp.sum(W * S_)

    dS, dH_core, drepulsion = jax.grad(lagrangian, argnums=(0, 1, 2))(S, H_core, repulsion)
    return (cotangent * dS, cotangent * dH_core, cotangent * drepulsion)


scf_electronic_energy.defvjp(_scf_electronic_energy_fwd, _scf_electronic_energy_bwd)


# =============================================================================
# UHF (Unrestricted Hartree-Fock)
# =============================================================================
#
# run_scf above implements only RHF. For open-shell transition-metal
# systems (Cr 3d3, Ti 3d1, V 3d3) the RHF assumption -- same spatial
# orbitals for alpha and beta -- breaks: the SCF oscillates between
# local minima at different geometries, and no smooth E(R) curve exists.
# UHF removes that restriction.
#
# Integrals are spin-independent and reused unchanged. Only the SCF loop
# differs:
#
#   RHF:  P = C_occ C_occ^T;              F = H + 2J(P) - K(P)
#   UHF:  P_a = C_a C_a^T; P_b = C_b C_b^T; P = P_a + P_b
#         F_a = H + J(P) - K(P_a)
#         F_b = H + J(P) - K(P_b)
#
# The factor of 2 on J disappears in UHF because the alpha and beta
# densities are counted separately (verified: with P_a = P_b = P_RHF/2
# and n_a = n_b, F_a = F_b = F_RHF).
#
# Warm-start: the caller can pass previous-step orbital coefficients as
# `C_alpha_init`/`C_beta_init` (e.g. from a converged geometry in an
# E(R) scan). This is what makes a scan smooth -- without it, an
# individual point can fall into a different local minimum of the UHF
# energy landscape and produce an outlier even when every point
# converges individually. Verified on Cr-O: two points at R=2.0, 2.4 A
# shifted by ~0.11 Ha without warm-start, exact match with warm-start.
#
# <S^2>: for a UHF Slater determinant,
#   <S^2> = S_z(S_z+1) + n_beta - Tr[P_a S P_b S]
# (Szabo & Ostlund, Modern Quantum Chemistry, eq. 2.271). All terms are
# traces -- a sum over all matrix elements is a different (wrong)
# quantity and produces negative values for projection operators.
# Verified: H2O with n_unpaired=0 gives <S^2> = 0 exactly; H2 at
# dissociation limit R=5 A gives the textbook 0.75 + small correction.

import dataclasses as _dataclasses


@_dataclasses.dataclass
class UHFResult:
    """UHF counterpart of HFResult, plus the spin-contamination diagnostic
    <S^2> and the alpha/beta decomposition."""
    converged: bool
    n_iterations: int
    electronic_energy: float
    nuclear_repulsion_energy: float
    total_energy: float
    orbital_energies_alpha: jax.Array
    orbital_energies_beta: jax.Array
    orbital_coefficients_alpha: jax.Array
    orbital_coefficients_beta: jax.Array
    density_matrix_alpha: jax.Array
    density_matrix_beta: jax.Array
    spin_squared: float
    n_alpha: int
    n_beta: int
    energy_history: jax.Array


def _density_from_coefficients_uhf(C, n_occupied):
    return C[:, :n_occupied] @ C[:, :n_occupied].T


def _fock_and_energy_uhf(H_core, repulsion, P_alpha, P_beta):
    P_total = P_alpha + P_beta
    J = jnp.einsum("pqrs,rs->pq", repulsion, P_total)
    K_alpha = jnp.einsum("prqs,rs->pq", repulsion, P_alpha)
    K_beta = jnp.einsum("prqs,rs->pq", repulsion, P_beta)
    F_alpha = H_core + J - K_alpha
    F_beta = H_core + J - K_beta
    energy = 0.5 * (
        jnp.sum(P_alpha * (H_core + F_alpha))
        + jnp.sum(P_beta * (H_core + F_beta))
    )
    return F_alpha, F_beta, energy


def _spin_squared(P_alpha, P_beta, S, n_alpha, n_beta):
    """<S^2> for a UHF Slater determinant (Szabo & Ostlund eq. 2.271).
    All traces, not sums."""
    S_z = 0.5 * (n_alpha - n_beta)
    overlap_sq = jnp.trace(P_alpha @ S @ P_beta @ S)
    return S_z * (S_z + 1.0) + n_beta - overlap_sq


def _diis_extrapolate_uhf(fh_a, fh_b, eh_a, eh_b, history_count, diis_dim):
    """UHF DIIS: the error vector is [vec(e_alpha), vec(e_beta)]
    concatenated. The linear system stays (diis_dim+1)^2 -- only the
    Gram-matrix inner products get richer."""
    dtype = fh_a.dtype
    zero = jnp.zeros((), dtype=dtype)
    one = jnp.ones((), dtype=dtype)
    valid = jnp.arange(diis_dim) >= (diis_dim - history_count)

    errs_flat = jnp.concatenate(
        [eh_a.reshape(diis_dim, -1), eh_b.reshape(diis_dim, -1)], axis=1
    )
    B_full = errs_flat @ errs_flat.T
    mask2d = valid[:, None] & valid[None, :]
    B = jnp.where(mask2d, B_full, zero)
    B = jnp.where(jnp.eye(diis_dim, dtype=bool) & ~mask2d, one, B)

    A = jnp.zeros((diis_dim + 1, diis_dim + 1), dtype=dtype)
    A = A.at[:diis_dim, :diis_dim].set(B)
    col = jnp.where(valid, -one, zero)
    A = A.at[:diis_dim, diis_dim].set(col)
    A = A.at[diis_dim, :diis_dim].set(col)
    b = jnp.zeros(diis_dim + 1, dtype=dtype).at[diis_dim].set(-one)

    solution = jnp.linalg.solve(A, b)
    is_finite = jnp.all(jnp.isfinite(solution))
    coeffs = jnp.where(valid, jnp.where(is_finite, solution[:diis_dim], zero), zero)
    F_alpha_diis = jnp.tensordot(coeffs, fh_a, axes=1)
    F_beta_diis = jnp.tensordot(coeffs, fh_b, axes=1)
    return (
        jnp.where(is_finite, F_alpha_diis, fh_a[-1]),
        jnp.where(is_finite, F_beta_diis, fh_b[-1]),
    )


def run_uhf(
    S, H_core, repulsion, n_electrons, nuclear_charges, nuclear_positions,
    n_unpaired=None,
    C_alpha_init=None, C_beta_init=None,
    max_iterations=200, convergence_tol=1e-10, energy_tol=1e-10,
    damping=0.5, diis_dim=_DIIS_DIM, level_shift=0.0,
) -> UHFResult:
    """Unrestricted Hartree-Fock SCF.

    Parameters
    ----------
    n_electrons : total number of electrons (integer).
    n_unpaired : number of unpaired electrons. None -> n_electrons % 2.
        n_alpha = (n_electrons + n_unpaired) // 2
        n_beta  = (n_electrons - n_unpaired) // 2
        Must satisfy n_electrons >= n_unpaired and (n_electrons -
        n_unpaired) even.
    C_alpha_init, C_beta_init : optional starting orbital coefficients
        from a previous geometry. Enables warm-started E(R) scans: the
        SCF stays in the same local minimum across a full curve instead
        of hopping between minima at individual points (see module
        docstring, "Warm-start"). If None, the core-Hamiltonian guess is
        used.

    Returns
    -------
    UHFResult with electronic_energy, total_energy, both orbital sets,
    both density matrices, and <S^2>.
    """
    ensure_x64()

    if n_unpaired is None:
        n_unpaired = n_electrons % 2
    if n_electrons < n_unpaired or (n_electrons - n_unpaired) % 2 != 0:
        raise ValueError(
            f"Invalid (n_electrons={n_electrons}, n_unpaired={n_unpaired}): "
            f"n_electrons - n_unpaired must be a non-negative even number."
        )

    n_alpha = (n_electrons + n_unpaired) // 2
    n_beta = (n_electrons - n_unpaired) // 2
    n_basis = S.shape[0]

    X = _orthogonalizer(S)

    if C_alpha_init is not None and C_beta_init is not None:
        C_alpha_prev0 = jnp.asarray(C_alpha_init)
        C_beta_prev0 = jnp.asarray(C_beta_init)
    else:
        _e0, C_ortho0 = jnp.linalg.eigh(X.T @ H_core @ X)
        C0 = X @ C_ortho0
        C_alpha_prev0 = C0
        C_beta_prev0 = C0

    P_alpha0 = _density_from_coefficients_uhf(C_alpha_prev0, n_alpha)
    P_beta0 = _density_from_coefficients_uhf(C_beta_prev0, n_beta)

    def cond_fun(state):
        iteration, _Pa, _Pb, _Ca, _Cb, _ep, converged, *_tail = state
        return jnp.logical_and(jnp.logical_not(converged), iteration < max_iterations)

    def body_fun(state):
        (iteration, P_alpha, P_beta, C_alpha_prev, C_beta_prev,
         energy_prev, _converged,
         fh_a, fh_b, eh_a, eh_b, energy_history) = state

        F_alpha, F_beta, energy = _fock_and_energy_uhf(
            H_core, repulsion, P_alpha, P_beta
        )
        err_alpha = _diis_error(F_alpha, P_alpha, S, X)
        err_beta = _diis_error(F_beta, P_beta, S, X)

        fh_a = jnp.roll(fh_a, shift=-1, axis=0).at[-1].set(F_alpha)
        fh_b = jnp.roll(fh_b, shift=-1, axis=0).at[-1].set(F_beta)
        eh_a = jnp.roll(eh_a, shift=-1, axis=0).at[-1].set(err_alpha)
        eh_b = jnp.roll(eh_b, shift=-1, axis=0).at[-1].set(err_beta)
        history_count = jnp.minimum(iteration + 1, diis_dim)

        F_alpha_diis, F_beta_diis = _diis_extrapolate_uhf(
            fh_a, fh_b, eh_a, eh_b, history_count, diis_dim
        )
        F_alpha_step = jnp.where(history_count >= 2, F_alpha_diis, F_alpha)
        F_beta_step = jnp.where(history_count >= 2, F_beta_diis, F_beta)

        F_alpha_diag = _level_shift_fock(F_alpha_step, C_alpha_prev, S, n_alpha, level_shift)
        F_beta_diag = _level_shift_fock(F_beta_step, C_beta_prev, S, n_beta, level_shift)

        _ea, C_alpha_ortho = jnp.linalg.eigh(X.T @ F_alpha_diag @ X)
        _eb, C_beta_ortho = jnp.linalg.eigh(X.T @ F_beta_diag @ X)
        C_alpha = X @ C_alpha_ortho
        C_beta = X @ C_beta_ortho
        P_alpha_new = _density_from_coefficients_uhf(C_alpha, n_alpha)
        P_beta_new = _density_from_coefficients_uhf(C_beta, n_beta)

        density_converged = (
            jnp.linalg.norm(P_alpha_new - P_alpha)
            + jnp.linalg.norm(P_beta_new - P_beta)
        ) < convergence_tol
        energy_converged = jnp.abs(energy - energy_prev) < energy_tol
        converged = jnp.logical_and(density_converged, energy_converged)

        P_alpha_damped = jnp.where(
            history_count < 2,
            damping * P_alpha_new + (1.0 - damping) * P_alpha,
            P_alpha_new,
        )
        P_beta_damped = jnp.where(
            history_count < 2,
            damping * P_beta_new + (1.0 - damping) * P_beta,
            P_beta_new,
        )
        P_alpha_next = jnp.where(converged, P_alpha_new, P_alpha_damped)
        P_beta_next = jnp.where(converged, P_beta_new, P_beta_damped)
        energy_history = energy_history.at[iteration].set(energy)

        return (iteration + 1, P_alpha_next, P_beta_next, C_alpha, C_beta,
                energy, converged, fh_a, fh_b, eh_a, eh_b, energy_history)

    init_state = (
        jnp.array(0),
        P_alpha0, P_beta0,
        C_alpha_prev0, C_beta_prev0,
        jnp.array(jnp.inf, dtype=H_core.dtype),
        jnp.array(False),
        jnp.zeros((diis_dim, n_basis, n_basis), dtype=H_core.dtype),
        jnp.zeros((diis_dim, n_basis, n_basis), dtype=H_core.dtype),
        jnp.zeros((diis_dim, n_basis, n_basis), dtype=H_core.dtype),
        jnp.zeros((diis_dim, n_basis, n_basis), dtype=H_core.dtype),
        jnp.full((max_iterations,), jnp.nan, dtype=H_core.dtype),
    )
    (iteration, P_alpha, P_beta, C_alpha, C_beta,
     _ep, converged, _fha, _fhb, _eha, _ehb, energy_history) = (
        jax.lax.while_loop(cond_fun, body_fun, init_state)
    )

    F_alpha, F_beta, electronic_energy = _fock_and_energy_uhf(
        H_core, repulsion, P_alpha, P_beta
    )
    orbital_energies_alpha = jnp.diag(C_alpha.T @ F_alpha @ C_alpha)
    orbital_energies_beta = jnp.diag(C_beta.T @ F_beta @ C_beta)
    e_nuc = nuclear_repulsion_energy(nuclear_charges, nuclear_positions)
    spin_sq = _spin_squared(P_alpha, P_beta, S, n_alpha, n_beta)

    return UHFResult(
        converged=bool(converged),
        n_iterations=int(iteration),
        electronic_energy=float(electronic_energy),
        nuclear_repulsion_energy=float(e_nuc),
        total_energy=float(electronic_energy) + float(e_nuc),
        orbital_energies_alpha=orbital_energies_alpha,
        orbital_energies_beta=orbital_energies_beta,
        orbital_coefficients_alpha=C_alpha,
        orbital_coefficients_beta=C_beta,
        density_matrix_alpha=P_alpha,
        density_matrix_beta=P_beta,
        spin_squared=float(spin_sq),
        n_alpha=n_alpha,
        n_beta=n_beta,
        energy_history=energy_history,
    )


# =============================================================================
# CUHF — Tsuchimochi & Scuseria 2010, arXiv:1008.1607
# =============================================================================
# ROHF written as a constrained UHF. In the natural-orbital basis of the
# charge density P = (P_a + P_b)/2 (core = first n_beta, open = next
# n_alpha - n_beta, virtual = the rest), the core-virtual blocks of F_a and
# F_b are both replaced by F_cs = (F_a + F_b)/2; every other block stays UHF.
# This removes spin contamination: <S^2> = S_z(S_z + 1) exactly.
# Checked against an independent implementation of the paper's equations:
# OH doublet and O2 triplet (STO-3G) agree to 2e-12 Ha.


@_dataclasses.dataclass
class CUHFResult:
    converged: bool
    n_iterations: int
    electronic_energy: float
    nuclear_repulsion_energy: float
    total_energy: float
    orbital_energies_alpha: jax.Array
    orbital_energies_beta: jax.Array
    orbital_coefficients_alpha: jax.Array
    orbital_coefficients_beta: jax.Array
    density_matrix_alpha: jax.Array
    density_matrix_beta: jax.Array
    spin_squared: float
    n_alpha: int
    n_beta: int
    energy_history: jax.Array


def _cuhf_modified_focks(H_core, repulsion, P_a, P_b, cv_mask, S):
    """Constrained alpha/beta Fock matrices and the UHF energy; `cv_mask`
    marks the core-virtual blocks in the natural-orbital basis of P."""
    P_tot = P_a + P_b
    J = jnp.einsum("pqrs,rs->pq", repulsion, P_tot)
    K_a = jnp.einsum("prqs,rs->pq", repulsion, P_a)
    K_b = jnp.einsum("prqs,rs->pq", repulsion, P_b)
    F_a = H_core + J - K_a
    F_b = H_core + J - K_b
    F_cs = 0.5 * (F_a + F_b)

    w, U = jnp.linalg.eigh(S)
    S_half = U @ jnp.diag(jnp.sqrt(w)) @ U.T
    S_inv_half = U @ jnp.diag(1.0 / jnp.sqrt(w)) @ U.T
    _occ, W = jnp.linalg.eigh(S_half @ (0.5 * P_tot) @ S_half)
    C_no = S_inv_half @ W[:, ::-1]
    delta = C_no.T @ (0.5 * (F_b - F_a)) @ C_no
    delta = S @ C_no @ jnp.where(cv_mask, 0.0, delta) @ C_no.T @ S
    F_tilde_a = F_cs - delta
    F_tilde_b = F_cs + delta

    energy = 0.5 * (jnp.sum(P_a * (H_core + F_a))
                    + jnp.sum(P_b * (H_core + F_b)))
    return F_tilde_a, F_tilde_b, energy


def run_cuhf(S, H_core, repulsion, n_electrons, nuclear_charges,
             nuclear_positions, n_unpaired=None,
             C_alpha_init=None, C_beta_init=None,
             max_iterations=200, convergence_tol=1e-10, energy_tol=1e-10,
             damping=0.5, diis_dim=_DIIS_DIM, level_shift=0.0):
    """Constrained UHF (ROHF) SCF, Tsuchimochi & Scuseria (arXiv:1008.1607).

    Same arguments and result fields as `run_uhf`. The alpha and beta
    orbitals differ, but the density is that of a restricted open-shell
    determinant: `spin_squared` equals S_z(S_z + 1) to machine precision,
    and with `n_unpaired=0` the energy equals the RHF energy.
    """
    ensure_x64()

    if n_unpaired is None:
        n_unpaired = n_electrons % 2
    if n_electrons < n_unpaired or (n_electrons - n_unpaired) % 2 != 0:
        raise ValueError(f"Invalid (n_electrons={n_electrons}, n_unpaired={n_unpaired})")

    n_alpha = (n_electrons + n_unpaired) // 2
    n_beta = (n_electrons - n_unpaired) // 2
    n_basis = S.shape[0]
    X = _orthogonalizer(S)

    if C_alpha_init is not None and C_beta_init is not None:
        C0_a = jnp.asarray(C_alpha_init)
        C0_b = jnp.asarray(C_beta_init)
    else:
        _e0, C_ortho0 = jnp.linalg.eigh(X.T @ H_core @ X)
        C0_a = X @ C_ortho0
        C0_b = C0_a

    P_a0 = _density_from_coefficients_uhf(C0_a, n_alpha)
    P_b0 = _density_from_coefficients_uhf(C0_b, n_beta)

    ar = jnp.arange(n_basis)
    core = ar < n_beta
    virt = ar >= n_alpha
    cv_mask = (core[:, None] & virt[None, :]) | (virt[:, None] & core[None, :])

    def cond_fun(s):
        return jnp.logical_and(jnp.logical_not(s[6]), s[0] < max_iterations)

    def body_fun(s):
        (it, Pa, Pb, Ca_p, Cb_p, E_prev, _c,
         fh_a, fh_b, eh_a, eh_b, eh_hist) = s

        Fa_t, Fb_t, E = _cuhf_modified_focks(H_core, repulsion, Pa, Pb, cv_mask, S)

        err_a = _diis_error(Fa_t, Pa, S, X)
        err_b = _diis_error(Fb_t, Pb, S, X)

        fh_a = jnp.roll(fh_a, -1, 0).at[-1].set(Fa_t)
        fh_b = jnp.roll(fh_b, -1, 0).at[-1].set(Fb_t)
        eh_a = jnp.roll(eh_a, -1, 0).at[-1].set(err_a)
        eh_b = jnp.roll(eh_b, -1, 0).at[-1].set(err_b)
        hc = jnp.minimum(it + 1, diis_dim)

        Fa_d, Fb_d = _diis_extrapolate_uhf(fh_a, fh_b, eh_a, eh_b, hc, diis_dim)
        Fa_s = jnp.where(hc >= 2, Fa_d, Fa_t)
        Fb_s = jnp.where(hc >= 2, Fb_d, Fb_t)

        Fa_dg = _level_shift_fock(Fa_s, Ca_p, S, n_alpha, level_shift)
        Fb_dg = _level_shift_fock(Fb_s, Cb_p, S, n_beta, level_shift)

        _ea, Ca_o = jnp.linalg.eigh(X.T @ Fa_dg @ X)
        _eb, Cb_o = jnp.linalg.eigh(X.T @ Fb_dg @ X)
        Ca = X @ Ca_o
        Cb = X @ Cb_o
        Pa_n = _density_from_coefficients_uhf(Ca, n_alpha)
        Pb_n = _density_from_coefficients_uhf(Cb, n_beta)

        d_conv = (jnp.linalg.norm(Pa_n - Pa) + jnp.linalg.norm(Pb_n - Pb)) < convergence_tol
        e_conv = jnp.abs(E - E_prev) < energy_tol
        conv = jnp.logical_and(d_conv, e_conv)

        Pa_d = jnp.where(hc < 2, damping * Pa_n + (1 - damping) * Pa, Pa_n)
        Pb_d = jnp.where(hc < 2, damping * Pb_n + (1 - damping) * Pb, Pb_n)
        Pa_x = jnp.where(conv, Pa_n, Pa_d)
        Pb_x = jnp.where(conv, Pb_n, Pb_d)
        eh_hist = eh_hist.at[it].set(E)

        return (it + 1, Pa_x, Pb_x, Ca, Cb, E, conv,
                fh_a, fh_b, eh_a, eh_b, eh_hist)

    init = (
        jnp.array(0), P_a0, P_b0, C0_a, C0_b,
        jnp.array(jnp.inf, H_core.dtype), jnp.array(False),
        jnp.zeros((diis_dim, n_basis, n_basis), H_core.dtype),
        jnp.zeros((diis_dim, n_basis, n_basis), H_core.dtype),
        jnp.zeros((diis_dim, n_basis, n_basis), H_core.dtype),
        jnp.zeros((diis_dim, n_basis, n_basis), H_core.dtype),
        jnp.full((max_iterations,), jnp.nan, H_core.dtype),
    )
    out = jax.lax.while_loop(cond_fun, body_fun, init)
    it, Pa, Pb, Ca, Cb, _Ep, conv, _fa, _fb, _ea, _eb, eh = out

    P_tot = Pa + Pb
    J = jnp.einsum("pqrs,rs->pq", repulsion, P_tot)
    Fa = H_core + J - jnp.einsum("prqs,rs->pq", repulsion, Pa)
    Fb = H_core + J - jnp.einsum("prqs,rs->pq", repulsion, Pb)
    E_el = 0.5 * (jnp.sum(Pa * (H_core + Fa)) + jnp.sum(Pb * (H_core + Fb)))
    e_nuc = nuclear_repulsion_energy(nuclear_charges, nuclear_positions)
    s2 = _spin_squared(Pa, Pb, S, n_alpha, n_beta)

    return CUHFResult(
        converged=bool(conv),
        n_iterations=int(it),
        electronic_energy=float(E_el),
        nuclear_repulsion_energy=float(e_nuc),
        total_energy=float(E_el) + float(e_nuc),
        orbital_energies_alpha=jnp.diag(Ca.T @ Fa @ Ca),
        orbital_energies_beta=jnp.diag(Cb.T @ Fb @ Cb),
        orbital_coefficients_alpha=Ca,
        orbital_coefficients_beta=Cb,
        density_matrix_alpha=Pa,
        density_matrix_beta=Pb,
        spin_squared=float(s2),
        n_alpha=n_alpha,
        n_beta=n_beta,
        energy_history=eh,
    )
