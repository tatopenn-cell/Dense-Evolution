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
import jax.numpy as jnp

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
    valid = jnp.arange(diis_dim) >= (diis_dim - history_count)

    errs_flat = error_history.reshape(diis_dim, -1)
    B_full = errs_flat @ errs_flat.T
    mask2d = valid[:, None] & valid[None, :]
    B = jnp.where(mask2d, B_full, 0.0)
    B = jnp.where(jnp.eye(diis_dim, dtype=bool) & ~mask2d, 1.0, B)

    A = jnp.zeros((diis_dim + 1, diis_dim + 1))
    A = A.at[:diis_dim, :diis_dim].set(B)
    col = jnp.where(valid, -1.0, 0.0)
    A = A.at[:diis_dim, diis_dim].set(col)
    A = A.at[diis_dim, :diis_dim].set(col)
    b = jnp.zeros(diis_dim + 1).at[diis_dim].set(-1.0)

    solution = jnp.linalg.solve(A, b)
    is_finite = jnp.all(jnp.isfinite(solution))
    coeffs = jnp.where(valid, jnp.where(is_finite, solution[:diis_dim], 0.0), 0.0)

    F_diis = jnp.tensordot(coeffs, fock_history, axes=1)
    return jnp.where(is_finite, F_diis, fock_history[-1])


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
) -> HFResult:
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
        iteration, _P, _C, _orbital_energies, _energy_prev, converged, _fock_history, _error_history = state
        return jnp.logical_and(jnp.logical_not(converged), iteration < max_iterations)

    def body_fun(state):
        iteration, P, _C, _orbital_energies, energy_prev, _converged, fock_history, error_history = state

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

        orbital_energies, C_ortho = jnp.linalg.eigh(X.T @ F_step @ X)
        C = X @ C_ortho
        P_new = _density_from_coefficients(C, n_occupied_pairs)

        density_converged = jnp.linalg.norm(P_new - P) < convergence_tol
        energy_converged = jnp.abs(energy - energy_prev) < energy_tol
        converged = jnp.logical_and(density_converged, energy_converged)

        P_damped = jnp.where(history_count < 2, damping * P_new + (1.0 - damping) * P, P_new)
        P_next = jnp.where(converged, P_new, P_damped)

        return (iteration + 1, P_next, C, orbital_energies, energy, converged, fock_history, error_history)

    init_state = (
        jnp.array(0),
        P0,
        C0,
        orbital_energies0,
        jnp.array(jnp.inf),
        jnp.array(False),
        jnp.zeros((diis_dim, n_basis, n_basis)),
        jnp.zeros((diis_dim, n_basis, n_basis)),
    )
    iteration, P, C, orbital_energies, _energy_prev, converged, _fh, _eh = jax.lax.while_loop(cond_fun, body_fun, init_state)

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
    )


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
