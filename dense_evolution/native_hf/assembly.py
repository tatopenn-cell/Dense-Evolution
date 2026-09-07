"""Assembling full molecular AO integral matrices from contracted shells.

Each pair (or quartet, for repulsion) of shells contributes a block to
the overall S/T/V matrices (or ERI tensor). We loop over shell
pairs/quartets in plain Python -- for a minimal basis like STO-3G there
are only a handful of shells per atom (Si2/STO-3G: 6 shells total), so
this loop is cheap. The sum over primitives *within* a shell pair/quartet
and the slicing down to physical Cartesian components both happen
inside a single jax.jit-compiled call per shell pair/quartet: doing
that slicing eagerly (one jnp.ndarray.__getitem__ per primitive
combination) turned out to cost as much per-call dispatch overhead as
PennyLane's own scalar-at-a-time autograd loop, defeating the purpose,
so everything from "loop over primitives" to "slice out the physical
components" is fused into one compiled program per shell pair/quartet
and only that program's *output* touches eager Python.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np

from dense_evolution.native_hf.basis import ContractedShell, n_cartesian_functions
from dense_evolution.native_hf.cartesian import cartesian_powers
from dense_evolution.native_hf.gaussians import GaussianShell3D
from dense_evolution.native_hf.overlap import overlap_3d
from dense_evolution.native_hf.kinetic import kinetic_3d
from dense_evolution.native_hf.coulomb import nuclear_attraction, electron_repulsion


@functools.partial(jax.jit, static_argnames=("degree_a", "degree_b", "primitive_fn"))
def _pair_block(exponents_a, coeffs_a, center_a, degree_a, exponents_b, coeffs_b, center_b, degree_b, primitive_fn, extra=()):
    """extra: additional *traced* arrays (e.g. nuclear charges/positions)
    forwarded to primitive_fn unchanged -- traced (not static) so that
    calling this with a different molecule geometry reuses the same
    compiled program instead of retracing. primitive_fn must itself be a
    stable, module-level (not freshly-created-per-call) callable, since
    it IS a static jit key."""
    powers_a = jnp.asarray(cartesian_powers(degree_a))
    powers_b = jnp.asarray(cartesian_powers(degree_b))

    def one_primitive_pair(ea, ca, eb, cb):
        ga = GaussianShell3D(degree=degree_a, exponent=ea, center=center_a)
        gb = GaussianShell3D(degree=degree_b, exponent=eb, center=center_b)
        full = primitive_fn(ga, gb, *extra)
        sliced = full[powers_a[:, 0], powers_a[:, 1], powers_a[:, 2]][:, powers_b[:, 0], powers_b[:, 1], powers_b[:, 2]]
        return ca * cb * sliced

    batched = jax.vmap(
        jax.vmap(one_primitive_pair, in_axes=(None, None, 0, 0)),
        in_axes=(0, 0, None, None),
    )(exponents_a, coeffs_a, exponents_b, coeffs_b)
    return jnp.sum(batched, axis=(0, 1))


@functools.partial(jax.jit, static_argnames=("degrees", "primitive_fn"))
def _quartet_block(exponents, coeffs, centers, degrees, primitive_fn):
    """exponents/coeffs: tuples of 4 arrays (one per shell), centers: tuple
    of 4 (3,) arrays, degrees: tuple of 4 static ints."""
    powers = [jnp.asarray(cartesian_powers(d)) for d in degrees]

    def one_quartet(ea, ca, eb, cb, ec, cc, ed, cd):
        ga = GaussianShell3D(degree=degrees[0], exponent=ea, center=centers[0])
        gb = GaussianShell3D(degree=degrees[1], exponent=eb, center=centers[1])
        gc = GaussianShell3D(degree=degrees[2], exponent=ec, center=centers[2])
        gd = GaussianShell3D(degree=degrees[3], exponent=ed, center=centers[3])
        full = primitive_fn(ga, gb, gc, gd)
        sliced = full[powers[0][:, 0], powers[0][:, 1], powers[0][:, 2]]
        sliced = sliced[:, powers[1][:, 0], powers[1][:, 1], powers[1][:, 2]]
        sliced = sliced[:, :, powers[2][:, 0], powers[2][:, 1], powers[2][:, 2]]
        sliced = sliced[:, :, :, powers[3][:, 0], powers[3][:, 1], powers[3][:, 2]]
        return ca * cb * cc * cd * sliced

    vmapped = one_quartet
    # vmap outward-in: d, c, b, a -- each adds a leading batch axis.
    vmapped = jax.vmap(vmapped, in_axes=(None, None, None, None, None, None, 0, 0))
    vmapped = jax.vmap(vmapped, in_axes=(None, None, None, None, 0, 0, None, None))
    vmapped = jax.vmap(vmapped, in_axes=(None, None, 0, 0, None, None, None, None))
    vmapped = jax.vmap(vmapped, in_axes=(0, 0, None, None, None, None, None, None))

    ea, ca, eb, cb, ec, cc, ed, cd = (
        exponents[0], coeffs[0], exponents[1], coeffs[1],
        exponents[2], coeffs[2], exponents[3], coeffs[3],
    )
    batched = vmapped(ea, ca, eb, cb, ec, cc, ed, cd)
    return jnp.sum(batched, axis=(0, 1, 2, 3))


def _shell_offsets(shells: list[ContractedShell]) -> list[int]:
    offsets, running = [], 0
    for s in shells:
        offsets.append(running)
        running += len(cartesian_powers(s.degree))
    return offsets


def _overlap_primitive(ga, gb):
    return overlap_3d(ga, gb)


def build_overlap_matrix(shells: list[ContractedShell]) -> np.ndarray:
    return _build_two_index(shells, _overlap_primitive)


def _core_hamiltonian_primitive(ga, gb, charges, positions):
    T = kinetic_3d(ga, gb)
    per_nucleus = jax.vmap(lambda z, r: -z * nuclear_attraction(ga, gb, r))(charges, positions)
    V = jnp.sum(per_nucleus, axis=0)
    return -0.5 * T + V


def build_core_hamiltonian(shells: list[ContractedShell], nuclear_charges: list[float], nuclear_positions: np.ndarray) -> np.ndarray:
    charges = jnp.asarray(nuclear_charges, dtype=jnp.float64)
    positions = jnp.asarray(nuclear_positions, dtype=jnp.float64)
    return _build_two_index(shells, _core_hamiltonian_primitive, extra=(charges, positions))


def _build_two_index(shells: list[ContractedShell], primitive_fn, extra=()) -> np.ndarray:
    n = n_cartesian_functions(shells)
    offsets = _shell_offsets(shells)
    M = np.zeros((n, n))
    for i, sa in enumerate(shells):
        for j, sb in enumerate(shells):
            block = np.array(
                _pair_block(
                    sa.exponents, sa.coefficients, sa.center, sa.degree,
                    sb.exponents, sb.coefficients, sb.center, sb.degree,
                    primitive_fn, extra=extra,
                )
            )
            oi, oj = offsets[i], offsets[j]
            M[oi : oi + block.shape[0], oj : oj + block.shape[1]] = block
    return M


def _schwarz_bound(sa: ContractedShell, sb: ContractedShell) -> float:
    block = np.array(
        _quartet_block(
            (sa.exponents, sb.exponents, sa.exponents, sb.exponents),
            (sa.coefficients, sb.coefficients, sa.coefficients, sb.coefficients),
            (sa.center, sb.center, sa.center, sb.center),
            (sa.degree, sb.degree, sa.degree, sb.degree),
            electron_repulsion,
        )
    )
    na, nb = block.shape[0], block.shape[1]
    diag = np.array([[block[p, q, p, q] for q in range(nb)] for p in range(na)])
    return float(np.sqrt(np.max(np.abs(diag))))


def _shell_pair_schwarz_bounds(shells: list[ContractedShell]) -> dict:
    bounds = {}
    for i, sa in enumerate(shells):
        for j in range(i + 1):
            sb = shells[j]
            q = _schwarz_bound(sa, sb)
            bounds[(i, j)] = q
            bounds[(j, i)] = q
    return bounds


def build_repulsion_tensor(shells: list[ContractedShell], screening_tol: float = 1e-12) -> np.ndarray:
    n = n_cartesian_functions(shells)
    offsets = _shell_offsets(shells)
    V = np.zeros((n, n, n, n))
    schwarz = _shell_pair_schwarz_bounds(shells)
    for i, sa in enumerate(shells):
        for j in range(i + 1):
            sb = shells[j]
            ij_index = i * (i + 1) // 2 + j
            for k, sc in enumerate(shells):
                for l in range(k + 1):
                    sd = shells[l]
                    kl_index = k * (k + 1) // 2 + l
                    if ij_index < kl_index:
                        continue
                    if schwarz[(i, j)] * schwarz[(k, l)] < screening_tol:
                        continue
                    block = np.array(
                        _quartet_block(
                            (sa.exponents, sb.exponents, sc.exponents, sd.exponents),
                            (sa.coefficients, sb.coefficients, sc.coefficients, sd.coefficients),
                            (sa.center, sb.center, sc.center, sd.center),
                            (sa.degree, sb.degree, sc.degree, sd.degree),
                            electron_repulsion,
                        )
                    )
                    oi, oj, ok, ol = offsets[i], offsets[j], offsets[k], offsets[l]
                    for pi, pj, pk, pl, b in (
                        (oi, oj, ok, ol, block),
                        (oj, oi, ok, ol, block.transpose(1, 0, 2, 3)),
                        (oi, oj, ol, ok, block.transpose(0, 1, 3, 2)),
                        (oj, oi, ol, ok, block.transpose(1, 0, 3, 2)),
                        (ok, ol, oi, oj, block.transpose(2, 3, 0, 1)),
                        (ol, ok, oi, oj, block.transpose(3, 2, 0, 1)),
                        (ok, ol, oj, oi, block.transpose(2, 3, 1, 0)),
                        (ol, ok, oj, oi, block.transpose(3, 2, 1, 0)),
                    ):
                        V[pi : pi + b.shape[0], pj : pj + b.shape[1], pk : pk + b.shape[2], pl : pl + b.shape[3]] = b
    return V
