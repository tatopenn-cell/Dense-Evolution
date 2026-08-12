"""Primitive Gaussian shells and the Gaussian product theorem.

A primitive Gaussian of angular momentum degree L centered at R with
exponent a is, in 3D:

    G(r) = (x-Rx)^lx (y-Ry)^ly (z-Rz)^lz * exp(-a |r-R|^2)

We only ever need the *maximum* degree L for a shell (s: L=0, p: L=1,
...) because the recursions below build every (lx,ly,lz) with
lx+ly+lz <= L in one shot, so a shell is fully described by (L, a, R).
"""

import dataclasses

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class


@register_pytree_node_class
@dataclasses.dataclass
class GaussianShell1D:
    """A single Cartesian component (x, y, or z) of a Gaussian shell."""

    degree: int
    exponent: jax.Array  # shape ()
    center: jax.Array  # shape ()

    def tree_flatten(self):
        return (self.exponent, self.center), self.degree

    @classmethod
    def tree_unflatten(cls, degree, children):
        exponent, center = children
        return cls(degree=degree, exponent=exponent, center=center)


@register_pytree_node_class
@dataclasses.dataclass
class GaussianShell3D:
    """A 3D Gaussian shell: one exponent/center, all (lx,ly,lz) with
    lx+ly+lz <= degree implicitly represented."""

    degree: int
    exponent: jax.Array  # shape ()
    center: jax.Array  # shape (3,)

    def tree_flatten(self):
        return (self.exponent, self.center), self.degree

    @classmethod
    def tree_unflatten(cls, degree, children):
        exponent, center = children
        return cls(degree=degree, exponent=exponent, center=center)

    def component(self, axis: int) -> GaussianShell1D:
        return GaussianShell1D(
            degree=self.degree,
            exponent=self.exponent,
            center=jnp.asarray(self.center)[axis],
        )


def product_center(g1: GaussianShell3D, g2: GaussianShell3D) -> jax.Array:
    """The center P of the Gaussian obtained by multiplying two Gaussians
    (Gaussian product theorem): P = (a*A + b*B) / (a+b)."""
    a, b = jnp.asarray(g1.exponent), jnp.asarray(g2.exponent)
    A, B = jnp.asarray(g1.center), jnp.asarray(g2.center)
    return (a * A + b * B) / (a + b)


def product_prefactor(g1: GaussianShell3D, g2: GaussianShell3D) -> jax.Array:
    """The scalar prefactor K = exp(-mu |A-B|^2), mu = a*b/(a+b), that the
    product of two Gaussians picks up (everything else about the product
    is folded into the recursions below)."""
    a, b = jnp.asarray(g1.exponent), jnp.asarray(g2.exponent)
    diff = jnp.asarray(g1.center) - jnp.asarray(g2.center)
    mu = (a * b) / (a + b)
    return jnp.exp(-mu * jnp.dot(diff, diff))
