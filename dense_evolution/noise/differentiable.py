from typing import Optional, List

import jax

__all__ = ["NoiseSpec"]


class NoiseSpec:
    """Native JAX-differentiable representation of a noise configuration --
    a real JAX PyTree, so noise parameters thread through jax.jit/jax.grad/
    jax.vmap natively -- e.g. as the `noise=` argument to
    circuit_to_energy_fn's energy_fn -- instead of being applied as an
    external, Python-side step around the already-traced circuit (the old
    way: build sv, exit the trace, call apply_to_sv separately).

    `model`/`qubits` are static (aux_data): they select which code path
    runs, not values to differentiate or batch over -- the same role
    `static_argnames` plays for a plain jax.jit function, but automatic
    here because it's part of the pytree structure. `p`/`jax_key` are
    pytree leaves (children): `p` can be a traced/differentiable value
    (e.g. optimizing noise strength itself), and `jax_key` flows through
    jit/vmap/scan the way any other JAX array does -- no external
    Python-level key management, no OS-entropy fallback (unlike
    apply_to_sv called standalone with jax_key=None), so a NoiseSpec's
    result is always reproducible from the key it was built with.

    `jax_key` is required (not Optional) -- the whole point of wiring
    noise into the traced computation this way is to remove the need for
    an external, ad-hoc key-management workaround; a caller who wants a
    fresh key per call should split one themselves (`jax.random.split`)
    and build a fresh NoiseSpec, the same as any other JAX-idiomatic
    stateless-key pattern.

    Examples
    --------
    >>> import jax
    >>> from dense_evolution.noise import NoiseSpec
    >>> key = jax.random.PRNGKey(0)
    >>> spec = NoiseSpec(model="depolarizing", p=0.05, jax_key=key, qubits=[0, 1])
    >>> spec
    NoiseSpec(model='depolarizing', p=0.05, qubits=(0, 1))
    """
    def __init__(self, model: str, p, jax_key, qubits: Optional[List[int]] = None):
        self.model = model
        self.p = p
        self.jax_key = jax_key
        self.qubits = tuple(qubits) if qubits is not None else None

    def __repr__(self):
        return f"NoiseSpec(model={self.model!r}, p={self.p!r}, qubits={self.qubits!r})"

    def tree_flatten(self):
        children = (self.p, self.jax_key)
        aux_data = (self.model, self.qubits)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        model, qubits = aux_data
        p, jax_key = children
        return cls(model, p, jax_key, qubits)


jax.tree_util.register_pytree_node(
    NoiseSpec, NoiseSpec.tree_flatten, NoiseSpec.tree_unflatten
)
