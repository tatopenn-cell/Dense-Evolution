"""Identity channel -- no noise. K0 = I."""

__all__ = ["apply"]


def apply(sv_out, idx_0, idx_1, p, rng, key, is_jax):
    """No-op: returns `(sv_out, key)` unchanged. Kept as its own channel
    (rather than a special case elsewhere) so `NoiseModel.MODELS` has one
    real implementation per entry, `'ideal'` included."""
    return sv_out, key
