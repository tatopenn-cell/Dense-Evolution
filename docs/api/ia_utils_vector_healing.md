# IA Utils — Vector Sequence Healing

Standalone module (`ia_utils/`, a separate top-level package alongside
`dense_evolution/`) for cleaning sequences of vectors — e.g. hidden
states, embeddings, VQE/MD telemetry — that may contain `NaN` or `Inf`
entries, or spurious spikes that look like noise rather than genuine
signal. `median_healing` and `enhanced_dense_healing_hybrid` both
preprocess the input (Inf → NaN → column-mean imputation) before
healing, so corrupted values never propagate into the output.

::: ia_utils.vector_healing

---

**See also**: [`dense_evolution.healing`](healing.md) for the predictive
"Phi-Trigger" primitives `enhanced_dense_healing_hybrid` calls internally
to decide, per step, whether an observed change looks like genuine
dynamics (kept as-is) or static noise (replaced with the local median).
[`ia_utils.adversarial_vector_attack`](ia_utils_adversarial_vector_attack.md)
for a gradient-based robustness test of that same Phi-Trigger decision.
