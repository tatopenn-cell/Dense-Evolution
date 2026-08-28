# Concepts: Noise, Mitigation, and Healing

One decision primitive, `evaluate_phi_trigger` — is this step's change genuine
dynamics, or noise? — lives in `dense_evolution.mitigation.healing` and gets
reused by two different modules for two different kinds of data. That reuse,
not a family of similar-sounding names, is the real reason "healing" and
"mitigation" are easy to mix up. This page exists to make the difference
un-ambiguous.

| | Corrupts / heals what | Object it acts on | Module |
|---|---|---|---|
| **Noise** | Corrupts a real computation | A statevector, via a stochastic Kraus channel | [Noise](api/noise.md) |
| **Mitigation** | Un-corrupts a noisy result | An expectation value / density matrix, extrapolated across noise strengths (Zero-Noise Extrapolation) | [Mitigation](api/mitigation.md) |
| **Vector Healing** | Un-corrupts one bad reading in a sequence | Any `(n_steps, dim)` numeric sequence — a VQE/MD log, an embedding stream. Nothing quantum-specific. | [IA Utils — Vector Healing](api/ia_utils_vector_healing.md) |

[Healing](api/healing.md) (`dense_evolution.mitigation.healing`, still
importable as `dense_evolution.healing` via a compatibility shim after an
internal move) is the shared primitive layer the last two both call into — not
a fourth thing to run on its own.

## The actual code-level connection

`dense_evolution.mitigation.zne`'s own "healing-adapted" extrapolation branch
imports `calculate_delta_preemp` from `.healing` and uses it directly on the
noise-strength extrapolation curve. `ia_utils.vector_healing.enhanced_dense_healing_hybrid`'s
default `trigger_mode='phi'` imports `evaluate_phi_trigger` — the *same*
function — from `dense_evolution.mitigation.healing`, and applies it to
whatever plain vector sequence was passed in. Same decision primitive, two
call sites, two unrelated kinds of data: a quantum-noise-strength curve on one
side, an arbitrary numeric log on the other.

## Which page do I actually want?

- "I ran a noisy circuit and want the *ideal* result back" → [Mitigation](api/mitigation.md).
- "I have a log/trajectory/embedding sequence and one entry looks wrong" →
  [IA Utils — Vector Healing](api/ia_utils_vector_healing.md).
- "I want to add noise to a circuit on purpose" → [Noise](api/noise.md).
