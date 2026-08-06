# Traversable-wormhole-inspired quantum teleportation (SYK model)

Companion write-up for `wormhole_syk.py`. See that file's own module
docstring for the full technical account; this is the short version.

## Why this exists

An old, discarded `dashboard_core` circuit called "Traversable Wormhole
(BGQ)" used real vocabulary (SYK scrambling, a phase "kick") but wasn't
real physics: run on a single qubit register, forward scrambling then a
bare `RZ` "kick" then backward scrambling. Verified directly (not
assumed): applying that kick with either sign gave *identical* results.
That's not a tuning problem — a single-register readout structurally
cannot show sign-dependent behavior, because a qubit inside a Bell pair
has a maximally-mixed marginal that no local operation changes (the
no-signaling theorem forbids it outright, regardless of circuit design).

## The real recipe

Found in **arXiv:2604.10090**, "Quantum simulation of
traversable-wormhole-inspired quantum teleportation in a chaotic binary
sparse SYK model" (2026) — a real hardware reproduction of the
Gao-Jafferis-Wall traversable-wormhole teleportation protocol. Key
ingredients missing from the old circuit, all implemented here:

1. **Two coupled systems**, not one — an L and R copy of a chaotic
   Hamiltonian (here: a binary sparse N=8 Sachdev-Ye-Kitaev model, K=10
   of the C(8,4)=70 possible four-Majorana coupling terms, random
   coefficients `+-J/sqrt(K)`).
2. **A message injected via a separate reference pair** (P, Q Bell
   pair, Q swapped into the L register) — not a bare bit-flip on a qubit
   that's already maximally entangled (which, again, no-signaling makes
   pointless).
3. **A real bilinear L-R coupling** `exp(i*mu*V)`,
   `V = (1/4N) * sum_j chi_L^j chi_R^j` — not a same-register global phase.
4. **The right readout**: mutual information between the reference qubit
   P and a qubit read out from R, not a single-qubit expectation value.
   This needs a real partial trace over multiple qubits and von Neumann
   entropy — neither existed anywhere in this codebase before this file.

## What was verified, not assumed

- Majorana Jordan-Wigner mapping: `{chi_a, chi_b} = 2*delta_ab*I` checked
  exactly (error 0.0, not "small") for every pair.
- The sparse SYK Hamiltonian: exactly Hermitian (error 0.0). An earlier
  draft had an extra factor of `i` on each term, on the mistaken
  assumption that a 4-Majorana product is anti-Hermitian — it's actually
  Hermitian (reversing 4 anticommuting factors is an even number of
  transpositions), and the bug showed up immediately as a large,
  unambiguous Hermiticity error once checked.
- Partial trace / mutual information: reproduces the exact textbook value
  `I = 2*ln(2)` for a plain Bell pair and for one qubit of a GHZ state.

## The central, honest finding

The sign-dependent signal is **real but realization-dependent**. A
uniformly-random choice of which K=10 terms to keep does not reliably
show a clean signature — tried across several seeds, results ranged from
a fairly clean positive trend, to a signal with the *opposite* sign for
most of a sweep, to a small case that stayed roughly flat.

This matches the paper directly: they didn't use an arbitrary random
instance either. They picked one "selected for favorable commutation
properties" — specifically, among the `C(10,2)=45` pairs of their chosen
terms, 34 commute and 11 anticommute. `select_good_instance()` in
`wormhole_syk.py` reproduces that same selection criterion (screen many
random instances by their *exact* commuting-pair count, keep the one
closest to the paper's ratio) instead of trusting one arbitrary seed.
Screened across 200 candidates for N=8, **seed 61 matched exactly**
(34 commuting / 11 anticommuting) — and running it through the full
protocol gave the cleanest result of every seed tried: a single smooth
peak in the sign-dependent mutual-information difference, positive and
consistent across ten straight sweep points, before crossing back near
the edge of the sweep. See `wormhole_syk.py --help`'s module docstring
and its `__main__` output for the actual numbers.

## Deliberate simplification

Real-time evolution here is exact matrix exponentiation applied directly
to the statevector (`scipy`/`numpy` eigendecomposition), not a
Trotterized gate circuit. The paper needed gates because that's what
real IBM hardware executes; we have exact statevector access, and the
paper's own hardware run is validated against exactly this kind of exact
baseline. A gate-circuit (Trotterized) version — closer to what real
hardware would run, and a candidate for a future Composer/MCP feature —
is an explicit, separate follow-on, not attempted here.

## Status

This is a research reproduction living in `research/`, not a shipped
package feature. Nothing in `dense_evolution`, `dashboard_core`, the
Composer, or the MCP server was touched. If this line of work continues,
the most obviously reusable pieces are the Majorana JW mapping and the
partial-trace/von-Neumann-entropy/mutual-information utilities — neither
existed in the package before, and both are generically useful beyond
this one experiment.
