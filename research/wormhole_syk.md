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

## Two evolution backends — and the real gate circuit survives the check

`run_wormhole_protocol` uses exact matrix exponentiation applied directly
to the statevector (`scipy`/`numpy` eigendecomposition) — the paper
needed gates because that's what real IBM hardware executes, but we have
exact statevector access, and the paper's own hardware run is validated
against exactly this kind of exact baseline.

`run_wormhole_protocol_trotter` is the follow-on originally left open:
a real gate circuit, closer to what hardware would run. Every
Hamiltonian/coupling term becomes an actual `pauli_rotation_ops` circuit
(basis-change + CNOT-staircase + RZ + inverse), verified to fidelity
**1.0** against exact `expm` for 1-, 2-, 3-, and 4-qubit mixed Pauli
strings (including a 4-qubit case matching a real SYK term), composed
via first-order Trotterization — verified to converge smoothly to exact
evolution as step count increases (infidelity drops ~4x per doubling of
steps, consistent with the expected quadratic scaling, checked directly
against the real N=8 SYK Hamiltonian rather than a toy example).

At the known peak (seed 61, t0=0.3, t1=0.60, a ~6300-gate real circuit),
the Trotterized version reproduces the exact result closely:

| | I(mu=+12) | I(mu=-12) | delta |
|---|---|---|---|
| exact | 0.01326 | 0.01793 | +0.00468 |
| Trotter (real gates) | 0.01301 | 0.01821 | +0.00520 |

And across a reduced sweep (t1 = 0.10 to 1.00), the Trotterized delta
tracks the exact curve's shape point-for-point — same sign everywhere,
same rise-then-fall structure peaking around t1≈0.6-0.7. **The
sign-dependent asymmetry is not an artifact of the exact-evolution
shortcut — it survives when computed the way real hardware would have
to compute it.**

## Status

Promoted into a real, shipped, tested package feature (v8.1.49). This
file (`research/wormhole_syk.py`) is kept as the original reproduction
and its own standalone verification suite (`__main__`), but is no longer
the only place this logic lives:

- The four generic building blocks this section originally flagged as
  reusable — Majorana JW mapping, partial-trace/von-Neumann-entropy/
  mutual-information, and `pauli_rotation_ops`/`trotter_evolve_ops` —
  now live in `dense_evolution.fermions`, `.entropy`, `.trotter`
  respectively, each with its own real tests (`tests/test_fermions.py`,
  `tests/test_entropy.py`, `tests/test_trotter.py`) checked against
  known-exact cases, not just imported and trusted.
- The SYK/wormhole-specific logic (sparse Hamiltonian construction, the
  paper's instance-selection criterion, the two-sided protocol itself)
  was ported into `dashboard_core.wormhole`, built on the promoted
  `dense_evolution` utilities, mirroring how `dashboard_core.hamiltonians`
  / `.vqe` are structured.
- The Composer kernel gained two real endpoints
  (`POST /api/wormhole_select_instance`, `POST /api/wormhole_teleportation`
  with `backend='exact'|'trotter'`), and `dense_evolution_mcp` gained
  three tools (`dense_evolution_wormhole_select_instance`,
  `_wormhole_teleportation`, `_wormhole_scan`) — all tested end to end
  against a live kernel, reproducing this file's exact reference numbers
  (seed 61, t0=0.3, t1=0.60: I(mu=+12)=0.01326/0.01301 exact/Trotter,
  I(mu=-12)=0.01793/0.01821).
- One real bug was found and fixed during promotion: the MCP scan tool's
  first implementation ran sweep points concurrently (mirroring
  `dense_evolution_energy_scan`'s pattern), but this protocol's per-call
  cost is real seconds, not a cheap lookup, and concurrent calls crashed
  the kernel process via a BLAS/`numpy.linalg.eigh` thread-safety issue
  under the heavier linear algebra — fixed by running the scan
  sequentially (see `mcp_server/server.py`'s `dense_evolution_wormhole_scan`).

See the main `README.md`'s Changelog (v8.1.49) and
`.claude/skills/dense-evolution-mcp/SKILL.md` for the user-facing summary.
