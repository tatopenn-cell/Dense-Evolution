---
name: dense-evolution-mcp
description: Drive Dense-Evolution's real quantum circuit simulator, VQE solver, molecular Hamiltonian builder, QM/MM force calculator, and MD trajectory engine directly through the dense_evolution_mcp MCP server, instead of writing new simulation code or describing what a circuit should do in the abstract. Use this skill whenever the user asks to run or build a quantum circuit, simulate a molecule, compute a ground-state/binding energy, run VQE, apply zero-noise extrapolation (ZNE) or noise mitigation, get Hellmann-Feynman forces, run a molecular dynamics trajectory, or mentions OpenQASM, qubits, an ansatz, Hartree energies, the Ising model, or the Dense Evolution / Composer kernel -- even if they don't name a tool exactly or say "use MCP".
compatibility: Requires the dense_evolution_mcp MCP server registered (see mcp_server/README.md in this repo) and the Composer kernel running (`dense-evolution serve`). Call dense_evolution_health first if unsure.
---

# Dense Evolution MCP

## Why this exists

`dense_evolution_mcp` gives you 16 tools that call the *same* local kernel
the published Composer web page uses (`local_site/app/server.py`) -- real
`DenseSVSimulator` runs, real Hartree-Fock Hamiltonians, real VQE with
adjoint differentiation, real Hellmann-Feynman forces. Prefer these tools
over writing new Python against `dense_evolution`/`dashboard_core`
directly: the kernel already handles qubit-count safety limits, backend
selection (dense vs MPS), and known platform pitfalls (e.g. never
constructing a `qiskit.QuantumCircuit` from QASM, which segfaults on
macOS -- the kernel parses QASM through `dense_evolution`'s own parser
instead). Writing fresh simulation code duplicates work that's already
correct and tested.

## Before anything else

Call `dense_evolution_health`. If it errors, the kernel isn't running --
the error message gives the exact command to start it
(`dense-evolution serve`). Don't try to work around a down kernel by
reimplementing the simulation locally; wait for it or tell the user to
start it.

If tackling something at the edge of what this machine can handle (many
qubits, a long MD trajectory, UCCSD-ansatz VQE), call
`dense_evolution_system_limits` and check `dense_evolution_health`'s RAM
figures first. Note that this only protects against a dense statevector
that literally won't fit in RAM -- it does *not* mean a request will be
fast. Measured directly: `dense_evolution_run_vqe` on H2 (4 qubits, 150
hardware-efficient iterations) took ~3s; the same ansatz on LiH (12
qubits) didn't finish even 10 iterations within this adapter's 180s
timeout. VQE cost scales steeply with qubit count, not just RAM -- for
anything beyond the smallest catalog molecules (H2, HeH+, H3+), start
with a very small `maxiter` (2-5) to see how long one iteration actually
takes on this machine before committing to a real run, rather than
assuming a size that worked for a small molecule will scale.

## Tool map

| Task | Tool(s) |
|---|---|
| Is the kernel up? What can this machine handle? | `dense_evolution_health`, `dense_evolution_system_limits` |
| What circuits/gates/noise models/molecules exist? | `dense_evolution_list_presets`, `_list_gates`, `_list_noise_models`, `_list_molecules` |
| Build QASM from a gate list instead of hand-writing it | `dense_evolution_build_circuit` |
| Run a circuit (counts, probabilities, statevector) | `dense_evolution_run_circuit` |
| Ground-state energy of a known/custom molecule | `dense_evolution_molecule_energy`, `_custom_molecule_energy` |
| Energy at several geometries (e.g. a dissociation curve) | `dense_evolution_energy_scan` -- one call, not one per point |
| Combine two molecular Hamiltonians | `dense_evolution_mix_molecules` |
| Optimize a variational ground state | `dense_evolution_run_vqe` |
| Nuclear forces / a dynamics trajectory | `dense_evolution_qmmm_forces`, `_md_trajectory` |
| Correct a noisy result back toward the ideal one | `dense_evolution_mitigate_zne` (scalar observable), `_mitigate_density_matrix` (full state) |

Every tool's own docstring has the full parameter/return schema -- this
table is for picking the right one quickly, not a substitute for reading
the tool description before calling it.

## Working with results

**Statevectors and probability arrays are truncated to their top ~25
entries by magnitude**, not returned in full. A 20-qubit run has over a
million basis states; the kernel already knows this and gives you the
dominant amplitudes plus a total count. If the user needs a *specific*
low-probability state's amplitude, say so rather than assuming it's
missing -- ask for a narrower circuit or note the limitation.

**Circuit/histogram/Q-sphere/Bloch images are never inlined as base64.**
`dense_evolution_run_circuit` only renders and saves them if you pass
`include_visualizations: true`, and then returns file *paths* (under
`~/.dense_evolution_mcp/images` by default). Read the file at that path
if you need to look at the image; don't expect it inline in the tool
response, and don't ask for visualizations you don't actually need to
answer the user's question -- rendering four PNGs costs real time on top
of the simulation itself.

**Mapping consistency matters for `_mix_molecules`.** Both molecules must
use the same `mapping` value (`jordan_wigner` or `bravyi_kitaev`) and have
the same qubit count -- they're being added as Hamiltonians on the same
Hilbert space, so a mismatch is physically meaningless, not just a format
error. The tool rejects it; don't retry with different weights hoping it
resolves the qubit-count mismatch, fix the molecule choice instead.

**`_custom_molecule_energy` caps at 12 qubits** (exact dense
diagonalization has a real limit here, not an arbitrary one) and
**`_md_trajectory` caps `n_steps` at 200 normally, 30 if
`recompute_electronic_state=true`** (true ab-initio MD re-solves
Hartree-Fock every step, which is much more expensive). If a request
would exceed these, say so up front with the actual numbers rather than
attempting it and reporting the tool's rejection as a surprise.

## Common workflows

**"Run this circuit / build me a Bell state / GHZ state":**
Check `dense_evolution_list_presets` first -- common circuits are likely
already there. Otherwise write OpenQASM directly (or use
`dense_evolution_build_circuit` from a gate list) and call
`dense_evolution_run_circuit`. Use `noise_model: "ideal"` unless the user
specifically wants noise.

**"What's the ground-state energy of X" / "simulate this molecule":**
`dense_evolution_list_molecules` to check if it's in the catalog (fast
path: `dense_evolution_molecule_energy`). If not, and it's small enough,
`dense_evolution_custom_molecule_energy` with symbols + geometry in
Angstrom.

**"Optimize / find the ground state with VQE":**
`dense_evolution_run_vqe`. Default `ansatz_type="hardware_efficient"` is
fine for a first pass at any molecule size. Only reach for `"uccsd"` when
the user cares about chemical accuracy or parameter-efficiency over
wall-clock time -- it converges in fewer iterations but each iteration is
far more expensive, so mention that tradeoff rather than silently
defaulting to it.

**"How noisy is this / correct for noise":**
Run once with `noise_model: "ideal"` for a baseline, once with a real
noise model to see the effect, then `dense_evolution_mitigate_zne` (single
observable) or `_mitigate_density_matrix` (full state fidelity) to show
the corrected result alongside both.

**"Simulate the dynamics / forces on this molecule":**
`dense_evolution_qmmm_forces` for a single-point force calculation;
`dense_evolution_md_trajectory` for a trajectory over time. Default
(`recompute_electronic_state=false`) is fine for short trajectories close
to the starting geometry -- flag to the user if they're asking about
larger geometry changes, where the fixed-electronic-state approximation
degrades and the (much slower) `true` setting matters more.

## If the kernel isn't set up yet

Point the user at `mcp_server/README.md` in this repo -- it has the
install command, how to start the kernel, and how to register the MCP
server with `claude mcp add`. Don't restate that setup here; this file is
about using the tools once they're available, not installing them.
