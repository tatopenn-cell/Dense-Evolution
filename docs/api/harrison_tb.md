# Harrison Tight-Binding (universal parameters)

Building a solid's electronic band structure normally means DFT or an SCF calculation
-- this module skips both. Harrison's tight-binding model builds the Hamiltonian from
two universal ingredients only: tabulated free-atom orbital energies, and one
bond-scaling law with the *same* four coefficients for every element pair. No fitting
per material -- the tradeoff for that convenience is accuracy (see the Accuracy
section below).

## Step 1. A real Si-Si bond, eigenvalues

```python
import numpy as np
import dense_evolution as de

H = de.sp3_dimer_hamiltonian('Si', 'Si', bond_length_angstrom=2.35)
np.round(np.linalg.eigvalsh(H), 3)
```

```
array([-16.626, -12.25 ,  -9.847,  -7.638,  -7.638,  -5.402,  -5.402,  -1.418])
```

`sp3_dimer_hamiltonian(element_a, element_b, bond_length_angstrom)` builds an `8x8`
Hamiltonian -- one s and three p orbitals per atom, two atoms -- for a real Si-Si bond
at Si's real bulk bond length (`2.35` Angstrom). The eigenvalues are the dimer's
molecular-orbital energies in eV, built entirely from Harrison's tabulated atomic term
values and universal `eta` bond-scaling coefficients, no per-material fitting anywhere.

## Step 2. The full periodic crystal, at one k-point

```python
H_k = de.zincblende_hamiltonian((0, 0, 0), 'Si', 'Si', lattice_constant_angstrom=5.43)
eigvals = np.linalg.eigvalsh(H_k)
float(eigvals[4] - eigvals[3])
```

```
3.665861271362627
```

`zincblende_hamiltonian(k, cation, anion, lattice_constant_angstrom)` sums the same
bond physics with Bloch phases over the 4 nearest-neighbor bonds of a real zincblende/
diamond crystal -- `k=(0,0,0)` is the Gamma point, `5.43` Angstrom is silicon's real
lattice constant. With 4 valence electrons per Si atom (8 total, 4 filled bands), the
gap between the 4th and 5th eigenvalue at Gamma comes out `3.67` eV -- real silicon's
actual band structure is more subtle than this single number (see Accuracy below).

---

## Details

**Accuracy**: this is a *universal* model -- one parameter table for every material, no
per-material fitting, no d-orbitals. That buys zero setup cost per new material at the
price of accuracy: gaps typically come out ~2-3x off from experiment, and for
*indirect*-gap materials (silicon included) this simple Gamma-only reading can
altogether miss where the true conduction-band minimum sits -- see
[`vhd_tb`](vhd_tb.md)'s own Step 2 for the real Si example (Gamma-only gives the wrong
picture; scanning along a k-path finds the true, off-Gamma minimum). Validation numbers
against real experimental gaps (GaAs, Si, Ge) are tracked in
[Dense-Evolution-Discovery](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/harrison_tight_binding/).

**Source**: Walter A. Harrison, *Electronic Structure and the Properties of Solids: The
Physics of the Chemical Bond* (W. H. Freeman, 1980; Dover, 1989, ISBN 0-486-66021-4).
Atomic term values (`HARRISON_ELEMENTS`) and universal eta coefficients (`HARRISON_ETA`)
are transcribed from that book's Solid State Table, cross-checked against
[jarvist/HarrisonSolidStateTable.jl](https://github.com/jarvist/HarrisonSolidStateTable.jl),
an independent Julia implementation of the same table.

**The bond-scaling law**: off-diagonal hopping matrix elements follow
`V = eta * hbar^2/(m_e * d^2)` (`hbar^2/m_e = 7.62 eV*Angstrom^2`), with `eta_ss_sigma
= -1.40`, `eta_sp_sigma = +1.84`, `eta_pp_sigma = +3.24`, `eta_pp_pi = -0.81` -- the
same four numbers for every element pair, only the bond length `d` and each element's
own atomic term values change between materials. Full sp3 Slater-Koster (1954) matrix
elements for a bond of direction cosines `(l, m, n)` are in the module source.

::: dense_evolution.solvers.harrison_tb

---

**See also**: [`vhd_tb`](vhd_tb.md) for material-specific fitted parameters when the
universal table's ~2-3x gap error isn't good enough.
