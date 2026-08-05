# Harrison Tight-Binding (universal parameters)

Dependency-free (numpy only) sp3 tight-binding Hamiltonian builder for real
atoms and crystals -- no PySCF/OpenFermion, no SCF/DFT. See
[`vhd_tb`](vhd_tb.md) for the material-specific alternative when this
module's accuracy isn't enough.

## Source

**Walter A. Harrison**, *Electronic Structure and the Properties of Solids:
The Physics of the Chemical Bond*. Originally published by W. H. Freeman,
1980; reprinted by Dover Publications (Dover Books on Physics), 1989,
ISBN 0-486-66021-4. Atomic term values (`ELEMENTS`) and the universal eta
coefficients (`ETA`) are transcribed from that book's Solid State Table,
cross-checked against
[`jarvist/HarrisonSolidStateTable.jl`](https://github.com/jarvist/HarrisonSolidStateTable.jl),
an independent Julia implementation of the same table.

## The method

Harrison's tight-binding model builds a solid's electronic Hamiltonian from
two ingredients only, both universal (materials-independent functional
form):

1. **Atomic term values** -- the free-atom s and p orbital energies
   (on-site Hamiltonian diagonal), tabulated per element.
2. **A universal bond-scaling law** for the off-diagonal (hopping) matrix
   elements between neighboring atoms' orbitals:

   $$V_{ll'm} = \eta_{ll'm} \cdot \frac{\hbar^2}{m_e d^2}$$

   where $d$ is the bond length and the four dimensionless $\eta$
   coefficients are the *same for every element pair* -- only $d$ and the
   atomic term values change between materials. This is what makes the
   method "universal": no fitting per material.

$\hbar^2/m_e = 7.62\ \text{eV·Å}^2$.

| coefficient | value |
|---|---|
| $\eta_{ss\sigma}$ | -1.40 |
| $\eta_{sp\sigma}$ | +1.84 |
| $\eta_{pp\sigma}$ | +3.24 |
| $\eta_{pp\pi}$ | -0.81 |

Off-diagonal sp3 matrix elements follow the standard Slater-Koster (1954)
table for an (s, px, py, pz) basis and a bond of direction cosines
$(l, m, n)$:

$$E(s,s) = V_{ss\sigma}, \quad E(s,x) = l\,V_{sp\sigma}, \quad E(x,s) = -l\,V_{sp\sigma}$$
$$E(x,x) = l^2 V_{pp\sigma} + (1-l^2)V_{pp\pi}, \quad E(x,y) = lm\,(V_{pp\sigma}-V_{pp\pi})$$

(and cyclic permutations for y, z). `sp3_bond_block` implements this;
`sp3_dimer_hamiltonian` builds a 2-atom cluster from it, and
`zincblende_hamiltonian` sums it with Bloch phases over the 4
nearest-neighbor bonds to build the full periodic crystal Hamiltonian.

## Accuracy

This is a *universal* model -- one parameter table for every material, no
per-material fitting, no d-orbitals. That buys zero setup cost per new
material at the price of accuracy: gaps typically come out ~2-3x off from
experiment, and for indirect-gap materials it can misplace the
conduction-band minimum entirely (see [`vhd_tb`](vhd_tb.md) for why and the
fix). Validation numbers against real experimental gaps (GaAs, Si, Ge) are
tracked in
[Dense-Evolution-Ising-Tests](https://tatopenn-cell.github.io/Dense-Evolution-Ising-Tests/harrison_tight_binding/).

::: dense_evolution.harrison_tb

---

**See also**: [`vhd_tb`](vhd_tb.md) for material-specific fitted parameters when the
universal table's ~2-3x gap error isn't good enough.
