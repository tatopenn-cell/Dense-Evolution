# Vogl-Hjalmarson-Dow Tight-Binding (material-specific parameters)

sp3s\* tight-binding Hamiltonian builder using material-specific fitted
parameters -- far more accurate than [`harrison_tb`](harrison_tb.md)'s
universal table, at the cost of needing a separately fitted parameter row
per material instead of one table for everything.

## Source

P. Vogl, H. P. Hjalmarson, J. D. Dow, "A Semi-empirical tight-binding
theory of the electronic structure of semiconductors", J. Phys. Chem.
Solids 44 (5), 365-378 (1983). Parameter values (`MATERIALS`) are
transcribed from an independent open-source implementation,
[github.com/rpmuller/TightBinding](https://github.com/rpmuller/TightBinding)
(`TB.py`), which cites the same paper. 16 zinc-blende/diamond
semiconductors included: C, Si, Ge, Sn, SiC, AlP, AlAs, AlSb, GaP, GaAs,
GaSb, InP, InAs, InSb, ZnSe, ZnTe.

## The method

Same Slater-Koster sp3 machinery as [`harrison_tb`](harrison_tb.md), plus
one addition: an extra **s\*** orbital per atom (5 orbitals/atom instead of
4, 10x10 Hamiltonian per unit cell instead of 8x8). s\* has no literal
physical meaning -- it exists purely as a fitting degree of freedom to pull
the lowest conduction band down to the right energy, something a bare sp3
basis structurally cannot do (see `harrison_tb`'s accuracy notes: it
routinely misplaces the conduction-band minimum for indirect-gap
materials). Each element's `Material` row (`Esa/Epa/Esc/Epc/Essa/Essc` +
7 hopping integrals `Vss/Vxx/Vxy/Vsapc/Vscpa/Vssapc/Vsscpa`) is fitted as a
whole per material, not derived from a universal formula the way
`harrison_tb.ETA` is.

`sp3s_star_hamiltonian(kxyz, material)` builds the periodic Bloch
Hamiltonian directly (k in units of $2\pi/a$ along the conventional cubic
axes: $\Gamma=(0,0,0)$, $X=(1,0,0)$, $L=(0.5,0.5,0.5)$).
`direct_gap_at_gamma` reads off the gap at $\Gamma$ -- only the true
fundamental gap for *direct*-gap materials. For *indirect*-gap materials
(Si, Ge), the real conduction-band minimum sits off-$\Gamma$, so
`band_extrema_along_path(material, k_start, k_end)` scans a k-space line
and finds the true valence-band max / conduction-band min instead of
reading only $\Gamma$.

## Accuracy

Validated against real experimental gaps (GaAs, Si, Ge) -- see the
[Dense-Evolution-Discovery validation page](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/harrison_tight_binding/)
for the full writeup and numbers.

::: dense_evolution.vhd_tb

---

**See also**: [`harrison_tb`](harrison_tb.md) for the zero-per-material-fitting
universal alternative, when the ~2-3x gap error is an acceptable tradeoff for
not needing a fitted parameter row per material.
