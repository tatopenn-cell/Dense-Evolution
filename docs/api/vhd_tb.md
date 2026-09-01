# Vogl-Hjalmarson-Dow Tight-Binding (material-specific parameters)

Same Slater-Koster sp3 machinery as [`harrison_tb`](harrison_tb.md), plus one addition
-- an extra **s\*** orbital per atom, a pure fitting degree of freedom with no physical
meaning of its own, whose only job is pulling the lowest conduction band down to the
right energy. `harrison_tb`'s universal table structurally cannot do that (see its own
Accuracy section); this module trades the "one table fits everything" convenience for
a parameter row fitted per material, and gets real accuracy back in return.

## Step 1. The gap at Gamma

```python
import dense_evolution as de

Si = de.VHD_MATERIALS['Si']
de.direct_gap_at_gamma(Si)
```

```
3.4299999999999984
```

`VHD_MATERIALS['Si']` is silicon's real fitted parameter row (Vogl, Hjalmarson & Dow
1983). `direct_gap_at_gamma(material)` reads the gap at the Gamma point (`k=(0,0,0)`)
-- `3.43` eV, matching real silicon's actual *direct* gap almost exactly. That number
alone would be misleading, though: silicon's fundamental gap -- the one that actually
sets its optical/electronic behavior -- is not this one.

## Step 2. Why Gamma alone isn't enough

```python
vbm, vbm_k, cbm, cbm_k, gap = de.band_extrema_along_path(
    Si, k_start=(0, 0, 0), k_end=(1, 0, 0)
)
cbm_k, gap
```

```
(array([0.732, 0.   , 0.   ]), np.float64(1.171344217906544))
```

`band_extrema_along_path(material, k_start, k_end)` scans a k-space line (here Gamma to
X, in units of `2*pi/a`) and finds the real valence-band maximum and conduction-band
minimum -- which don't have to sit at the same k-point at all. For silicon, they don't:
the conduction-band minimum lands at `k=0.732` along Gamma-X, not at Gamma, and the true
gap between it and the valence-band maximum is `1.17` eV -- close to real silicon's
well-known experimental indirect gap (`1.12` eV), and nothing like Step 1's `3.43` eV
Gamma-only number. This is exactly what makes silicon an *indirect*-gap material, and
exactly the failure mode `harrison_tb`'s simpler universal model can't resolve on its
own.

---

## Details

**16 materials included**: C, Si, Ge, Sn, SiC, AlP, AlAs, AlSb, GaP, GaAs, GaSb, InP,
InAs, InSb, ZnSe, ZnTe -- all zinc-blende/diamond semiconductors, each with its own
fitted `VHD_MATERIALS` row (`Esa`/`Epa`/`Esc`/`Epc`/`Essa`/`Essc` on-site energies plus
7 hopping integrals), not derived from a universal formula the way `harrison_tb`'s
`ETA` table is.

**`sp3s_star_hamiltonian(kxyz, material)`**: the lower-level Bloch Hamiltonian builder
both `direct_gap_at_gamma` and `band_extrema_along_path` call internally -- a `10x10`
matrix per unit cell (5 orbitals/atom, including s\*), not the `8x8` `harrison_tb`
builds. `k` is in units of `2*pi/a` along the conventional cubic axes: Gamma=`(0,0,0)`,
X=`(1,0,0)`, L=`(0.5,0.5,0.5)`.

**Source**: P. Vogl, H. P. Hjalmarson, J. D. Dow, "A Semi-empirical tight-binding
theory of the electronic structure of semiconductors," J. Phys. Chem. Solids 44 (5),
365-378 (1983). Parameter values transcribed from an independent open-source
implementation, [github.com/rpmuller/TightBinding](https://github.com/rpmuller/TightBinding)
(`TB.py`), which cites the same paper.

**Accuracy**: validated against real experimental gaps (GaAs, Si, Ge) -- see the
[Dense-Evolution-Discovery validation page](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/harrison_tight_binding/)
for the full writeup and numbers.

::: dense_evolution.solvers.vhd_tb

---

**See also**: [`harrison_tb`](harrison_tb.md) for the zero-per-material-fitting
universal alternative, when the ~2-3x gap error is an acceptable tradeoff for not
needing a fitted parameter row per material.
