# Vogl-Hjalmarson-Dow Tight-Binding (material-specific parameters)

sp3s\* tight-binding Hamiltonian builder using Vogl-Hjalmarson-Dow (1983)
per-material fitted parameters -- far more accurate than
[`harrison_tb`](harrison_tb.md)'s universal table at the cost of needing a
separately fitted parameter row per material. 16 zinc-blende/diamond
semiconductors included (C, Si, Ge, Sn, SiC, AlP, AlAs, AlSb, GaP, GaAs,
GaSb, InP, InAs, InSb, ZnSe, ZnTe).

Validation against real experimental gaps (GaAs, Si, Ge) is a physics
result, not an engine/API concern -- see the
[Dense-Evolution-Ising-Tests validation page](https://tatopenn-cell.github.io/Dense-Evolution-Ising-Tests/harrison_tight_binding/)
for the full writeup and numbers.

::: dense_evolution.vhd_tb

---

**See also**: [`harrison_tb`](harrison_tb.md) for the zero-per-material-fitting
universal alternative, when the ~2-3x gap error is an acceptable tradeoff for
not needing a fitted parameter row per material.
