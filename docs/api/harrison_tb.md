# Harrison Tight-Binding (universal parameters)

Dependency-free (numpy only) sp3 tight-binding Hamiltonian builder from Harrison's
*universal* (materials-independent) parameter table -- no PySCF/OpenFermion needed.
Validated against real GaAs and Si; see
[`vhd_tb`](vhd_tb.md) for the more accurate material-specific alternative, and the
[Dense-Evolution-Ising-Tests validation page](https://tatopenn-cell.github.io/Dense-Evolution-Ising-Tests/harrison_tight_binding/)
for the full validation writeup (GaAs, Si, Ge gaps vs. experiment).

::: dense_evolution.harrison_tb

---

**See also**: [`vhd_tb`](vhd_tb.md) for material-specific fitted parameters when the
universal table's ~2-3x gap error isn't good enough.
