# Mass Decomposition

> Checking whether a mass difference between two spectrometry peaks
> corresponds to a chemically real neutral loss — not a general chemistry
> library, just this one question.

A mass spectrometer reports peaks, not atoms. Given a precursor's molecular
formula and a mass difference observed between two of its peaks, the real
question is: does *any* combination of the precursor's own atoms add up to
that mass difference, and is that combination a chemically valid fragment
(not just an arithmetic coincidence)? `mass_decomposition` answers both
parts: an exact search for reachable masses, and a chemical-validity filter
on top of it.

## Step 1. Parse a formula and check its validity

```python
from dense_evolution.utils.mass_decomposition import parse_formula, rdbe

formula = parse_formula("C6H12O6")
formula, rdbe(formula)
```

```
({'C': 6, 'H': 12, 'O': 6}, 1.0)
```

`parse_formula` turns a formula string into `{element: count}`. `rdbe`
(Ring-plus-Double-Bond-Equivalent, the standard organic-chemistry degree of
unsaturation) checks whether that atom count could belong to a real,
closed-shell molecule: `RDBE = 1 + sum(count_i * (valence_i - 2)) / 2`, and
a real molecule always has `RDBE >= 0`. Glucose's `RDBE = 1.0` is real and
valid; benzene (`C6H6`) comes out to `4.0` (three double bonds plus one
ring), and water (`H2O`) to `0.0` — both real, independently known values.

## Step 2. Find whether a specific mass loss is reachable

```python
from dense_evolution.utils.mass_decomposition import build_reachable_masses, nearest_reachable_mass

reach = build_reachable_masses(formula, max_mass=50.0)
water_loss = 18.0106
len(reach), nearest_reachable_mass(water_loss, reach)
```

```
(53, 18.010565)
```

`build_reachable_masses` enumerates every sub-formula mass reachable by
using between 0 and the full count of each element — an exact, bounded
integer subset-sum, solved via iterative Minkowski sums rather than a
continuous approximation. `nearest_reachable_mass` then answers the actual
question: glucose's own atoms can reach `18.010565`, matching a water-loss
neutral fragment (`H2O = 18.0106`) to within microscopic error, using only
atoms glucose actually has.

## Step 3. The same landscape via FFT, for larger formulas

```python
from dense_evolution.utils.mass_decomposition import build_reachable_density_fft, density_at_mass

mass_grid, density = build_reachable_density_fft(formula, max_mass=50.0)
density_at_mass(mass_grid, density, water_loss)
```

```
0.3362
```

![Glucose's reachable-mass landscape: exact reachable masses (grey dots on the axis) and the FFT plausibility density (blue curve), against the water-loss target at 18.0106 Da](assets/mass_decomposition/glucose_reachable_density.png)

`build_reachable_masses` truncates once the number of distinct reachable
states passes `max_states` (150,000 by default) — fine for small formulas,
a real limitation for large ones. `build_reachable_density_fft` builds the
same landscape a different way: each element's own reachable masses are a
spike train, and the combined landscape across every element is their
*convolution* — which the convolution theorem turns into a plain pointwise
product of FFTs, and one inverse FFT back. The result is a plausibility
*density*, not a 0/1 reachability array: `0.3362` at the water-loss mass
means real, well-supported combinations land there, not that a fragment is
present with 34% probability.

## Details

### Where this came from

Promoted from Dense-Evolution-Discovery's CASMI26 spectral-identification
experiments (real MS/MS data, the OTRF/Enveda CASMI26 Kaggle dataset). Six
different formula-scoring methods were tried there for true-vs-wrong-formula
discrimination on real spectra; RDBE-filtered exact subset-sum was the best
of the six.

### Why the RDBE filter matters

Checked directly on real CASMI26 data (n=1952 real spectra): without the
RDBE filter, an unrelated, *wrong* candidate formula "explains" a real
peak-pair mass difference by pure arithmetic coincidence 58.5% of the time
(mean). Filtering by `RDBE >= 0` does not close that gap by itself, but it
does remove chemically impossible matches from both the right and wrong
candidates — so any downstream threshold built on `build_reachable_masses`
is scoring real, buildable neutral losses, not arithmetic accidents.

### The FFT construction

`build_reachable_density_fft` uses the same technique Rockwood & Van Orden
use to compute isotope distributions (*Ultrahigh-Speed Calculation of
Isotope Distributions*, Anal. Chem. 1996): a 0..count_i spike train per
element instead of an isotope-abundance distribution, combined by FFT
multiplication instead of direct convolution. A single Gaussian broadening
(`relative_tolerance`, ppm-level) is applied once, after combining every
element, matching how real mass-measurement uncertainty scales with the
final observed mass rather than with each element's own contribution.
`test_mass_decomposition.py` verifies this FFT route agrees with both
brute-force convolution and `build_reachable_masses`'s own exact
enumeration on the same formula.

### Bounded, not unbounded, search

Both `build_reachable_masses` and `build_reachable_density_fft` cap each
element's usable count at 60 and take an explicit `max_mass` — real
molecules in this use case (CASMI26 precursors) are small enough that the
reachable-state space stays enumerable, and `max_mass` (usually the
precursor's own observed mass) keeps the search from exploring atom counts
no real fragment of that precursor could ever reach.

::: dense_evolution.utils.mass_decomposition
