# QM/MM — Region Partitioning, Embedding, Forces

Everything QM/MM in this library, in one place (issue #283): real
region partitioning around a reactive bond, real coordinate slicing, a
Diffuse2Seg-derived relevance propagation, and the real Hellmann-Feynman
forces + Velocity-Verlet MD this repo already had (moved here from
`dashboard_core.qmmm`, which still re-exports them for compatibility).

Region partitioning and propagation need the `qmmm` extra
(`pip install dense-evolution[qmmm]`, installs RDKit); forces/MD need no
extra dependency beyond what `dashboard_core` already requires.

```python
from rdkit import Chem
from dense_evolution.qmmm import partition_qm_mm_region, sliced_geometry

mol = Chem.AddHs(Chem.MolFromSmiles("OCCCCCC"))  # 1-hexanol
qm_atoms, boundary_pairs = partition_qm_mm_region(mol, {0, 1}, radius=1)
print(sorted(qm_atoms))       # [0, 1, 2, 7, 8, 9, 10, 11]
print(boundary_pairs)         # [(2, 3)] -- one bond crosses the boundary
```

`partition_qm_mm_region` walks outward from the reactive bond's atoms in
heavy-atom hops. Hydrogens always follow their own heavy atom (never
independently walked -- that spuriously cuts a terminal C-H bond and
leaves an empty MM fragment). A boundary bond that would cut into an
aromatic ring pulls the whole ring into the QM region first: cutting a
lone aromatic atom out of its ring and capping it with hydrogen is not a
valid molecule.

```python
import numpy as np

atoms, geom_bohr = sliced_geometry(atomic_numbers, geom_bohr, qm_atoms, boundary_pairs)
```

`sliced_geometry` takes a coordinate SUBSET of one whole-molecule
conformer -- never an independently re-embedded fragment, which gives
unrelated 3D structures across fragments and was the real cause behind
an MMFF94 mechanical correction that looked like it helped but turned
out to be compensating for that geometry choice instead of truncation
itself (see Dense-Evolution-Discovery's
`docs/qmmm_region_partitioning_mmff_correction.md` for the full
retraction).

```python
from dense_evolution.qmmm import propagate_relevance

relevance = propagate_relevance(bond_order_matrix, seed_idx=[0, 1], n_nodes=n_atoms)
region = {i for i in range(n_atoms) if relevance[i] > threshold}
```

`propagate_relevance` is Algorithm 1 of Hümmer, Sicking, Hüger &
Gottschalk 2026 ("Diffuse2Seg", arXiv:2609.06491) -- non-linear
p-Laplacian graph propagation, solved by Gauss-Jacobi iteration,
originally built to spread point prompts through a diffusion model's
self-attention for image segmentation. Here "affinity" can be any
node-graph weight, e.g. real Mayer/Wiberg bond order between atoms
instead of self-attention between image patches. **Measured, not
assumed**: on a real branched-aromatic molecule, this does NOT reproduce
a naive "stronger bond propagates further" result at the paper's own
calibrated hyperparameters (tuned for a dense multi-prompt image
pipeline, not a single molecular seed) -- see
Dense-Evolution-Discovery's `docs/qmmm_utils.md` for the full lambda
sweep. Included here as a real, correctly-implemented primitive; not a
proven QM/MM region-selection win yet.

```python
from ase import Atoms
from dense_evolution.qmmm.ase_bridge import DenseEvolutionCalculator

h2 = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.7414]])
h2.calc = DenseEvolutionCalculator(atomic_numbers=[1, 1], nuclear_charges=[1.0, 1.0],
                                    n_electrons=2, basis_name="sto-3g")
print(h2.get_potential_energy())  # -30.39 eV
```

`DenseEvolutionCalculator` (issue #288, needs the `ase` extra:
`pip install dense-evolution[ase]`) is an ASE Calculator backed by
`native_hf`'s own differentiable energy (`build_energy_fn`) -- real
Obara-Saika integrals and SCF, not a stub, for interop with ASE's
optimizers/MD drivers and other engines' `Atoms` representations.

```python
h2.calc = DenseEvolutionCalculator(atomic_numbers=[1, 1], nuclear_charges=[1.0, 1.0],
                                    n_electrons=2, basis_name="6-31g*")
print(h2.get_potential_energy())  # -30.66 eV
```

`basis_name` is a plain string -- `"sto-3g"`, `"6-31g"`, `"6-31g*"`,
anything `basis_set_exchange` has data for -- passed straight through to
`native_hf`, which already supported arbitrary bases before this bridge
existed (nothing here adds new basis-set capability; this is purely the
ASE-interop layer). Swapping the basis on the SAME geometry is the whole
point: 6-31G gives a lower (better, more variational freedom) energy
than STO-3G for real hydrogen, exactly as physics requires. 6-31G and
6-31G* give the IDENTICAL energy for H2 specifically -- not a bug: `*`
adds polarization d-functions to heavy atoms only, and hydrogen has none
to add here.

Only `energy` is implemented (`implemented_properties = ["energy"]`) --
`native_hf`'s own forces come from a separate, already-real
implementation, `compute_hellmann_feynman_forces` above, with its own
finite-difference derivative and its own calling convention (a molecule
name from `MOLECULE_CATALOG`, not a bare ASE `Atoms` object). This
bridge does not wrap that here, so ASE's gradient-based optimizers
(`BFGS`, `FIRE`, ...) cannot be driven by it yet -- only single-point
energies at any geometry/basis you construct directly.

::: dense_evolution.qmmm

---

**See also**: [`dense_evolution.native_hf`](native_hf.md) for the
Hartree-Fock engine these region functions feed into;
[`dashboard_core.hamiltonians`](dashboard_core_hamiltonians.md) for
`MOLECULE_CATALOG` and `build_molecular_hamiltonian`, which
`compute_hellmann_feynman_forces`/`run_md_trajectory` are built from.
