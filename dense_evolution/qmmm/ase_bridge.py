"""
ASE Calculator bridge for dense_evolution.native_hf (issue #288).

Motivation: QM/MM point 5's basis-flexibility test (Dense-Evolution-
Discovery, docs/qmmm_bond_order_and_embedding.md) needed a richer basis
(6-31G*) to test a real hypothesis about STO-3G's rigidity -- native_hf
already supports arbitrary basis names via `build_energy_fn`'s own
`basis_name` string parameter (nothing was actually hardcoded at the
library level; the QM/MM scripts' own `BASIS = "sto-3g"` constants were
just a script-level convenience). What native_hf does NOT give for free
is interop with the wider Python computational-chemistry ecosystem (ASE
Atoms objects, ASE's own optimizers/MD drivers, other engines' Atoms
representations) -- this bridge is exactly that interop layer, not a new
basis-set capability.

Requires ASE (the `ase` extra: pip install dense-evolution[ase]).
"""
import numpy as np

HARTREE_TO_EV = 27.211386245988  # CODATA, matches this project's own use of scipy.constants elsewhere (dense_evolution.qmmm.forces.ACCEL_CONVERSION)
ANGSTROM_TO_BOHR = 1.8897259886


def _import_ase_calculator():
    try:
        from ase.calculators.calculator import Calculator, all_changes
    except ImportError as exc:
        raise ImportError(
            "dense_evolution.qmmm.ase_bridge needs ASE, an optional dependency "
            "(pip install dense-evolution[ase]); it is not installed."
        ) from exc
    return Calculator, all_changes


class DenseEvolutionCalculator:
    """An ASE Calculator backed by dense_evolution.native_hf's own
    differentiable RHF energy (build_energy_fn) -- real Obara-Saika
    integrals and SCF, not a stub. `basis_name` is a plain string, passed
    straight through to native_hf (e.g. "sto-3g", "6-31g*", any basis
    basis_set_exchange has data for).

    ASE units: eV, Angstrom (this class converts from native_hf's own
    Hartree/Bohr at the boundary; everything inside native_hf itself
    stays atomic units).

    Only `energy` is implemented -- native_hf's forces come from
    dense_evolution.qmmm.forces.compute_hellmann_feynman_forces (a
    separate, already-real Hellmann-Feynman implementation with its own
    finite-difference derivative and its own molecule-catalog-shaped
    calling convention); this bridge does not reimplement or wrap that
    here.

    Examples
    --------
    >>> from ase import Atoms
    >>> from dense_evolution.qmmm.ase_bridge import DenseEvolutionCalculator
    >>> h2 = Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.7414]])
    >>> h2.calc = DenseEvolutionCalculator(atomic_numbers=[1, 1], nuclear_charges=[1.0, 1.0],
    ...                                    n_electrons=2, basis_name="sto-3g")
    >>> round(h2.get_potential_energy(), 2)
    -30.39
    """

    implemented_properties = ["energy"]

    def __new__(cls, atomic_numbers, nuclear_charges, n_electrons, basis_name):
        Calculator, all_changes = _import_ase_calculator()

        class _Impl(Calculator):
            implemented_properties = ["energy"]

            def __init__(self):
                super().__init__()
                self._atomic_numbers = atomic_numbers
                self._nuclear_charges = nuclear_charges
                self._n_electrons = n_electrons
                self._basis_name = basis_name
                self._energy_fn = None
                self._reference_geometry_bohr = None

            def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
                super().calculate(atoms, properties, system_changes)
                from dense_evolution.config import ensure_x64
                from dense_evolution.native_hf.differentiable import build_energy_fn
                ensure_x64()

                geometry_bohr = np.asarray(atoms.get_positions()) * ANGSTROM_TO_BOHR
                if self._energy_fn is None or "numbers" in system_changes:
                    self._reference_geometry_bohr = geometry_bohr
                    self._energy_fn = build_energy_fn(
                        self._atomic_numbers, self._nuclear_charges, self._n_electrons,
                        self._basis_name, self._reference_geometry_bohr,
                    )
                energy_hartree = float(self._energy_fn(geometry_bohr))
                self.results["energy"] = energy_hartree * HARTREE_TO_EV

        return _Impl()
