"""
Real tests for dense_evolution.qmmm.region/propagation (issue #283) and
dense_evolution.qmmm.ase_bridge (issue #288). rdkit/ase-dependent tests
use pytest.importorskip(...) per-test, same convention as
test_libcint_bridge.py's pyscf tests -- both are installed in CI (the
`qmmm`/`ase` extras) so these cover the real logic, not just the
ImportError path.

Every number below was computed directly by running the real functions
(not invented) -- see the module docstring's promotion note for the
Dense-Evolution-Discovery experiments this was validated against first.
"""
import numpy as np
import pytest

from dense_evolution.qmmm.propagation import propagate_relevance


class TestPropagateRelevance:
    def test_monotonic_decay_along_a_path_graph(self):
        """Path graph 0-1-2 (uniform affinity 1.0 on both edges), seeded
        at node 0: relevance must strictly decrease with graph distance
        from the seed."""
        affinity = np.array([[0, 1.0, 0], [1.0, 0, 1.0], [0, 1.0, 0]])
        rel = propagate_relevance(affinity, [0], 3, lam=1.0)
        assert rel[0] > rel[1] > rel[2]

    def test_stronger_edge_propagates_further(self):
        """Two branches of equal length from a shared seed, one with a
        stronger bond (2.0) than the other (0.5): the stronger branch
        must end up with more relevance at the far node."""
        # 0 (seed) -- 1 (strong, w=2.0) -- 2
        # 0 (seed) -- 3 (weak,   w=0.5) -- 4
        n = 5
        affinity = np.zeros((n, n))
        affinity[0, 1] = affinity[1, 0] = 2.0
        affinity[1, 2] = affinity[2, 1] = 2.0
        affinity[0, 3] = affinity[3, 0] = 0.5
        affinity[3, 4] = affinity[4, 3] = 0.5
        rel = propagate_relevance(affinity, [0], n, lam=1.0)
        assert rel[2] > rel[4]

    def test_degenerate_case_leaves_uniform_component_unchanged(self):
        """A node with zero local affinity gradient (isolated from the
        seed's component entirely) must stay at exactly 0, not raise or
        produce NaN from the g_i=0 singularity in the p-Laplacian weight."""
        # node 2 is disconnected from the seed's component {0, 1}
        affinity = np.array([[0, 1.0, 0], [1.0, 0, 0], [0, 0, 0]])
        rel = propagate_relevance(affinity, [0], 3, lam=1.0)
        assert np.isfinite(rel).all()
        assert rel[2] == 0.0

    def test_seed_itself_stays_the_most_relevant_node(self):
        affinity = np.array([[0, 1.0, 0.5], [1.0, 0, 1.0], [0.5, 1.0, 0]])
        rel = propagate_relevance(affinity, [0], 3, lam=1.0)
        assert rel[0] == rel.max()


class TestPartitionQmMmRegion:
    def test_bfs_radius_one_on_hexanol(self):
        """1-hexanol (OCCCCCC), seeded at the O-C1 bond (atoms 0, 1):
        radius=1 must include exactly the O, C1, C2 heavy atoms plus
        their hydrogens, with one boundary bond (C2-C3) -- verified by
        running the function directly."""
        pytest.importorskip("rdkit")
        from rdkit import Chem
        from dense_evolution.qmmm.region import partition_qm_mm_region

        mol = Chem.AddHs(Chem.MolFromSmiles("OCCCCCC"))
        qm_atoms, boundary_pairs = partition_qm_mm_region(mol, {0, 1}, radius=1)
        assert qm_atoms == {0, 1, 2, 7, 8, 9, 10, 11}
        assert boundary_pairs == [(2, 3)]

    def test_ring_safety_pulls_in_the_whole_aromatic_ring(self):
        """OCC(c1ccccc1)CCC: a radius=2 cut from the O-C1 bond would
        naively land inside the aromatic ring. Every aromatic atom in the
        molecule must end up in the QM region -- cutting a lone aromatic
        atom out of its ring and capping it with hydrogen is not a valid
        molecule (this is the real AtomKekulizeException bug this
        function was written to fix)."""
        pytest.importorskip("rdkit")
        from rdkit import Chem
        from dense_evolution.qmmm.region import partition_qm_mm_region

        mol = Chem.AddHs(Chem.MolFromSmiles("OCC(c1ccccc1)CCC"))
        qm_atoms, _boundary_pairs = partition_qm_mm_region(mol, {0, 1}, radius=2)
        aromatic_atoms = {a.GetIdx() for a in mol.GetAtoms() if a.GetIsAromatic()}
        assert aromatic_atoms <= qm_atoms

    def test_hydrogens_always_follow_their_own_heavy_atom(self):
        """Every hydrogen bonded to a QM heavy atom must itself be in the
        QM region -- the real bug this function fixes is independently
        BFS-expanding hydrogens, which can spuriously cut a terminal C-H
        bond instead."""
        pytest.importorskip("rdkit")
        from rdkit import Chem
        from dense_evolution.qmmm.region import partition_qm_mm_region

        mol = Chem.AddHs(Chem.MolFromSmiles("OCCCCCC"))
        qm_atoms, _boundary_pairs = partition_qm_mm_region(mol, {0, 1}, radius=1)
        for idx in list(qm_atoms):
            atom = mol.GetAtomWithIdx(idx)
            if atom.GetAtomicNum() > 1:
                for nbr in atom.GetNeighbors():
                    if nbr.GetAtomicNum() == 1:
                        assert nbr.GetIdx() in qm_atoms


class TestSlicedGeometry:
    def test_capping_hydrogen_at_the_boundary_bond(self):
        from dense_evolution.qmmm.region import sliced_geometry, CH_BOND_BOHR

        numbers = [8, 6, 6]
        geom = np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [2.8, 0.0, 0.0]])
        new_numbers, new_geom = sliced_geometry(numbers, geom, {0, 1}, [(1, 2)])
        assert new_numbers == [8, 6, 1]
        assert new_geom.shape == (3, 3)
        cap_bond_length = np.linalg.norm(new_geom[2] - geom[1])
        assert cap_bond_length == pytest.approx(CH_BOND_BOHR)


class TestAseBridge:
    def test_h2_sto3g_energy_matches_native_hf(self):
        """Real value, computed directly by running the bridge (not
        invented): -30.39 eV, consistent with native_hf's own H2/STO-3G
        energy elsewhere in this project (~-1.117 Hartree)."""
        pytest.importorskip("ase")
        from ase import Atoms
        from dense_evolution.qmmm.ase_bridge import DenseEvolutionCalculator

        h2 = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.7414]])
        h2.calc = DenseEvolutionCalculator(atomic_numbers=[1, 1], nuclear_charges=[1.0, 1.0],
                                            n_electrons=2, basis_name="sto-3g")
        energy_ev = h2.get_potential_energy()
        assert energy_ev == pytest.approx(-30.39, abs=0.01)

    def test_richer_basis_gives_a_lower_variational_energy(self):
        """6-31G has more variational freedom than STO-3G, so its energy
        must be lower (more negative) for the same real geometry -- the
        actual point of this bridge: swapping basis_name needs no other
        code change. Real values, not invented."""
        pytest.importorskip("ase")
        from ase import Atoms
        from dense_evolution.qmmm.ase_bridge import DenseEvolutionCalculator

        positions = [[0, 0, 0], [0, 0, 0.7414]]
        e_sto3g = Atoms("H2", positions=positions)
        e_sto3g.calc = DenseEvolutionCalculator(atomic_numbers=[1, 1], nuclear_charges=[1.0, 1.0],
                                                 n_electrons=2, basis_name="sto-3g")
        e_631g = Atoms("H2", positions=positions)
        e_631g.calc = DenseEvolutionCalculator(atomic_numbers=[1, 1], nuclear_charges=[1.0, 1.0],
                                                n_electrons=2, basis_name="6-31g")
        assert e_631g.get_potential_energy() < e_sto3g.get_potential_energy()

    def test_missing_ase_raises_clear_error(self, monkeypatch):
        import builtins
        real_import = builtins.__import__

        def blocked_import(name, *args, **kwargs):
            if name == "ase.calculators.calculator":
                raise ImportError("simulated missing ase")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocked_import)
        from dense_evolution.qmmm.ase_bridge import _import_ase_calculator
        with pytest.raises(ImportError, match="qmmm.ase_bridge needs ASE"):
            _import_ase_calculator()
