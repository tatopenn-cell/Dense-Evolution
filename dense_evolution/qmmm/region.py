"""
Real QM/MM region-partitioning utilities (Dense-Evolution issue #283).
Validated on Dense-Evolution-Discovery on two molecules (1-hexanol,
OCC(c1ccccc1)CCC) before promotion here -- see that repo's
docs/qmmm_utils.md for the full experiment record.

Two real bugs already found and fixed:

1. Hydrogens must always follow their own heavy atom, never be
   independently BFS-expanded -- doing so spuriously cuts terminal C-H
   bonds and leaves an empty MM fragment (found on 1-hexanol).

2. A boundary bond that would cut INTO an aromatic ring must instead pull
   the WHOLE ring into the QM region -- otherwise RDKit's FragmentOnBonds
   leaves one aromatic atom outside its ring, capped with H, which is not
   a valid molecule (`AtomKekulizeException: non-ring atom marked
   aromatic`, found on OCC(c1ccccc1)CCC at radius=2).

Deliberately NOT included here, both real negative results with the full
record in Dense-Evolution-Discovery's docs/qmmm_bond_order_and_embedding.md:

- An MMFF94 ONIOM-style mechanical correction that originally looked like
  it halved 1-hexanol's error, but that used a cruder geometry (each
  fragment independently re-embedded by RDKit instead of sliced from one
  shared conformer). Once corrected to use the same shared-conformer
  geometry as everything here, plain truncation is already accurate to
  <0.2 kcal/mol on every radius tested on two different molecules -- and
  applying that correction on top makes it WORSE in every single case.
  The correction was compensating for a geometry artifact of an older
  embedding choice, not for truncation itself.

- Electrostatic embedding: five different treatments were tried (plain
  point charge, charge-shifting, Gaussian-smeared over the whole MM
  region, Gaussian-smeared on just the boundary atom, boundary-atom
  charge deletion) and every one made the isodesmic-energy error worse
  than no embedding at all, on the one molecule with a real MM charge to
  embed. The sign convention was independently verified correct (a
  minimal He-atom test: E(+1 charge nearby) < E(isolated) < E(-1 charge
  nearby), exactly as physics requires) -- the failure is not a bug, just
  genuinely unsolved. Plain truncation remains the right default.

Requires RDKit (the `qmmm` extra: `pip install dense-evolution[qmmm]`).
"""
import numpy as np
from rdkit import Chem

ANGSTROM_TO_BOHR = 1.8897259886
CH_BOND_BOHR = 1.09 * ANGSTROM_TO_BOHR


def partition_qm_mm_region(mol, seed_heavy_atoms, radius):
    """BFS outward from `seed_heavy_atoms` (a set/list of atom indices)
    across `radius` heavy-atom hops. Hydrogens always follow their own
    heavy atom afterward (never independently expanded -- see module
    docstring). Any boundary bond that would cut into an aromatic ring
    pulls that whole ring into the QM region first, iterating until
    stable, before hydrogens are added and the boundary is finalized.

    Returns (qm_atoms: set[int], boundary_pairs: list[(kept_idx, cut_idx)]),
    one pair per bond crossing the QM/MM boundary, `kept_idx` on the QM
    side.

    Examples
    --------
    >>> from rdkit import Chem
    >>> from dense_evolution.qmmm import partition_qm_mm_region
    >>> mol = Chem.AddHs(Chem.MolFromSmiles("OCCCCCC"))
    >>> qm_atoms, boundary_pairs = partition_qm_mm_region(mol, {0, 1}, radius=1)
    >>> len(boundary_pairs)
    1
    """
    qm_heavy = set(seed_heavy_atoms)
    frontier = set(seed_heavy_atoms)
    for _ in range(radius):
        new_frontier = set()
        for idx in frontier:
            for nbr in mol.GetAtomWithIdx(idx).GetNeighbors():
                if nbr.GetAtomicNum() > 1 and nbr.GetIdx() not in qm_heavy:
                    new_frontier.add(nbr.GetIdx())
        qm_heavy |= new_frontier
        frontier = new_frontier

    ring_info = mol.GetRingInfo()
    changed = True
    while changed:
        changed = False
        for bond in mol.GetBonds():
            if not bond.GetIsAromatic():
                continue
            a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if (a in qm_heavy) != (b in qm_heavy):
                for ring in ring_info.AtomRings():
                    if a in ring or b in ring:
                        newly = set(ring) - qm_heavy
                        if newly:
                            qm_heavy |= newly
                            changed = True

    qm_atoms = set(qm_heavy)
    for idx in qm_heavy:
        for nbr in mol.GetAtomWithIdx(idx).GetNeighbors():
            if nbr.GetAtomicNum() == 1:
                qm_atoms.add(nbr.GetIdx())

    raw_boundary = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()
                    if (b.GetBeginAtomIdx() in qm_atoms) != (b.GetEndAtomIdx() in qm_atoms)]
    boundary_pairs = [(i, j) if i in qm_atoms else (j, i) for i, j in raw_boundary]
    return qm_atoms, boundary_pairs


def sliced_geometry(atomic_numbers, geom_bohr, keep_idx, boundary_pairs):
    """A coordinate SUBSET of one whole-molecule conformer (never an
    independently re-embedded fragment -- that would give unrelated 3D
    structures across fragments, and was the real cause of the MMFF94
    correction's apparent benefit turning out to be a geometry artifact,
    see module docstring). Boundary bonds get a capping H placed along
    the kept->cut bond direction at a standard C-H bond length.

    Examples
    --------
    >>> import numpy as np
    >>> from dense_evolution.qmmm import sliced_geometry
    >>> numbers = [8, 6, 6]
    >>> geom = np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [2.8, 0.0, 0.0]])
    >>> new_numbers, new_geom = sliced_geometry(numbers, geom, {0, 1}, [(1, 2)])
    >>> new_numbers
    [8, 6, 1]
    """
    keep_idx = sorted(keep_idx)
    new_numbers = [atomic_numbers[i] for i in keep_idx]
    new_geom = [geom_bohr[i] for i in keep_idx]
    for kept, cut in boundary_pairs:
        vec = geom_bohr[cut] - geom_bohr[kept]
        vec = vec / np.linalg.norm(vec)
        new_geom.append(geom_bohr[kept] + vec * CH_BOND_BOHR)
        new_numbers.append(1)
    return new_numbers, np.array(new_geom)
