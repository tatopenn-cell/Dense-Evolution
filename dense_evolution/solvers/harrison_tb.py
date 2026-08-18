"""
Harrison empirical tight-binding parameters -- builds an sp3 tight-binding
Hamiltonian for a cluster of real atoms directly from published atomic term
values and the universal bond-scaling law, with no SCF/DFT and no external
quantum-chemistry dependency (PySCF, OpenFermion).

Source: Walter A. Harrison, "Electronic Structure and the Properties of
Solids" (Dover reprint). Atomic term values (ELEMENTS) and the universal
eta coefficients (ETA) below are transcribed from that book's Solid State
Table, cross-checked against the numbers in jarvist/HarrisonSolidStateTable.jl
(github.com/jarvist/HarrisonSolidStateTable.jl), a Julia implementation of
the same table. Only elements with both an s and a p term value in that
table are included here (the "simple atom" sp3 entries Harrison uses for
tetrahedral semiconductors); d-block entries are left out since Harrison's
own table flags them "not well checked".

Sign convention: ELEMENTS stores true orbital energies (negative, eV --
bound states below vacuum), i.e. the negative of the magnitudes printed in
Harrison's table.

Two-center bond integrals follow Harrison's universal scaling law:
    V_ll'm = eta_ll'm * hbar^2 / (m_e * d^2)
with d the bond length in Angstrom and hbar^2/m_e = 7.62 eV*Angstrom^2 (the
standard constant quoted alongside this law). eta_ssσ/spσ/ppσ/ppπ are
dimensionless and materials-independent -- the same four numbers apply to
every element pair.

Slater-Koster sp3 matrix elements (sp3_bond_block) follow the standard
1954 Slater-Koster table for an (s, px, py, pz) basis.

zincblende_hamiltonian builds the periodic Bloch Hamiltonian for a
two-atom zinc-blende basis (nearest-neighbor sp3, 4 bonds per atom) --
validated against real GaAs (a=5.6533 Angstrom): computed direct gap
at Gamma is 2.91 eV vs. the experimental 1.42 eV, roughly 2x too
large. This is a known, documented limitation of Harrison's universal
(materials-independent) parameter set on polar/ionic compound
semiconductors -- not a bug here -- since it uses no per-material
fitting and omits d-orbitals. Useful as a fast, dependency-free
qualitative estimate; not a substitute for this project's DFT-derived
GaAs parameters where quantitative accuracy matters.
"""
import numpy as np

__all__ = [
    'ELEMENTS', 'ETA', 'HBAR2_OVER_M_EV_ANG2',
    'hopping_integral', 'sp3_bond_block', 'sp3_dimer_hamiltonian',
    'zincblende_hamiltonian',
]

# name -> (Z, eps_s [eV], eps_p [eV], atomic mass [amu])
# Values transcribed from Harrison's Solid State Table (magnitudes in the
# book are positive "term values"; stored here as negative orbital energies).
ELEMENTS = {
    'Be': dict(Z=4,  eps_s=-8.17,  eps_p=-4.14,  mass=9.01),
    'B':  dict(Z=5,  eps_s=-12.54, eps_p=-6.64,  mass=10.81),
    'C':  dict(Z=6,  eps_s=-17.52, eps_p=-8.97,  mass=12.01),
    'N':  dict(Z=7,  eps_s=-23.04, eps_p=-11.47, mass=14.01),
    'O':  dict(Z=8,  eps_s=-29.14, eps_p=-14.13, mass=16.00),
    'Mg': dict(Z=12, eps_s=-6.86,  eps_p=-2.99,  mass=24.31),
    'Si': dict(Z=14, eps_s=-13.55, eps_p=-6.52,  mass=28.09),
    'P':  dict(Z=15, eps_s=-17.10, eps_p=-8.33,  mass=30.97),
    'S':  dict(Z=16, eps_s=-20.80, eps_p=-10.27, mass=32.06),
    'Cu': dict(Z=29, eps_s=-6.92,  eps_p=-1.83,  mass=63.54),
    'Zn': dict(Z=30, eps_s=-8.40,  eps_p=-3.38,  mass=65.37),
    'Ga': dict(Z=31, eps_s=-11.37, eps_p=-4.90,  mass=69.82),
    'Ge': dict(Z=32, eps_s=-14.38, eps_p=-6.36,  mass=72.59),
    'As': dict(Z=33, eps_s=-17.33, eps_p=-7.91,  mass=74.92),
    'Se': dict(Z=34, eps_s=-20.32, eps_p=-9.53,  mass=78.96),
    'Sn': dict(Z=50, eps_s=-12.50, eps_p=-5.94,  mass=118.7),
    'I':  dict(Z=53, eps_s=-19.42, eps_p=-9.97,  mass=126.9),
    'Pb': dict(Z=82, eps_s=-12.07, eps_p=-5.77,  mass=207.2),
}

# Universal Harrison interatomic matrix element coefficients (dimensionless),
# from the Dover reprint's "top right table". Materials-independent.
ETA = dict(ss_sigma=-1.40, sp_sigma=1.84, pp_sigma=3.24, pp_pi=-0.81)

# hbar^2 / m_e in eV*Angstrom^2, the constant in Harrison's d^-2 scaling law.
HBAR2_OVER_M_EV_ANG2 = 7.62


def hopping_integral(eta, d_angstrom):
    """V_ll'm = eta * hbar^2/(m_e d^2) [eV], for bond length d in Angstrom."""
    if d_angstrom <= 0:
        raise ValueError(f"bond length must be positive, got {d_angstrom}")
    return eta * HBAR2_OVER_M_EV_ANG2 / d_angstrom ** 2


def sp3_bond_block(l, m, n, d_angstrom, eta=ETA):
    """
    4x4 Slater-Koster hopping block <A, {s,px,py,pz}| H |B, {s,px,py,pz}>
    for a bond from atom A to atom B along direction cosines (l, m, n)
    (unit vector, l^2+m^2+n^2 = 1) and bond length d_angstrom.

    Basis order: s, px, py, pz. Standard Slater-Koster (1954) table.
    """
    norm = l * l + m * m + n * n
    if not np.isclose(norm, 1.0, atol=1e-6):
        raise ValueError(f"(l, m, n) must be a unit vector, got norm={norm}")

    Vssσ = hopping_integral(eta['ss_sigma'], d_angstrom)
    Vspσ = hopping_integral(eta['sp_sigma'], d_angstrom)
    Vppσ = hopping_integral(eta['pp_sigma'], d_angstrom)
    Vppπ = hopping_integral(eta['pp_pi'], d_angstrom)

    block = np.zeros((4, 4), dtype=np.complex128)
    block[0, 0] = Vssσ
    block[0, 1], block[0, 2], block[0, 3] = l * Vspσ, m * Vspσ, n * Vspσ
    block[1, 0], block[2, 0], block[3, 0] = -l * Vspσ, -m * Vspσ, -n * Vspσ

    block[1, 1] = l * l * Vppσ + (1 - l * l) * Vppπ
    block[2, 2] = m * m * Vppσ + (1 - m * m) * Vppπ
    block[3, 3] = n * n * Vppσ + (1 - n * n) * Vppπ

    block[1, 2] = block[2, 1] = l * m * (Vppσ - Vppπ)
    block[2, 3] = block[3, 2] = m * n * (Vppσ - Vppπ)
    block[1, 3] = block[3, 1] = l * n * (Vppσ - Vppπ)
    return block


def sp3_dimer_hamiltonian(element_a, element_b, bond_length_angstrom,
                           direction=(0.0, 0.0, 1.0), eta=ETA):
    """
    8x8 sp3 tight-binding Hamiltonian for a 2-atom A-B cluster (one bond),
    basis order [A:s,px,py,pz, B:s,px,py,pz]. On-site blocks are each
    atom's diagonal (eps_s, eps_p, eps_p, eps_p); the A-B off-diagonal
    block is sp3_bond_block along `direction` (unit vector, default: bond
    along z), Hermitian-conjugated into the B-A block.

    This is a minimal, directly checkable sanity case (bonding/antibonding
    sp3 splitting), not a periodic-solid band structure.
    """
    for name in (element_a, element_b):
        if name not in ELEMENTS:
            raise ValueError(f"no Harrison sp term values for element {name!r}; "
                              f"available: {sorted(ELEMENTS)}")
    l, m, n = direction
    a, b = ELEMENTS[element_a], ELEMENTS[element_b]

    H = np.zeros((8, 8), dtype=np.complex128)
    H[0, 0] = a['eps_s']
    H[1, 1] = H[2, 2] = H[3, 3] = a['eps_p']
    H[4, 4] = b['eps_s']
    H[5, 5] = H[6, 6] = H[7, 7] = b['eps_p']

    hop = sp3_bond_block(l, m, n, bond_length_angstrom, eta=eta)
    H[0:4, 4:8] = hop
    H[4:8, 0:4] = hop.conj().T
    return H


def zincblende_hamiltonian(k, cation, anion, lattice_constant_angstrom, eta=ETA):
    """
    8x8 Bloch Hamiltonian for a zinc-blende crystal's two-atom basis
    (cation at (0,0,0), anion at (1/4,1/4,1/4) of the conventional cubic
    cell), sp3 nearest-neighbor tight-binding, at crystal momentum k
    (Cartesian, 1/Angstrom -- e.g. Gamma=(0,0,0)).

    Basis order [cation:s,px,py,pz, anion:s,px,py,pz]. The four
    cation->anion nearest-neighbor bonds are the standard zinc-blende
    tetrahedral set (lattice_constant/4)*(1,1,1), (1,-1,-1), (-1,1,-1),
    (-1,-1,1); each contributes sp3_bond_block(...) weighted by its
    Bloch phase exp(i k . d), summed into the off-diagonal block.
    """
    for name in (cation, anion):
        if name not in ELEMENTS:
            raise ValueError(f"no Harrison sp term values for element {name!r}; "
                              f"available: {sorted(ELEMENTS)}")
    k = np.asarray(k, dtype=float)
    d_vectors = (lattice_constant_angstrom / 4) * np.array([
        [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1],
    ], dtype=float)
    bond_length = np.linalg.norm(d_vectors[0])

    T = np.zeros((4, 4), dtype=np.complex128)
    for d in d_vectors:
        l, m, n = d / bond_length
        phase = np.exp(1j * np.dot(k, d))
        T += phase * sp3_bond_block(l, m, n, bond_length, eta=eta)

    c, a = ELEMENTS[cation], ELEMENTS[anion]
    H = np.zeros((8, 8), dtype=np.complex128)
    H[0, 0] = c['eps_s']
    H[1, 1] = H[2, 2] = H[3, 3] = c['eps_p']
    H[4, 4] = a['eps_s']
    H[5, 5] = H[6, 6] = H[7, 7] = a['eps_p']
    H[0:4, 4:8] = T
    H[4:8, 0:4] = T.conj().T
    return H
