"""The simulation cells for E1 and the fidelity suite.

Every cell is constructed from a script rather than shipped as a file, which is
the reproducibility commitment in the proposal's artifact section.  Sizes match
the cell table in Frost.pdf.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import bulk, molecule

CACHE = Path(__file__).resolve().parents[2] / "data" / "systems"


# --------------------------------------------------------------------------
# E1 primary systems
# --------------------------------------------------------------------------
def si512() -> Atoms:
    """Diamond Si, 512 atoms.  Stiff, correlated -- predicted best case."""
    return bulk("Si", "diamond", a=5.43, cubic=True).repeat((4, 4, 4))


def cu500() -> Atoms:
    """fcc Cu, 500 atoms.  TorchSim's tutorial system, so baselines are trivial."""
    return bulk("Cu", "fcc", a=3.61, cubic=True).repeat((5, 5, 5))


def mgo512() -> Atoms:
    """Rocksalt MgO, 512 atoms: a held-out system for the certificate (gap C16), never used to tune it."""
    return bulk("MgO", "rocksalt", a=4.21, cubic=True).repeat((4, 4, 4))


def water216(density: float = 0.997, min_contact: float = 1.7) -> Atoms:
    """216 rigid H2O in a cubic box at ambient density (648 atoms).

    Molecules are placed on a jittered lattice with random orientation, and a
    placement is rejected if it puts any atom closer than ``min_contact`` to an
    atom of an already-placed molecule. Without that rejection the packing
    produces dozens of intermolecular contacts below 1 A. A short equilibration
    does anneal them away, because the potential is strongly repulsive there,
    but starting a trajectory from overlapping molecules is not something to
    rely on: it depends on the seed, and it fails silently.

    The default threshold of 1.7 A sits just below the O...H hydrogen bond at
    about 1.8 A, so it removes every repulsive contact without distorting the
    structure the liquid is supposed to have. Packing denser than that is not
    geometrically achievable on this lattice, and is unnecessary: equilibration
    builds the hydrogen-bond network.

    This is the hardest case for Frost: diffusive, hydrogen bonded, and
    neighbour identities turn over on picosecond timescales.
    """
    n = 216
    m_water = 18.01528 / 6.02214076e23  # g
    vol_cm3 = n * m_water / density
    L = (vol_cm3 * 1e24) ** (1 / 3)  # Angstrom

    rng = np.random.default_rng(0)
    mono = molecule("H2O")
    mono.set_positions(mono.get_positions() - mono.get_positions()[0])

    per_side = int(np.ceil(n ** (1 / 3)))
    spacing = L / per_side
    cells = [
        (i, j, k)
        for i in range(per_side)
        for j in range(per_side)
        for k in range(per_side)
    ]
    rng.shuffle(cells)

    placed = np.empty((0, 3))
    out = Atoms(cell=[L, L, L], pbc=True)
    for i, j, k in cells[:n]:
        origin0 = (np.array([i, j, k]) + 0.5) * spacing
        for _attempt in range(64):
            w = mono.copy()
            w.rotate(rng.uniform(0, 360), rng.normal(size=3), center=(0, 0, 0))
            w.translate(origin0 + rng.uniform(-0.15, 0.15, 3) * spacing)
            pos = w.get_positions()
            if placed.size:
                # minimum image against everything already placed
                delta = pos[:, None, :] - placed[None, :, :]
                delta -= L * np.round(delta / L)
                if np.linalg.norm(delta, axis=-1).min() < min_contact:
                    continue
            out += w
            placed = np.vstack([placed, pos])
            break
        else:  # pragma: no cover - only if the box is over-filled
            out += w
            placed = np.vstack([placed, w.get_positions()])
    out.wrap()
    return out


def silica1152() -> Atoms:
    """Beta-cristobalite SiO2, 1152 atoms: the seed for the melt-quench.

    Silicon sits on a diamond lattice and each oxygen bridges one Si-Si bond.
    The bridging oxygen must be placed at the **minimum-image** midpoint: taking
    the raw average of two fractional coordinates puts oxygens on the wrong side
    of the cell and lands several of them on top of each other, which produces a
    cell with coincident atoms and forces that are not finite. Finding the bonds
    with a neighbour list rather than hard-coding a pair list makes that class of
    mistake impossible.

    The production cell is a melt-quench from 3000 K at 1e12 K/s; this is the
    crystalline starting point for that quench.
    """
    from ase.neighborlist import neighbor_list

    a = 7.16  # beta-cristobalite lattice parameter
    si = bulk("Si", "diamond", a=a, cubic=True)  # 8 Si, the cation sublattice

    # Each Si-Si nearest-neighbour bond carries one bridging O at its midpoint.
    # Half the bond length of diamond at this lattice constant is ~1.55 A, the
    # right Si-O distance for cristobalite.
    i, j, D = neighbor_list("ijD", si, 0.5 * a)  # first shell only
    seen, mid = set(), []
    for a_i, b_i, d in zip(i, j, D, strict=True):
        key = (min(a_i, b_i), max(a_i, b_i), tuple(np.round(np.abs(d), 4)))
        if key in seen:
            continue
        seen.add(key)
        mid.append(si.positions[a_i] + 0.5 * d)  # D is the min-image vector

    cell = Atoms(
        f"Si{len(si)}O{len(mid)}",
        positions=np.vstack([si.positions, np.asarray(mid)]),
        cell=si.cell,
        pbc=True,
    )
    cell.wrap()
    return cell.repeat((4, 4, 3))


def silica_glass1152() -> Atoms:
    """Amorphous SiO2 glass, 1152 atoms, from experiments/e1_redundancy/quench_silica.py.

    This is the system E1 actually specifies. The crystalline cristobalite
    seed above is a different material and is kept only as the quench input.
    Raises RuntimeError, not FileNotFoundError, when the quench has not been
    run, so callers that skip unavailable systems handle it uniformly.
    """
    from ase.io import read

    path = CACHE / "silica_glass1152.xyz"
    if not path.exists():
        raise RuntimeError(
            f"no quenched glass at {path}; run experiments/e1_redundancy/quench_silica.py"
        )
    return read(path)


def li3po4(
    mp_id: str = "mp-13725", supercell: tuple[int, int, int] = (4, 4, 3)
) -> Atoms:
    """Li3PO4 solid electrolyte from the Materials Project.

    mp-13725 is the Pmn2_1 ground state: 16 sites, on the convex hull.  The
    default supercell gives 768 atoms, which is the scale at which the
    message-passing pruning study reports per-atom step times, so our numbers
    are externally checkable.

    Needs MP_API_KEY on the first call only; the CIF is cached afterwards.
    """
    import os

    cif = CACHE / f"{mp_id}.cif"
    if cif.exists():
        from ase.io import read

        return read(cif) * supercell

    key = os.environ.get("MP_API_KEY")
    if not key:
        raise RuntimeError(
            f"MP_API_KEY not set and no cached CIF at {cif}. "
            "Get a free key at next-gen.materialsproject.org, or drop a CIF there."
        )
    from mp_api.client import MPRester
    from pymatgen.io.ase import AseAtomsAdaptor

    with MPRester(key) as mpr:
        struct = mpr.get_structure_by_material_id(mp_id)
    CACHE.mkdir(parents=True, exist_ok=True)
    struct.to(filename=str(cif))
    return AseAtomsAdaptor.get_atoms(struct) * supercell


# --------------------------------------------------------------------------
# Scaling sweep for E26 (phi should fall as N^-alpha)
# --------------------------------------------------------------------------
def si_scaling(n_cells: int, defect: str | None = None) -> Atoms:
    """Diamond Si at a chosen size with a controlled hot region.

    defect=None      -> uniform bulk, Nhot = 0
    defect="vacancy" -> point defect, Nhot = O(1),      predicted alpha = 1
    defect="surface" -> free surface, Nhot = O(N^2/3),  predicted alpha = 1/3
    """
    at = bulk("Si", "diamond", a=5.43, cubic=True).repeat((n_cells,) * 3)
    if defect == "vacancy":
        del at[len(at) // 2]
    elif defect == "surface":
        c = at.cell.array.copy()
        c[2, 2] += 15.0  # vacuum slab
        at.set_cell(c)
        at.center(axis=2)
    return at


def tio2_anatase600() -> Atoms:
    """Anatase TiO2 (I4_1/amd; a 3.785, c 9.514 A, O at z 0.2081), 5 x 5 x 2 conventional cells: 600 atoms. A held-out
    system for the frozen certificate (2026-11-11), and the material of Kanda et al.'s benchmark."""
    from ase.spacegroup import crystal

    cell = crystal(["Ti", "O"], basis=[(0, 0, 0), (0, 0, 0.2081)], spacegroup=141,
                   cellpar=[3.785, 3.785, 9.514, 90, 90, 90])
    return cell.repeat((5, 5, 2))


def nacl512() -> Atoms:
    """Rock-salt NaCl at its 300 K lattice constant (5.64 A), 4^3 conventional cells: 512 atoms. Held out (2026-11-11)."""
    return bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat((4, 4, 4))


def nacl_liquid368() -> Atoms:
    """NaCl at the molten density near 1200 K (~1.55 g/cm^3): the 512-atom rock-salt box (a 5.64 A) with 72 Na-Cl pairs
    removed at random (seed 0), 368 atoms. Bonds keep their length; the lower density comes from fewer neighbours, as
    in the melt. Melted by its trajectory (2000 K, then 1200 K). Held out (2026-11-11): an ionic liquid with diffusion."""
    at = nacl512()
    rng = np.random.default_rng(0)
    z = at.get_atomic_numbers()
    drop = np.concatenate([rng.choice(np.flatnonzero(z == 11), 72, replace=False),
                           rng.choice(np.flatnonzero(z == 17), 72, replace=False)])
    del at[sorted(drop.tolist())]
    return at


E1_SYSTEMS = {
    "si512": si512,
    "cu500": cu500,
    "water216": water216,
    "silica1152": silica1152,
    "silica_glass1152": silica_glass1152,
    "li3po4": li3po4,
    "mgo512": mgo512,
    "tio2_anatase600": tio2_anatase600,
    "nacl512": nacl512,
    "nacl_liquid368": nacl_liquid368,
}



# --- Structure families added 2026-11-12 (gaps.md E.8): held-out classes for the frozen certificate ----------------
# Lattice constants are room-temperature experimental values unless stated; the teacher (MACE-MP-0 medium) relaxes
# them slightly under NVT, as for every earlier system.
BCC_A = {"W": 3.165, "Mo": 3.147, "Ta": 3.303, "Nb": 3.300, "Fe": 2.866, "V": 3.024, "Cr": 2.885,
         "Li": 3.510, "Na": 4.291, "K": 5.328}


def bcc(symbol: str, n: int = 8, a: float | None = None) -> Atoms:
    """A BCC metal, n^3 cubic cells (2 n^3 atoms)."""
    return bulk(symbol, "bcc", a=a or BCC_A[symbol], cubic=True).repeat((n, n, n))


def bcc_vacancy(symbol: str, n: int = 8) -> Atoms:
    """One vacancy in a BCC metal (2 n^3 - 1 atoms): the site at the cell's centre is removed."""
    at = bcc(symbol, n)
    centre = at.cell.array.sum(0) / 2
    del at[int(np.argmin(np.linalg.norm(at.get_positions() - centre, axis=1)))]
    return at


def bcc_dumbbell(symbol: str, n: int = 8, direction=(1, 1, 1), frac: float = 0.4) -> Atoms:
    """One self-interstitial in a BCC metal (2 n^3 + 1 atoms): the centre site replaced by a dumbbell along
    ``direction``, each atom ``frac`` of a nearest-neighbour distance from the site. <111> for W (the crowdion /
    <111> dumbbell is its ground state), <110> for Fe; the run relaxes the exact geometry."""
    at = bcc(symbol, n)
    centre = at.cell.array.sum(0) / 2
    i = int(np.argmin(np.linalg.norm(at.get_positions() - centre, axis=1)))
    x = at.positions[i].copy()
    u = np.asarray(direction, dtype=float)
    u /= np.linalg.norm(u)
    h = frac * BCC_A[symbol] * np.sqrt(3) / 2
    at.positions[i] = x + h * u
    at += Atoms(symbol, positions=[x - h * u])
    return at


def screw_dipole_field(xy: np.ndarray, cores, b: float, lx: float, ly: float, images: int = 30) -> np.ndarray:
    """u_z at in-plane points xy (n, 2) of straight screw dislocations along z at cores [(x, y, sign)], each the isotropic
    field (b / 2 pi) theta, summed over a symmetric square of periodic images (+-images cells in x and y). For a dipole
    the sum converges conditionally to a periodic field plus a linear term, which the cell's z-tilts absorb
    (bcc_screw_dipole measures them). Values carry jumps of exactly +-b across each core's branch cut, which leave a
    crystal with period b along z unchanged."""
    u = np.zeros(len(xy))
    shifts = np.arange(-images, images + 1)
    for xc, yc, sign in cores:
        for i in shifts:
            dx = xy[:, 0] - (xc + i * lx)
            for j in shifts:
                u += sign * np.arctan2(xy[:, 1] - (yc + j * ly), dx)
    return u * b / (2.0 * np.pi)


def bcc_screw_dipole(symbol: str = "W", nx: int = 8, ny: int = 13, nz: int = 6, a: float | None = None,
                     images: int = 30, stress_free: bool = False) -> Atoms:
    """Two opposite a/2<111> screw dislocations in a periodic BCC crystal (2026-11-18, further_expansions_beyond.md 7.5).

    x [11-2], y [-110], z [111] (ASE's oriented cell: 6 atoms, a sqrt6 x a sqrt2 x b); the cores sit at the centroids of
    three neighbouring [111] columns near (lx / 4, ly / 4) and (3 lx / 4, 3 ly / 4), the quadrupolar arrangement that
    minimises the image stress (Bulatov and Cai 2006). Atoms move along z by the image-summed isotropic field; the cell
    vectors x and y get the z-tilts that make the field periodic, measured from the field itself (the mean jump across
    one cell, modulo b), so no sign convention is hand-derived. matscipy's builder cannot be imported here (it needs an
    older ASE), so this is ours; the run relaxes the core.

    ``stress_free`` (2026-11-18, the fix): the symmetric image sum is periodic only up to a linear term, +0.708 A per cell
    along x and -0.654 A along y for w_screw3744 (measured without folding by b). The default absorbs that term into the
    cell's z-tilts, which leaves a homogeneous shear strain (the teacher: -0.79 / +0.68 GPa xz / yz). With ``stress_free``
    the term is subtracted and the dipole's plastic distortion is put in instead, as the cell tilts (b dy / ly, -b dx /
    lx) with the matching linear field. The default is kept for the trajectories already built from it."""
    from ase.lattice.cubic import BodyCenteredCubic

    a = a or BCC_A[symbol]
    at = BodyCenteredCubic(directions=[[1, 1, -2], [-1, 1, 0], [1, 1, 1]], size=(nx, ny, nz), symbol=symbol,
                           latticeconstant=a, pbc=True)
    lx, ly, _ = at.cell.lengths()
    b = a * np.sqrt(3.0) / 2.0
    pos = at.get_positions()
    cols = np.unique(np.round(pos[:, :2] % [lx, ly], 3), axis=0)

    def core_near(fx: float, fy: float) -> tuple[float, float]:
        target = np.array([fx * lx, fy * ly])
        order = np.argsort(np.linalg.norm(cols - target, axis=1))
        c0 = cols[order[0]]
        rest = cols[order[1:12]]
        near = rest[np.argsort(np.linalg.norm(rest - c0, axis=1))]
        # the triangle of mutually nearest columns: c0, its nearest neighbour, and their common nearest neighbour
        c1 = near[0]
        third = near[1:][np.argsort(np.linalg.norm(near[1:] - c0, axis=1) + np.linalg.norm(near[1:] - c1, axis=1))[0]]
        return tuple((c0 + c1 + third) / 3.0)

    (xa, ya), (xb, yb) = core_near(0.25, 0.25), core_near(0.75, 0.75)
    cores = [(xa, ya, +1), (xb, yb, -1)]
    pos[:, 2] += screw_dipole_field(pos[:, :2], cores, b, lx, ly, images)
    rng = np.random.default_rng(0)
    probe = rng.random((64, 2)) * [lx, ly]
    t = []
    for shift in ((lx, 0.0), (0.0, ly)):
        d = screw_dipole_field(probe + shift, cores, b, lx, ly, images) - screw_dipole_field(probe, cores, b, lx, ly,
                                                                                             images)
        t.append(float(np.mean(d % b)) if not stress_free else float(np.mean((d + b / 2) % b - b / 2)))
    cell = at.cell.array.copy()
    if stress_free:
        # the image sum's linear term out, then the dipole's plastic distortion in: the cut between the cores slips the
        # band by b, a mean distortion b dy / (lx ly) along x and -b dx / (lx ly) along y, which the cell takes as tilts
        # (Bulatov and Cai 2006). Measured (2026-11-18, the teacher's stress, one call per cell): the tilts holding the
        # linear term gave shear -0.79 / +0.68 GPa (xz / yz); the linear term out with no tilt, -1.49 / +1.47 (so the
        # first "fix", no tilt, was wrong); linear between them, zero stress lands at these plastic tilts within 0.06 A.
        tp = (b * (yb - ya) / ly, -b * (xb - xa) / lx)
        pos[:, 2] += (tp[0] - t[0]) * pos[:, 0] / lx + (tp[1] - t[1]) * pos[:, 1] / ly
        cell[0, 2] += tp[0]
        cell[1, 2] += tp[1]
    else:
        cell[0, 2] += t[0]
        cell[1, 2] += t[1]
    out = Atoms(at.get_chemical_symbols(), positions=pos, cell=cell, pbc=True)
    out.info["screw_cores"] = [(xa, ya, 1), (xb, yb, -1)]
    out.info["burgers"] = b
    out.info["linear_term"] = t if stress_free else None
    out.info["plastic_tilts"] = (float(cell[0, 2]), float(cell[1, 2])) if stress_free else None
    return out


def bcc_random_alloy(symbols, n: int = 8, seed: int = 0) -> Atoms:
    """An equiatomic random BCC alloy (2 n^3 atoms), the lattice constant the mean of the elements' (Vegard)."""
    a = float(np.mean([BCC_A[s] for s in symbols]))
    at = bulk("W", "bcc", a=a, cubic=True).repeat((n, n, n))
    labels = np.array([symbols[i % len(symbols)] for i in range(len(at))])
    np.random.default_rng(seed).shuffle(labels)
    at.set_chemical_symbols(labels.tolist())
    return at


def hcp_metal(symbol: str, a: float, c: float, reps=(8, 5, 6)) -> Atoms:
    """An HCP metal in its orthorhombic 4-atom cell, repeated."""
    return bulk(symbol, "hcp", a=a, c=c, orthorhombic=True).repeat(reps)


def cu111_slab() -> Atoms:
    """Cu(111), 8 x 8 surface cells x 8 layers (512 atoms), 10 A of vacuum on each side; periodic in all three axes
    (the 20 A gap is far beyond the 6 A cutoff), so the same code paths run it."""
    from ase.build import fcc111

    at = fcc111("Cu", size=(8, 8, 8), a=3.615, vacuum=10.0, orthogonal=True)
    at.pbc = True
    return at


def srtio3_625() -> Atoms:
    """Cubic perovskite SrTiO3 (a 3.905 A), 5^3 cells: 625 atoms."""
    from ase.spacegroup import crystal

    return crystal(["Sr", "Ti", "O"], basis=[(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0)], spacegroup=221,
                   cellpar=[3.905, 3.905, 3.905, 90, 90, 90]).repeat((5, 5, 5))


def agi_alpha864(n: int = 6, a: float = 5.07, seed: int = 0) -> Atoms:
    """Superionic alpha-AgI: iodide on a BCC lattice (a 5.07 A, its value above 420 K), one Ag per I on randomly
    chosen tetrahedral (12d) sites with no two Ag closer than 2.5 A; n^3 cells, 864 atoms. Run hot (600 K)."""
    from ase.spacegroup import crystal

    iod = bulk("I", "bcc", a=a, cubic=True).repeat((n, n, n))
    tet = crystal("Ag", basis=[(0.25, 0.0, 0.5)], spacegroup=229, cellpar=[a, a, a, 90, 90, 90])
    sites = np.concatenate([tet.get_positions() + a * np.array([i, j, k])
                            for i in range(n) for j in range(n) for k in range(n)])
    rng = np.random.default_rng(seed)
    box = a * n
    chosen = []
    for idx in rng.permutation(len(sites)):
        x = sites[idx]
        if chosen:
            d = np.asarray(chosen) - x
            d -= box * np.round(d / box)
            if (np.linalg.norm(d, axis=1) < 2.5).any():
                continue
        chosen.append(x)
        if len(chosen) == len(iod):
            break
    if len(chosen) < len(iod):
        raise RuntimeError("could not place one Ag per I on separated tetrahedral sites")
    return iod + Atoms("Ag" * len(chosen), positions=np.asarray(chosen))


DEPTH_SYSTEMS = {
    "w1024": lambda: bcc("W"), "mo1024": lambda: bcc("Mo"), "ta1024": lambda: bcc("Ta"),
    "nb1024": lambda: bcc("Nb"), "fe1024": lambda: bcc("Fe"),
    "li1024": lambda: bcc("Li"), "na1024": lambda: bcc("Na"), "k1024": lambda: bcc("K"),
    "w_vac1023": lambda: bcc_vacancy("W"), "w_sia1025": lambda: bcc_dumbbell("W", direction=(1, 1, 1)),
    "w_screw3744": lambda: bcc_screw_dipole("W"),
    "w_screw3744s": lambda: bcc_screw_dipole("W", stress_free=True),
    "fe_vac1023": lambda: bcc_vacancy("Fe"), "fe_sia1025": lambda: bcc_dumbbell("Fe", direction=(1, 1, 0)),
    "wmo1024": lambda: bcc_random_alloy(["W", "Mo"]), "nbmotaw1024": lambda: bcc_random_alloy(["Nb", "Mo", "Ta", "W"]),
    "ti_hcp960": lambda: hcp_metal("Ti", 2.951, 4.686), "mg_hcp800": lambda: hcp_metal("Mg", 3.209, 5.211, (8, 5, 5)),
    "ti_beta1024": lambda: bcc("Ti", a=3.31),
    "cu111_slab512": cu111_slab, "srtio3_625": srtio3_625, "agi_alpha864": agi_alpha864,
}

E1_SYSTEMS.update(DEPTH_SYSTEMS)


# --- Molecules in water (2026-11-13, further_expansions_beyond.md 6.2): DMTS-NC's own class, with the MACE-OFF teacher ----
# Ac-Ala-NHMe (alanine dipeptide, L), C7eq conformer (phi -80.6, psi 78.1): RDKit ETKDG + MMFF94, lowest of 60 embeddings,
# centred. Heavy atoms in SMILES order CC(=O)N[C@@H](C)C(=O)NC: C0 C1 O2 N3 CA4 CB5 C6 O7 N8 C9, then 12 H.
# Backbone dihedrals: phi = C1-N3-CA4-C6, psi = N3-CA4-C6-N8 (ALA2_PHI, ALA2_PSI).
ALA2 = (
    ("C", 2.2572, 0.2976, 2.0701), ("C", 1.6080, 0.3187, 0.7136), ("O", 1.7060, 1.2749, -0.0505),
    ("N", 0.8657, -0.8008, 0.4222), ("C", 0.1016, -0.8974, -0.8220), ("C", -0.1547, -2.3591, -1.1650),
    ("C", -1.2240, -0.1254, -0.6519), ("O", -2.2907, -0.6708, -0.3811), ("N", -1.0791, 1.2443, -0.7628),
    ("C", -2.1702, 2.1375, -0.4778), ("H", 2.2934, -0.7159, 2.4795), ("H", 3.2823, 0.6679, 1.9825),
    ("H", 1.6890, 0.9382, 2.7493), ("H", 0.5416, -1.3640, 1.1994), ("H", 0.6747, -0.4327, -1.6335),
    ("H", -0.7046, -2.8700, -0.3664), ("H", 0.7898, -2.8944, -1.3098), ("H", -0.7433, -2.4464, -2.0844),
    ("H", -0.1261, 1.6085, -0.7065), ("H", -1.9350, 3.1188, -0.8960), ("H", -3.0965, 1.7560, -0.9155),
    ("H", -2.2854, 2.2146, 0.6066),
)
ALA2_PHI, ALA2_PSI = (1, 3, 4, 6), (3, 4, 6, 8)


def solvated(solute: Atoms, box: float, n_water: int, min_contact: float = 1.7, solute_contact: float = 2.0,
             seed: int = 0) -> Atoms:
    """``solute`` at the centre of a cubic box filled with ``n_water`` rigid H2O, as ``water216`` packs them.

    Water goes on a shuffled, jittered lattice with random orientations; a placement is rejected when any of its atoms
    is closer than ``min_contact`` to an earlier water or ``solute_contact`` to the solute (minimum image), and a lattice
    site the solute blocks is skipped. The solute's atoms come first, so their indices are the solute's own.
    """
    rng = np.random.default_rng(seed)
    out = solute.copy()
    out.set_cell([box, box, box])
    out.pbc = True
    out.translate(0.5 * box - out.get_positions().mean(0))
    sol = out.get_positions()
    mono = molecule("H2O")
    mono.set_positions(mono.get_positions() - mono.get_positions()[0])
    per_side = int(np.ceil((1.1 * n_water) ** (1 / 3)))
    spacing = box / per_side
    sites = [(i, j, k) for i in range(per_side) for j in range(per_side) for k in range(per_side)]
    rng.shuffle(sites)

    def clearance(pos, others):
        if not len(others):
            return np.inf
        d = pos[:, None, :] - others[None, :, :]
        d -= box * np.round(d / box)
        return np.linalg.norm(d, axis=-1).min()

    placed = np.empty((0, 3))
    for site in sites:
        if len(placed) == 3 * n_water:
            break
        origin = (np.array(site) + 0.5) * spacing
        for _attempt in range(64):
            w = mono.copy()
            w.rotate(rng.uniform(0, 360), rng.normal(size=3), center=(0, 0, 0))
            w.translate(origin + rng.uniform(-0.15, 0.15, 3) * spacing)
            pos = w.get_positions()
            if clearance(pos, sol) >= solute_contact and clearance(pos, placed) >= min_contact:
                out += w
                placed = np.vstack([placed, pos])
                break
    if len(placed) < 3 * n_water:
        raise RuntimeError(f"placed {len(placed) // 3} of {n_water} waters; enlarge the box")
    out.wrap()
    return out


def ala2_water() -> Atoms:
    """Alanine dipeptide in 454 waters, a 24 A cube at 0.997 g/cm^3 for the water (1,384 atoms).

    454 = (24^3 - 233) / 29.91: the box minus the dipeptide's ~233 A^3 (molar volume ~140 cm^3/mol), at ambient water's
    29.91 A^3 per molecule. Periodic images of the solute are >= 17 A apart, beyond MACE-OFF's two-layer reach (9-10 A).
    """
    solute = Atoms([a[0] for a in ALA2], positions=[a[1:] for a in ALA2])
    return solvated(solute, 24.0, 454)


MOLECULE_SYSTEMS = {"ala2_water": ala2_water}
E1_SYSTEMS.update(MOLECULE_SYSTEMS)


# --- Gas-phase molecules (2026-11-14, further_expansions_beyond.md 7.1: the DFT teacher) -----------------------------
# One molecule centred in a periodic box large enough that no periodic image is within any student's cutoff plus skin
# (the DFT teacher, ``pyscf:``, treats it as isolated and ignores the box). Not a condensed phase: tests/unit/
# test_systems.py skips its density and coordination checks (GAS_SYSTEMS), never the coincident-atom one.
def molecule_box(name: str, box: float = 20.0) -> Atoms:
    at = molecule(name)
    at.set_cell([box, box, box])
    at.pbc = True
    at.translate(0.5 * box - at.get_positions().mean(0))
    return at


def ala2_gas(n_side: int = 4, spacing: float = 80.0, seed: int = 0) -> Atoms:
    """n_side^3 alanine dipeptides in the gas phase, each alone: centres on a cubic grid ``spacing`` apart, random
    orientations, each copy's 22 atoms contiguous (copy c holds atoms 22c .. 22c + 21, so ALA2_PHI / ALA2_PSI + 22c index
    its backbone). The BoostMD comparison's system (2026-11-14): BoostMD's own stability and free-energy test is this
    molecule in the gas phase with MACE-OFF23-M, and many copies make one GPU call worth timing and give pooled dihedral
    statistics. At 80 A spacing two copies stay > 60 A apart over 50 ps of Langevin diffusion at 1/ps (per-component
    spread ~13 A per copy), far beyond MACE-OFF's 5 A cutoff, so no copy ever sees another."""
    rng = np.random.default_rng(seed)
    mol = Atoms([a[0] for a in ALA2], positions=[a[1:] for a in ALA2])
    mol.translate(-mol.get_positions().mean(0))
    box = n_side * spacing
    out = Atoms(cell=[box, box, box], pbc=True)
    for c in range(n_side ** 3):
        m = mol.copy()
        m.rotate(rng.uniform(0, 360), rng.normal(size=3), center=(0, 0, 0))
        m.translate((np.array([c % n_side, (c // n_side) % n_side, c // (n_side * n_side)]) + 0.5) * spacing)
        out += m
    return out


GAS_SYSTEMS = {"ethanol_box": lambda: molecule_box("CH3CH2OH"), "ala2_gas64": ala2_gas}
E1_SYSTEMS.update(GAS_SYSTEMS)


# --- An interface with two spectra (2026-11-14, further_expansions_beyond.md 7.3: spatial multirate) ----------------
def cu111_water(layers: int = 24, film: float = 15.0, vacuum: float = 12.0, seed: int = 0) -> Atoms:
    """A water film on Cu(111): 6 x 6 surface cells x ``layers`` (24: 864 Cu, ~50 A, so a deep slow region exists more
    than MACE-MP-0's 24 A exact-force buffer below the interface), water at 0.997 g/cm^3 in a ``film``-thick layer 2.5 A
    above the top Cu layer, then ``vacuum`` (the periodic image of the slab's bottom is beyond any cutoff). Water's fastest
    period is ~9 fs (~15 fs at 4 amu hydrogens), copper's ~110 fs: one global interval must obey water's."""
    from ase.build import fcc111

    slab = fcc111("Cu", size=(6, 6, layers), a=3.615, vacuum=0.0, orthogonal=True)
    lx, ly = float(slab.cell[0, 0]), float(slab.cell[1, 1])
    z_top = float(slab.positions[:, 2].max())
    z0 = z_top + 2.5
    n_water = round(film * lx * ly / 29.91)
    rng = np.random.default_rng(seed)
    mono = molecule("H2O")
    mono.set_positions(mono.get_positions() - mono.get_positions()[0])
    nx, ny = max(1, round(lx / 3.1)), max(1, round(ly / 3.1))
    nz = int(np.ceil(1.15 * n_water / (nx * ny)))
    sites = [(i, j, k) for i in range(nx) for j in range(ny) for k in range(nz)]
    rng.shuffle(sites)
    cell = np.diag([lx, ly, z0 + film + vacuum])
    placed = np.empty((0, 3))
    cu = slab.get_positions()
    out = slab.copy()
    out.set_cell(cell)
    out.pbc = True

    def clearance(pos, others):
        if not len(others):
            return np.inf
        d = pos[:, None, :] - others[None, :, :]
        d[..., :2] -= np.array([lx, ly]) * np.round(d[..., :2] / np.array([lx, ly]))
        return np.linalg.norm(d, axis=-1).min()

    for i, j, k in sites:
        if len(placed) == 3 * n_water:
            break
        origin = np.array([(i + 0.5) * lx / nx, (j + 0.5) * ly / ny, z0 + (k + 0.5) * film / nz])
        for _attempt in range(64):
            w = mono.copy()
            w.rotate(rng.uniform(0, 360), rng.normal(size=3), center=(0, 0, 0))
            w.translate(origin + rng.uniform(-0.15, 0.15, 3) * np.array([lx / nx, ly / ny, film / nz]))
            pos = w.get_positions()
            if pos[:, 2].min() < z0 - 0.5 or pos[:, 2].max() > z0 + film + 0.5:
                continue
            if clearance(pos, placed) >= 1.7 and clearance(pos, cu) >= 2.2:
                out += w
                placed = np.vstack([placed, pos])
                break
    if len(placed) < 3 * n_water:
        raise RuntimeError(f"placed {len(placed) // 3} of {n_water} waters")
    return out


# the wide-interface test (2026-11-15): 72 layers (~150 A of Cu) and 30 A of vacuum, so MACE-MP-0's exact cut-out (the
# water, the top layers and a 24 A buffer) is ~28% of the cell; in the 24-layer cell it was 97%, reaching the slab's
# bottom through the periodic image across 12 A of vacuum
INTERFACE_SYSTEMS = {"cu111_water": cu111_water, "cu111_water72": lambda: cu111_water(layers=72, vacuum=30.0)}
E1_SYSTEMS.update(INTERFACE_SYSTEMS)


# --- Reactive chemistry (2026-11-14, further_expansions_beyond.md 7.4: bonds that break) ----------------------------
def molecular_mixture(species_counts: dict, density: float, seed: int = 0, min_contact: float = 2.0) -> Atoms:
    """Molecules (ASE names -> counts) at random positions and orientations in a cubic box at ``density`` (g/cm^3), no
    two atoms of different molecules closer than ``min_contact`` (minimum image)."""
    rng = np.random.default_rng(seed)
    mols = [name for name, k in species_counts.items() for _ in range(k)]
    rng.shuffle(mols)
    mass = sum(molecule(m).get_masses().sum() for m in mols)
    box = (mass * 1.66053907e-24 / density * 1e24) ** (1 / 3)
    out = Atoms(cell=[box, box, box], pbc=True)
    placed = np.empty((0, 3))
    for name in mols:
        mono = molecule(name)
        mono.translate(-mono.get_center_of_mass())
        for _attempt in range(2000):
            w = mono.copy()
            w.rotate(rng.uniform(0, 360), rng.normal(size=3), center=(0, 0, 0))
            w.translate(rng.uniform(0, box, 3))
            pos = w.get_positions()
            if placed.size:
                d = pos[:, None, :] - placed[None, :, :]
                d -= box * np.round(d / box)
                if np.linalg.norm(d, axis=-1).min() < min_contact:
                    continue
            out += w
            placed = np.vstack([placed, pos])
            break
        else:
            raise RuntimeError(f"could not place {name}; lower the density")
    out.wrap()
    return out


REACTIVE_SYSTEMS = {"ch4_o2_270": lambda: molecular_mixture({"CH4": 30, "O2": 60}, 0.4)}
E1_SYSTEMS.update(REACTIVE_SYSTEMS)

if __name__ == "__main__":
    for name, fn in E1_SYSTEMS.items():
        try:
            at = fn()
            vol = at.get_volume()
            print(
                f"{name:<12} {len(at):>6} atoms  "
                f"rho={len(at) / vol:.4f} A^-3  cell={np.diag(at.cell.array).round(2)}"
            )
        except Exception as exc:
            print(f"{name:<12} SKIPPED: {exc}")
