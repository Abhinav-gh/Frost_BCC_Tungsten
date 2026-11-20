"""Serve an MLIP's energy and forces over pipes from its own environment (client: ``frost.external_teacher``).

Runs inside ``.venvs/frost-flashmd`` (``pet:<mlip>``: FlashMD's reference energy model through ``flashmd.ase``'s
``EnergyCalculator``) or ``.venvs/frost-uma`` (``uma:<checkpoint>:<task>``: fairchem's ``FAIRChemCalculator`` on
``pretrained_mlip.get_predict_unit``) or ``.venvs/frost-lsd`` (``orb:<pretrained function>``: orb-models'
``ORBCalculator``) or ``.venvs/frost-mace`` (``mace:<teacher>``: our MACE teacher through ``frost.teachers.load_calc``,
as an oracle for tools in other environments, e.g. FLARE) or ``.venvs/frost-deepmd`` (``deepmd:<frozen model>``:
DeePMD-kit's ``deepmd.calculator.DP``) or ``.venvs/frost-kanda`` (``kanda:<frozen .pb>``: Kanda et al.'s released
TensorFlow fork, which ships no Python package, so the model is called through the documented C API of the library their
build installed, ``libdeepmd_c.so``, the same library their LAMMPS pair style links; no ASE) or ``.venvs/frost-nequip`` (``nequip:<compiled .nequip.pt2>``: nequip's
``NequIPCalculator.from_compiled_model``, for Allegro and NequIP models) or ``.venvs/frost-pyscf`` (``pyscf:<xc>/<basis>[/df]``:
Kohn-Sham DFT with analytic nuclear gradients on the GPU through GPU4PySCF's public ``gpu4pyscf.dft.rks.RKS`` and its gradient
scanner; a molecule in vacuum, the cell ignored; further_expansions_beyond.md 7.1) or ``.venvs/frost-ani`` (``ani:<model dir>``:
ANI-1xnr, the reactive C/H/N/O potential, built from its released NeuroChem files with TorchANI's public ``torchani.neurochem``
loaders, periodic; 7.4). Only the packages' public calculators are used.

Protocol: an 8-byte length and a JSON header (``numbers``, ``cell``, optional ``stress``) on stdin; the server answers
``OK``. Then each frame is n x 3 float64 positions in; one float64 energy and n x 3 float64 forces out, followed by the
six-component ASE stress (eV/A^3) when the header asked for it. Anything the libraries print goes to stderr, so stdout
carries only the protocol.
"""

from __future__ import annotations

import json
import os
import struct
import sys

import numpy as np

proto_in = sys.stdin.buffer
proto_out = os.fdopen(os.dup(1), "wb", buffering=0)
os.dup2(2, 1)  # library output from here on goes to stderr


def read_exact(n: int) -> bytes:
    out = bytearray()
    while len(out) < n:
        chunk = proto_in.read(n - len(out))
        if not chunk:
            return b""
        out.extend(chunk)
    return bytes(out)


SYMBOLS = {1: "H", 6: "C", 7: "N", 8: "O", 22: "Ti"}  # the elements of the type maps served so far


class DeepPotC:
    """A frozen DeePMD model through DeePMD-kit's public C API (``deepmd/c_api.h``) in a built ``libdeepmd_c.so``.

    Energies in eV and forces in eV/A (LAMMPS metal units, as ASE). The library builds its own neighbour list.
    """

    def __init__(self, model: str, numbers, cell, lib: str):
        import ctypes

        c = ctypes
        self.lib = c.CDLL(lib)
        self.lib.DP_NewDeepPot.restype = c.c_void_p
        self.lib.DP_NewDeepPot.argtypes = [c.c_char_p]
        self.lib.DP_DeepPotCheckOK.restype = c.c_char_p
        self.lib.DP_DeepPotCheckOK.argtypes = [c.c_void_p]
        self.lib.DP_DeepPotGetTypeMap.restype = c.c_char_p
        self.lib.DP_DeepPotGetTypeMap.argtypes = [c.c_void_p]
        self.lib.DP_DeepPotCompute.restype = None
        dbl, it = c.POINTER(c.c_double), c.POINTER(c.c_int)
        self.lib.DP_DeepPotCompute.argtypes = [c.c_void_p, c.c_int, dbl, it, dbl, dbl, dbl, dbl, c.c_void_p, c.c_void_p]
        self.dp = self.lib.DP_NewDeepPot(model.encode())
        err = self.lib.DP_DeepPotCheckOK(self.dp)
        if err:
            raise SystemExit(f"DP_NewDeepPot: {err.decode()}")
        type_map = self.lib.DP_DeepPotGetTypeMap(self.dp).decode().split()
        self.atype = np.ascontiguousarray([type_map.index(SYMBOLS[int(z)]) for z in numbers], dtype=np.int32)
        self.cell = np.ascontiguousarray(np.asarray(cell, dtype=np.float64).reshape(9))
        self.n, self.c = len(numbers), c
        self.force = np.zeros((self.n, 3), dtype=np.float64)
        self.virial = np.zeros(9, dtype=np.float64)
        self.energy = c.c_double(0.0)

    def __call__(self, positions: np.ndarray) -> tuple[float, np.ndarray]:
        c, dbl = self.c, self.c.POINTER(self.c.c_double)
        x = np.ascontiguousarray(positions, dtype=np.float64)
        self.lib.DP_DeepPotCompute(self.dp, self.n, x.ctypes.data_as(dbl), self.atype.ctypes.data_as(c.POINTER(c.c_int)),
                                   self.cell.ctypes.data_as(dbl), c.byref(self.energy), self.force.ctypes.data_as(dbl),
                                   self.virial.ctypes.data_as(dbl), None, None)
        return float(self.energy.value), self.force


class GPUDFT:
    """Kohn-Sham DFT (restricted, closed shell) energy and forces through GPU4PySCF's public API: ``gpu4pyscf.dft.rks.RKS``
    (optionally density-fitted) and ``nuc_grad_method().as_scanner()``, which starts each SCF from the previous geometry's
    density matrix, as AIMD codes do. A molecule in vacuum: the client's cell is a box large enough that no periodic image
    is within the student's cutoff, and DFT ignores it. Spec ``<xc>/<basis>[/df][/charge=Q][/grid=N][/nogridresp]``,
    e.g. ``pbe/def2-svp/df``. The gradient includes the grid response by default (2026-11-14): the quadrature grid moves
    with the atoms, and without its term the forces were not the energy's derivative (0.0839 against a finite difference of
    0.0797 eV/A on ethanol), which an NVE teacher would show as drift.
    """

    HARTREE_EV = 27.211386245988
    BOHR_A = 0.529177210903

    def __init__(self, spec: str, numbers):
        from ase.data import chemical_symbols

        parts = spec.split("/")
        self.xc, self.basis = parts[0], parts[1]
        self.df = "df" in parts[2:]
        self.charge = next((int(p.split("=")[1]) for p in parts[2:] if p.startswith("charge=")), 0)
        self.grid_level = next((int(p.split("=")[1]) for p in parts[2:] if p.startswith("grid=")), None)
        self.grid_response = "nogridresp" not in parts[2:]
        self.symbols = [chemical_symbols[int(z)] for z in numbers]
        self.scanner = None
        self.calls = 0

    def _build(self, positions):
        from gpu4pyscf.dft import rks
        from pyscf import gto

        mol = gto.M(atom=list(zip(self.symbols, positions.tolist(), strict=True)), basis=self.basis, unit="Angstrom",
                    charge=self.charge, spin=0, verbose=0)
        mf = rks.RKS(mol, xc=self.xc)
        if self.df:
            mf = mf.density_fit()
        mf.conv_tol = 1e-10
        if self.grid_level is not None:
            mf.grids.level = self.grid_level
        grad = mf.nuc_grad_method()
        grad.grid_response = self.grid_response  # the moving grid's term: forces = -dE/dx exactly
        self.scanner = grad.as_scanner()

    def __call__(self, positions: np.ndarray) -> tuple[float, np.ndarray]:
        if self.scanner is None:
            self._build(positions)
        e, g = self.scanner(positions)  # a geometry (Angstrom, the molecule's unit) in, Hartree and Hartree/Bohr out
        if not self.scanner.base.converged:
            print(f"pyscf: SCF not converged at call {self.calls}", file=sys.stderr, flush=True)
        self.calls += 1
        g = g.get() if hasattr(g, "get") else np.asarray(g)  # a CuPy array on the GPU path
        return float(e) * self.HARTREE_EV, np.ascontiguousarray(-g * self.HARTREE_EV / self.BOHR_A, dtype=np.float64)


class TorchANI:
    """An ANI model in NeuroChem format (ANI-1xnr: Zhang et al., Nat. Chem. 2024, github.com/atomistic-ml/ani-1xnr), built
    by TorchANI's public ``torchani.neurochem.load_model_from_info_file`` from the release's ``.info`` file (AEV
    parameters, self-atomic energies, the ensemble's network prefix and size: 8 members for ANI-1xnr); species as atomic
    numbers, periodic with the client's cell, forces by autograd. The spec is the ``.info`` file or the model directory
    beside it (``.../model/ani-1xnr`` -> ``.../model/ani-1xnr.info``). Energies in Hartree from the model, returned in eV."""

    HARTREE_EV = 27.211386245988

    def __init__(self, path: str, numbers, cell):
        import glob

        import torch
        import torchani

        info = path if path.endswith(".info") else (path + ".info" if os.path.exists(path + ".info")
                                                     else glob.glob(os.path.join(path, "*.info"))[0])
        # the info file's four lines (AEV parameters, self energies, ensemble prefix, ensemble size), resolved beside it:
        # TorchANI's own info-file parser resolves them inside its download directory instead, so the same public loaders
        # it calls are composed here with these paths
        with open(info) as fh:
            const_rel, sae_rel, prefix_rel, size = [line.strip() for line in fh.readlines()][:4]
        base = os.path.dirname(os.path.abspath(info))
        nc = torchani.neurochem
        # the compiled CUDA AEV (TorchANI's own `ani build-extensions`, built into its data directory) when present: the
        # strongest stack measured for the baseline (trap 17): 122 against 146 ms per call in float64 on 270 atoms; the
        # compiled cell lists were slower than all pairs at this size (142-175 ms)
        try:
            aev, symbols = nc.load_aev_computer_and_symbols(os.path.join(base, const_rel), strategy="cuaev")
            self.strategy = "cuaev"
        except Exception:  # the extension missing or unloadable: TorchANI's pure-PyTorch AEV
            aev, symbols = nc.load_aev_computer_and_symbols(os.path.join(base, const_rel))
            self.strategy = "pyaev"
        nets = nc.load_ensemble(symbols, os.path.join(base, prefix_rel), int(size))
        adder = nc.load_sae(os.path.join(base, sae_rel))
        self.dev = torch.device("cuda")
        # float64: the total energy is ~-58,600 eV for 270 atoms (almost all the atomic self energies), and float32 left
        # ~7 meV of it, below a finite difference's change (they read 0 against forces of 0.1-0.4 eV/A); in float64 the
        # forces match the energy's central differences to 5e-5 eV/A, at the same 150 ms per call (2026-11-14)
        self.model = torchani.arch.ANI(symbols, aev, nets, adder, periodic_table_index=True).to(self.dev).double()
        self.members = int(size)
        self.species = torch.as_tensor(np.asarray(numbers, dtype=np.int64), device=self.dev).unsqueeze(0)
        self.cell = torch.as_tensor(np.asarray(cell), dtype=torch.float64, device=self.dev)
        self.pbc = torch.ones(3, dtype=torch.bool, device=self.dev)
        self.torch = torch

    def __call__(self, positions: np.ndarray) -> tuple[float, np.ndarray]:
        torch = self.torch
        x = torch.as_tensor(positions, dtype=torch.float64, device=self.dev).unsqueeze(0).requires_grad_(True)
        _, e = self.model((self.species, x), cell=self.cell, pbc=self.pbc)
        (g,) = torch.autograd.grad(e.sum(), x)
        return float(e.item()) * self.HARTREE_EV, np.ascontiguousarray(-g[0].double().cpu().numpy() * self.HARTREE_EV)


def main() -> int:
    spec = sys.argv[1]
    kind, rest = spec.split(":", 1)
    header = json.loads(read_exact(struct.unpack("<Q", read_exact(8))[0]))
    numbers = np.asarray(header["numbers"])
    if kind in ("kanda", "pyscf", "ani"):
        if header.get("stress"):
            raise SystemExit(f"{kind}: no stress")
        if kind == "kanda":
            root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            pot = DeepPotC(rest, numbers, header["cell"], os.path.join(root, "third_party", "deepmd-mts", "install-gpu",
                                                                        "lib", "libdeepmd_c.so"))
        elif kind == "pyscf":
            pot = GPUDFT(rest, numbers)
        else:
            pot = TorchANI(rest, numbers, header["cell"])
        proto_out.write(b"OK")
        nbytes = 24 * len(numbers)
        while True:
            buf = read_exact(nbytes)
            if not buf:
                return 0
            energy, forces = pot(np.frombuffer(buf, dtype=np.float64).reshape(-1, 3))
            proto_out.write(struct.pack("<d", energy) + forces.tobytes())
    import ase

    atoms = ase.Atoms(numbers=numbers, positions=np.zeros((len(numbers), 3)), cell=header["cell"], pbc=True)
    if kind == "pet":
        from flashmd import get_pretrained
        from flashmd.ase import EnergyCalculator

        energy_model, _ = get_pretrained(rest, 16)  # the reference MLIP; the stride model is not used
        atoms.calc = EnergyCalculator(energy_model, device="cuda")
    elif kind == "uma":
        from fairchem.core import FAIRChemCalculator, pretrained_mlip

        name, task = rest.split(":")
        atoms.calc = FAIRChemCalculator(pretrained_mlip.get_predict_unit(name, device="cuda"), task_name=task)
    elif kind == "mace":
        from frost.teachers import load_calc

        atoms.calc = load_calc(rest, "cuda", "float32", enable_cueq=True)
    elif kind == "deepmd":
        from deepmd.calculator import DP

        atoms.calc = DP(model=rest)
    elif kind == "orb":
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.calculator import ORBCalculator

        atoms.calc = ORBCalculator(getattr(pretrained, rest)(device="cuda", precision="float32-high"), device="cuda")
    elif kind == "nequip":
        # a NequIP / Allegro model compiled by nequip-compile for ASE (e.g. Allegro-MP-L): nequip's own calculator
        from nequip.ase import NequIPCalculator

        atoms.calc = NequIPCalculator.from_compiled_model(rest, device="cuda")
    elif kind == "mtsallegro":
        # our MTS-Allegro rebuild's trained full Allegro model (experiments/e14_baselines/mts_allegro.py), on water
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from mts_allegro import AllegroCalculator

        atoms.calc = AllegroCalculator(rest or "allegro", device="cuda")
    else:
        raise SystemExit(f"unknown kind {kind!r}")
    stress = bool(header.get("stress", False))
    proto_out.write(b"OK")
    nbytes = 24 * len(numbers)
    while True:
        buf = read_exact(nbytes)
        if not buf:
            return 0
        atoms.set_positions(np.frombuffer(buf, dtype=np.float64).reshape(-1, 3))
        forces = np.ascontiguousarray(atoms.get_forces(), dtype=np.float64)
        energy = float(atoms.get_potential_energy())
        tail = np.asarray(atoms.get_stress(voigt=True), dtype=np.float64).tobytes() if stress else b""
        proto_out.write(struct.pack("<d", energy) + forces.tobytes() + tail)


if __name__ == "__main__":
    raise SystemExit(main())
