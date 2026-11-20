"""BoostMD (Schaaf et al., arXiv 2412.18633) rebuilt from the paper: no public code, checkpoint or data exists (logged
search, ``docs/design/comparisons/hunt_boostmd.md``). Specification, numbered decisions for every unstated detail and
the pre-registered validation targets: ``experiments/e14_baselines/boostmd_spec.md`` (skill ``unreleased-baseline``).

Public pieces only: MACE-OFF23 medium (the reference model) through ``frost.teachers.load_calc`` (mace's own
``mace_off``), its node energies and node features from its own output dictionary, and SPICE's dipeptide structures
(``openmm/spice-dataset``, MIT, ``data/datasets/spice_dipeptides``).

Stages (frost-mace environment, GPU):
- ``md``: training data. Gas-phase dipeptides, many per MACE call (each molecule alone in a large box, edges only within
  a molecule), BAOAB Langevin 300 K, 1 fs, friction 0.01/fs, 2 ps equilibration then 10 ps; every 100 fs a reference
  frame (positions, the reference model's per-atom energies and scalar node features of both layers) and the 10 frames
  after it (positions, energy, forces): BoostMD's reference every N = 10 steps (their Fig. 3).
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import re
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "datasets" / "spice_dipeptides" / "dipeptides.hdf5"
BASE = Path(os.environ.get("BOOSTMD_BASE", ROOT / "data" / "baselines" / "boostmd"))  # override: smoke tests only
RES = Path(os.environ.get("BOOSTMD_RES", ROOT / "data" / "results" / "e14"))  # override: smoke tests only
ACC = 1.0 / 103.642697  # eV/(A amu) -> A/fs^2
KB = 8.617333262e-5  # eV/K
R_MAX = 5.0


def molecules():
    """(name, symbols, positions A) for every SPICE dipeptide, first conformation; atom order from the SMILES maps."""
    import h5py

    out = []
    with h5py.File(DATA, "r") as f:
        for name in sorted(f):
            s = f[name]["smiles"][0]
            s = s.decode() if isinstance(s, bytes) else str(s)
            # bracket atoms carry map numbers (the atom order); aromatic atoms are lower case ([nH:7])
            at = re.findall(r"\[([A-Z][a-z]?|[cnos])[^\]]*?:(\d+)\]", s)
            sym = [""] * max(int(i) for _, i in at)
            for e, i in at:
                sym[int(i) - 1] = e.capitalize()
            x = np.asarray(f[name]["conformations"][0], dtype=np.float64) * 10.0  # nm -> A
            assert len(sym) == len(x) and all(sym), name
            out.append((name, sym, x))
    return out


def split_names(names, seed: int = 0):
    """80 / 10 / 10 by molecule (decision 6): no dipeptide in two splits."""
    rng = np.random.default_rng(seed)
    p = rng.permutation(sorted(names))
    n = len(p)
    return {"train": sorted(p[: int(0.8 * n)]), "val": sorted(p[int(0.8 * n): int(0.9 * n)]),
            "test": sorted(p[int(0.9 * n):])}


def scalar_index(model) -> torch.Tensor:
    """Positions of the l = 0 channels of every layer inside MACE's concatenated ``node_feats`` output."""
    idx, off = [], 0
    for prod in model.products:
        irreps = prod.linear.irreps_out
        for mul, ir in irreps:
            dim = mul * ir.dim
            if ir.l == 0:
                idx.extend(range(off, off + dim))
            off += dim
    return torch.as_tensor(idx, dtype=torch.long)


class MolBatch:
    """Many gas-phase molecules in one MACE call: each alone in a large periodic box (no molecule sees another; the
    box only satisfies MACE's data layer), edges recomputed within each molecule at every call."""

    def __init__(self, calc, mols, device, spacing: float = 40.0):
        from ase import Atoms

        from frost.calibrate import _batch_for

        g = max(1, math.ceil(len(mols) ** (1 / 3)))
        syms, xs, mol_id, ci, cj, start = [], [], [], [], [], 0
        for k, (_, sym, x) in enumerate(mols):
            off = np.array([k % g, (k // g) % g, k // (g * g)], dtype=np.float64) * spacing + spacing / 2
            xs.append(x - x.mean(0) + off)
            syms += sym
            mol_id += [k] * len(sym)
            a = np.arange(start, start + len(sym))
            i, j = np.meshgrid(a, a, indexing="ij")
            m = i != j
            ci.append(i[m])
            cj.append(j[m])
            start += len(sym)
        box = spacing * g
        self.atoms = Atoms(symbols=syms, positions=np.concatenate(xs), cell=[box, box, box], pbc=True)
        self.model = calc.models[0]
        self.data = _batch_for(calc, self.atoms, R_MAX, device)[0].to_dict()
        self.dtype = self.data["positions"].dtype
        self.ci = torch.as_tensor(np.concatenate(ci), device=device)
        self.cj = torch.as_tensor(np.concatenate(cj), device=device)
        self.mol = torch.as_tensor(mol_id, device=device)
        self.n_mol = len(mols)
        self.sizes = [len(s) for _, s, _ in mols]
        self.scal = scalar_index(self.model).to(device)
        self.m = torch.as_tensor(self.atoms.get_masses(), dtype=torch.float64, device=device).unsqueeze(1)
        self.x0 = torch.as_tensor(self.atoms.get_positions(), dtype=torch.float64, device=device)

    def __call__(self, x: torch.Tensor):
        """Per-atom energies (eV), forces (eV/A), the l = 0 node features of every layer."""
        d = (x[self.cj] - x[self.ci]).norm(dim=1)
        keep = d < R_MAX
        n_e = int(keep.sum())
        self.data["edge_index"] = torch.stack((self.ci[keep], self.cj[keep]))
        self.data["shifts"] = torch.zeros(n_e, 3, dtype=self.dtype, device=x.device)
        self.data["unit_shifts"] = torch.zeros(n_e, 3, dtype=self.dtype, device=x.device)
        self.data["positions"] = x.detach().to(self.dtype).clone()
        out = self.model(self.data, compute_force=True, training=False)
        return (out["node_energy"].detach().double(), out["forces"].detach().double(),
                out["node_feats"].detach()[:, self.scal])


def stage_md(args, dev) -> int:
    from frost.teachers import load_calc

    calc = load_calc("off-medium", str(dev), "float32", enable_cueq=True)
    mols = molecules()
    if args.limit:
        mols = mols[: args.limit]
    out_dir = BASE / "md"
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [m for m in mols if not (out_dir / f"{m[0]}.npz").exists()]
    print(f"{len(mols)} molecules, {len(todo)} to run, {args.batch} per call", flush=True)
    for b0 in range(0, len(todo), args.batch):
        part = todo[b0: b0 + args.batch]
        mb = MolBatch(calc, part, dev)
        n = mb.x0.shape[0]
        rng = torch.Generator(device="cpu").manual_seed(args.seed + b0)
        sig = torch.sqrt(KB * args.temp / mb.m * ACC)
        x = mb.x0.clone()
        v = sig * torch.randn(n, 3, generator=rng, dtype=torch.float64).to(dev)
        dt, g = 1.0, args.friction
        c1 = math.exp(-g * dt)
        _, f, _ = mb(x)
        refs, frames = [], []
        n_eq, n_prod = round(args.equil_ps * 1e3), round(args.ps * 1e3)
        t0 = time.perf_counter()
        temps = []
        for s in range(1, n_eq + n_prod + 1):
            v = v + 0.5 * dt * f / mb.m * ACC
            x = x + 0.5 * dt * v
            v = c1 * v + math.sqrt(1 - c1 * c1) * sig * torch.randn(n, 3, generator=rng, dtype=torch.float64).to(dev)
            x = x + 0.5 * dt * v
            e_at, f, feats = mb(x)
            v = v + 0.5 * dt * f / mb.m * ACC
            p = s - n_eq
            if p <= 0:
                continue
            phase = p % args.ref_every
            if phase == 0:
                refs.append((x.float().cpu(), e_at.float().cpu(), feats.half().cpu()))
                temps.append(float((mb.m * v * v).sum() / ACC) / (3 * n * KB))
            elif refs and phase <= args.gap_max:
                e_mol = torch.zeros(mb.n_mol, dtype=torch.float64, device=dev).index_add_(0, mb.mol, e_at)
                frames.append((len(refs) - 1, phase, x.float().cpu(), e_mol.cpu(), f.float().cpu()))
        wall = time.perf_counter() - t0
        # one file per molecule: reference frames and the frames that follow each one
        a = 0
        for k, (name, sym, _) in enumerate(part):
            sl = slice(a, a + len(sym))
            a += len(sym)
            np.savez(out_dir / f"{name}.npz", symbols=np.array(sym),
                     numbers=mb.atoms.get_atomic_numbers()[sl],
                     ref_x=np.stack([r[0][sl].numpy() for r in refs]),
                     ref_e_atom=np.stack([r[1][sl].numpy() for r in refs]),
                     ref_feats=np.stack([r[2][sl].numpy() for r in refs]),
                     fr_ref=np.array([fr[0] for fr in frames]), fr_gap=np.array([fr[1] for fr in frames]),
                     fr_x=np.stack([fr[2][sl].numpy() for fr in frames]),
                     fr_e=np.array([float(fr[3][k]) for fr in frames]),
                     fr_f=np.stack([fr[4][sl].numpy() for fr in frames]))
        print(f"batch {b0 // args.batch}: {len(part)} molecules, {n} atoms, {wall:.0f} s "
              f"({1e3 * wall / (n_eq + n_prod):.1f} ms/step), production T {np.mean(temps):.1f} K", flush=True)
    meta = {"molecules": len(mols), "split": split_names([m[0] for m in mols]), "temp": args.temp,
            "friction_per_fs": args.friction, "equil_ps": args.equil_ps, "ps": args.ps, "ref_every_fs": args.ref_every,
            "gap_max": args.gap_max, "reference": "MACE-OFF23 medium (mace_off), float32, cuEquivariance"}
    (BASE / "md_meta.json").write_text(json.dumps(meta, indent=1))
    return 0


SPECIES = [1, 6, 7, 8, 16]  # H C N O S: every element in the SPICE dipeptides and 3BPA


def _mlp(sizes):
    layers = []
    for a, b in itertools.pairwise(sizes[:-1]):
        layers += [torch.nn.Linear(a, b), torch.nn.SiLU()]
    layers.append(torch.nn.Linear(sizes[-2], sizes[-1]))
    return torch.nn.Sequential(*layers)


class BoostXS(torch.nn.Module):
    """BoostMD's XS model in energy mode with invariant reference features (their Table 1 row "XS, L_ref 0, energy"):
    Eq. 7 (edge message from the current edge and its change), Eq. 8 (weighted by the neighbour's reference features),
    Eq. 9 (MACE's symmetric contraction, correlation 2), Eq. 10-11 (readout times sum_j |x_ij|, added to the reference
    model's per-atom energy at the reference frame). No rotational framing, as in all their experiments. Built from
    e3nn's and MACE's public blocks; the wiring is ours (spec decisions 1-4, 10)."""

    def __init__(self, n_ref: int = 256, channels: int = 125, n_gauss: int = 8, x_max: float = 1.5,
                 avg_nn: float = 20.0, hidden: int = 64):
        super().__init__()
        from e3nn import o3
        from mace.modules.blocks import EquivariantProductBasisBlock
        from mace.modules.radial import GaussianBasis, PolynomialCutoff

        ir = o3.Irreps("0e+1o+2e")
        self.sh = o3.SphericalHarmonics(o3.Irreps.spherical_harmonics(2), normalize=False, normalization="component")
        self.tp = o3.FullyConnectedTensorProduct(ir, ir, ir, shared_weights=False, internal_weights=False)
        self.g_r, self.g_x, self.cut = GaussianBasis(R_MAX, n_gauss), GaussianBasis(x_max, n_gauss), PolynomialCutoff(R_MAX, p=5)
        self.F = _mlp([2 * n_gauss, hidden, hidden, hidden, self.tp.weight_numel])
        self.h_in = torch.nn.Linear(n_ref, 32)
        self.Fp = _mlp([2 * n_gauss + 32, hidden, hidden, hidden, channels])
        self.W = torch.nn.Linear(n_ref, channels, bias=False)
        self.prod = EquivariantProductBasisBlock(node_feats_irreps=o3.Irreps(f"{channels}x0e+{channels}x1o+{channels}x2e"),
                                                 target_irreps=o3.Irreps(f"{channels}x0e"), correlation=2, use_sc=False,
                                                 num_elements=len(SPECIES))
        self.readout = _mlp([channels, 16, 1])
        self.register_buffer("avg_nn", torch.tensor(float(avg_nn)))

    def forward(self, x, x_ref, edge_index, node_attrs, h_ref, e_ref, batch, n_graphs: int):
        from mace.tools.scatter import scatter_sum

        snd, rcv = edge_index[0], edge_index[1]  # receiver i, sender (neighbour) j
        r_ji = x[snd] - x[rcv]
        x_ji = r_ji - (x_ref[snd] - x_ref[rcv])
        dr = r_ji.norm(dim=1, keepdim=True)
        dx = x_ji.norm(dim=1, keepdim=True)
        y_r = self.sh(r_ji / dr.clamp_min(1e-9))
        # solid harmonics |x|^l Y_l(x^) of the change itself (spec decision 10, amended 2026-11-12): smooth at x -> 0, where
        # Y_l(x / |x|) has a 1/|x| gradient that, times the readout's sum_j |x_ij|, blew the first training run up
        y_x = self.sh(x_ji)
        emb = torch.cat([self.g_r(dr) * self.cut(dr), self.g_x(dx)], dim=-1)
        X = self.tp(y_r, y_x, self.F(emb))  # (E, 9): Eq. 7
        hj = h_ref[snd]
        w = self.Fp(torch.cat([emb, self.h_in(hj)], dim=-1)) * self.W(hj)  # Eq. 8 with L_ref = 0
        n = x.shape[0]
        A = scatter_sum(w.unsqueeze(-1) * X.unsqueeze(1), rcv, dim=0, dim_size=n) / self.avg_nn
        B = self.prod(node_feats=A, sc=None, node_attrs=node_attrs)  # Eq. 9-10
        s_i = self.readout(B).squeeze(-1)
        e_atom = e_ref + s_i * scatter_sum(dx.squeeze(-1), rcv, dim=0, dim_size=n)  # Eq. 11
        return scatter_sum(e_atom, batch, dim=0, dim_size=n_graphs)


def xs_plus(avg_nn: float, e0: np.ndarray, scale: float):
    """Their XS+ comparison: MACE with one interaction, 256 channels, correlation 2, l_max 2 (spec decision 8),
    through MACE's own constructor."""
    from e3nn import o3
    from mace.modules import interaction_classes
    from mace.modules.models import ScaleShiftMACE

    return ScaleShiftMACE(r_max=R_MAX, num_bessel=8, num_polynomial_cutoff=5, max_ell=2,
                          interaction_cls=interaction_classes["RealAgnosticResidualInteractionBlock"],
                          interaction_cls_first=interaction_classes["RealAgnosticInteractionBlock"],
                          num_interactions=1, num_elements=len(SPECIES), hidden_irreps=o3.Irreps("256x0e"),
                          MLP_irreps=o3.Irreps("16x0e"), atomic_energies=e0, avg_num_neighbors=avg_nn,
                          atomic_numbers=SPECIES, correlation=2, gate=torch.nn.functional.silu,
                          radial_MLP=[64, 64, 64], radial_type="bessel", atomic_inter_scale=scale,
                          atomic_inter_shift=0.0)


class PairData:
    """Every (reference frame, frame 1-10 steps later) pair of a split, kept on the GPU, batched as disjoint graphs."""

    def __init__(self, names, dev):
        self.dev, self.mols = dev, []
        for name in names:
            d = np.load(BASE / "md" / f"{name}.npz")
            n = len(d["numbers"])
            a = np.arange(n)
            i, j = np.meshgrid(a, a, indexing="ij")
            m = i != j
            z = torch.as_tensor([SPECIES.index(int(q)) for q in d["numbers"]], device=dev)
            self.mols.append({
                "n": n, "ci": torch.as_tensor(i[m], device=dev), "cj": torch.as_tensor(j[m], device=dev),
                "attrs": torch.nn.functional.one_hot(z, len(SPECIES)).float(),
                "ref_x": torch.as_tensor(d["ref_x"], device=dev), "ref_e": torch.as_tensor(d["ref_e_atom"], device=dev),
                "ref_h": torch.as_tensor(d["ref_feats"], device=dev), "fr_ref": torch.as_tensor(d["fr_ref"], device=dev),
                "fr_x": torch.as_tensor(d["fr_x"], device=dev), "fr_e": torch.as_tensor(d["fr_e"], device=dev),
                "fr_f": torch.as_tensor(d["fr_f"], device=dev)})
        self.index = [(k, f) for k, mo in enumerate(self.mols) for f in range(len(mo["fr_e"]))]

    def batch(self, items):
        xs, xr, hr, er, at, ei, bt, E, F, off = [], [], [], [], [], [], [], [], [], 0
        for g, (k, f) in enumerate(items):
            mo = self.mols[k]
            r = int(mo["fr_ref"][f])
            x = mo["fr_x"][f].double()
            d = (x[mo["cj"]] - x[mo["ci"]]).norm(dim=1)
            keep = d < R_MAX
            ei.append(torch.stack((mo["cj"][keep], mo["ci"][keep])) + off)  # sender j, receiver i
            xs.append(x)
            xr.append(mo["ref_x"][r].double())
            hr.append(mo["ref_h"][r].float())
            er.append(mo["ref_e"][r].double())
            at.append(mo["attrs"])
            bt.append(torch.full((mo["n"],), g, device=self.dev, dtype=torch.long))
            E.append(mo["fr_e"][f])
            F.append(mo["fr_f"][f].double())
            off += mo["n"]
        return {"x": torch.cat(xs).float(), "x_ref": torch.cat(xr).float(), "h_ref": torch.cat(hr),
                "e_ref": torch.cat(er).float(), "attrs": torch.cat(at), "edge_index": torch.cat(ei, dim=1),
                "batch": torch.cat(bt), "n_graphs": len(items), "E": torch.stack(E).float(), "F": torch.cat(F).float()}


def predict(kind, model, b, create_graph: bool):
    """Energies per graph and forces for one batch, by the model's own energy (forces by autograd)."""
    x = b["x"].clone().requires_grad_(True)
    if kind == "boost":
        E = model(x, b["x_ref"], b["edge_index"], b["attrs"], b["h_ref"], b["e_ref"], b["batch"], b["n_graphs"])
    else:
        n_e = b["edge_index"].shape[1]
        ptr = torch.zeros(b["n_graphs"] + 1, dtype=torch.long, device=x.device)
        ptr[1:] = torch.bincount(b["batch"], minlength=b["n_graphs"]).cumsum(0)
        data = {"positions": x, "node_attrs": b["attrs"], "edge_index": b["edge_index"],
                "shifts": torch.zeros(n_e, 3, device=x.device), "unit_shifts": torch.zeros(n_e, 3, device=x.device),
                "cell": torch.zeros(3 * b["n_graphs"], 3, device=x.device), "batch": b["batch"], "ptr": ptr,
                "weight": torch.ones(b["n_graphs"], device=x.device)}
        E = model(data, training=False, compute_force=False)["energy"]
    F = -torch.autograd.grad(E.sum(), x, create_graph=create_graph)[0]
    return E, F


def loss_of(E, F, b, n_atoms):
    return ((E - b["E"]) / n_atoms).pow(2).mean() + 10.0 * (F - b["F"]).pow(2).mean()


def stage_train(args, dev) -> int:
    meta = json.loads((BASE / "md_meta.json").read_text())
    tr, va = PairData(meta["split"]["train"], dev), PairData(meta["split"]["val"], dev)
    rng = np.random.default_rng(args.seed)
    vitems = [va.index[i] for i in rng.choice(len(va.index), min(2000, len(va.index)), replace=False)]
    probe = tr.batch([tr.index[i] for i in rng.choice(len(tr.index), min(256, len(tr.index)), replace=False)])
    avg_nn = probe["edge_index"].shape[1] / probe["x"].shape[0]
    if args.model == "boost":
        model = BoostXS(avg_nn=avg_nn).to(dev)
    else:
        from frost.teachers import load_calc

        ref = load_calc("off-medium", str(dev), "float32", enable_cueq=False).models[0]
        e0_all = ref.atomic_energies_fn.atomic_energies.detach().cpu().numpy().reshape(-1)
        zs = [int(z) for z in ref.atomic_numbers]
        e0 = np.array([e0_all[zs.index(z)] for z in SPECIES])
        model = xs_plus(avg_nn, e0, float(probe["F"].std())).to(dev)
    out = BASE / args.model
    out.mkdir(parents=True, exist_ok=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)
    start, best, log = 0, float("inf"), []
    if (out / "last.pt").exists():
        st = torch.load(out / "last.pt", map_location=dev, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        start, best, log = st["epoch"] + 1, st["best"], st["log"]
    t_start = time.time()
    for ep in range(start, args.epochs):
        model.train()
        t0 = time.time()
        for _ in range(args.pairs_per_epoch // args.batch_size):
            items = [tr.index[i] for i in rng.integers(0, len(tr.index), args.batch_size)]
            b = tr.batch(items)
            E, F = predict(args.model, model, b, create_graph=True)
            loss = loss_of(E, F, b, torch.bincount(b["batch"]).float())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)  # MACE's trainer default (spec decision 7, amended)
            opt.step()
        model.eval()
        vl, se_e, se_f, n_at, n_f = 0.0, 0.0, 0.0, 0, 0
        for c in range(0, len(vitems), 64):
            b = va.batch(vitems[c:c + 64])
            E, F = predict(args.model, model, b, create_graph=False)
            na = torch.bincount(b["batch"]).float()
            vl += float(loss_of(E.detach(), F.detach(), b, na)) * len(na)
            se_e += float((((E.detach() - b["E"]) / na) ** 2).sum())
            se_f += float(((F.detach() - b["F"]) ** 2).sum())
            n_at += len(na)
            n_f += F.numel()
        vl /= n_at
        sched.step(vl)
        lr = opt.param_groups[0]["lr"]
        row = {"epoch": ep, "val_loss": vl, "E_rmse_meV_atom": 1e3 * math.sqrt(se_e / n_at),
               "F_rmse_meV_A": 1e3 * math.sqrt(se_f / n_f), "lr": lr, "s": time.time() - t0}
        log.append(row)
        print(json.dumps(row), flush=True)
        if vl < best:
            best = vl
            torch.save({"model": model.state_dict(), "avg_nn": avg_nn}, out / "best.pt")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": ep,
                    "best": best, "log": log, "avg_nn": avg_nn}, out / "last.pt")
        if lr < 1e-5 or (args.max_hours and time.time() - t_start > 3600 * args.max_hours):
            break
    return 0



def load_trained(kind: str, dev, ckpt: str = "best"):
    st = torch.load(BASE / kind / f"{ckpt}.pt", map_location=dev, weights_only=False)
    if kind == "boost":
        model = BoostXS(avg_nn=st["avg_nn"]).to(dev)
    else:
        from frost.teachers import load_calc

        ref = load_calc("off-medium", str(dev), "float32", enable_cueq=False).models[0]
        e0_all = ref.atomic_energies_fn.atomic_energies.detach().cpu().numpy().reshape(-1)
        zs = [int(z) for z in ref.atomic_numbers]
        model = xs_plus(st["avg_nn"], np.array([e0_all[zs.index(z)] for z in SPECIES]), 1.0).to(dev)
    model.load_state_dict(st["model"])
    return model.eval()


def stage_test(args, dev) -> int:
    """V1: energy-per-atom and force RMSE against MACE-OFF23-M on the held-out molecules (every pair)."""
    meta = json.loads((BASE / "md_meta.json").read_text())
    te = PairData(meta["split"]["test"], dev)
    out = {"pairs": len(te.index), "molecules": len(meta["split"]["test"])}
    for kind in ("boost", "xsplus"):
        if not (BASE / kind / f"{args.ckpt}.pt").exists():
            continue
        model = load_trained(kind, dev, args.ckpt)
        se_e = se_f = 0.0
        n_g = n_f = 0
        for c in range(0, len(te.index), 64):
            b = te.batch(te.index[c:c + 64])
            E, F = predict(kind, model, b, create_graph=False)
            na = torch.bincount(b["batch"]).float()
            se_e += float((((E.detach() - b["E"]) / na) ** 2).sum())
            se_f += float(((F.detach() - b["F"]) ** 2).sum())
            n_g += len(na)
            n_f += F.numel()
        out[kind] = {"E_rmse_meV_atom": 1e3 * math.sqrt(se_e / n_g), "F_rmse_meV_A": 1e3 * math.sqrt(se_f / n_f)}
        print(kind, out[kind], flush=True)
    if "boost" in out and "xsplus" in out:
        r = out["boost"]["F_rmse_meV_A"] / out["xsplus"]["F_rmse_meV_A"]
        out["V1"] = {"force_ratio_boost_over_xsplus": r, "theirs": 75.3 / 121.1,
                     "pass": bool(abs(out["boost"]["F_rmse_meV_A"] - 75.3) <= 0.3 * 75.3 and r < 1 and abs(r - 0.622) <= 0.15)}
    (RES / "boostmd_test.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out.get("V1")), flush=True)
    return 0


def _three_bpa(n_copies: int = 125):
    from ase.io import read

    at = read(ROOT / "data" / "datasets" / "3bpa" / "dataset_3BPA" / "test_300K.xyz", index=0)
    return [(f"3bpa{k}", list(at.get_chemical_symbols()), at.get_positions()) for k in range(n_copies)]


def stage_time(args, dev) -> int:
    """V2: time per evaluation (energy and forces) on 125 3BPA molecules (3,375 atoms), 100 calls after 50 warm-up
    (their App. C.2): MACE-OFF23-M (cuEquivariance and plain e3nn), BoostMD XS, XS+; the same edges for all."""
    from frost.teachers import load_calc

    mols = _three_bpa()
    res = {"atoms": sum(len(m[1]) for m in mols)}

    def timed(fn):
        for _ in range(50):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(100):
            fn()
        torch.cuda.synchronize()
        return 1e3 * (time.perf_counter() - t0) / 100

    for label, cueq in (("mace_off_m_cueq", True), ("mace_off_m_e3nn", False)):
        mb = MolBatch(load_calc("off-medium", str(dev), "float32", enable_cueq=cueq), mols, dev)
        x0 = mb.x0.clone()
        res[label] = timed(lambda mb=mb, x0=x0: mb(x0))
    e_at, _, feats = mb(x0)
    z = torch.as_tensor([SPECIES.index(int(q)) for q in mb.atoms.get_atomic_numbers()], device=dev)
    keep = (x0[mb.cj] - x0[mb.ci]).norm(dim=1) < R_MAX
    b = {"x": (x0 + 0.01 * torch.randn_like(x0)).float(), "x_ref": x0.float(), "h_ref": feats.float(),
         "e_ref": e_at.float(), "attrs": torch.nn.functional.one_hot(z, len(SPECIES)).float(),
         "edge_index": torch.stack((mb.cj[keep], mb.ci[keep])), "batch": mb.mol, "n_graphs": mb.n_mol}
    for kind in ("boost", "xsplus"):
        if (BASE / kind / f"{args.ckpt}.pt").exists():
            model = load_trained(kind, dev, args.ckpt)
            res[kind] = timed(lambda model=model, kind=kind: predict(kind, model, b, create_graph=False))
    for kind in ("boost", "xsplus"):
        if kind in res:
            res[f"{kind}_speedup_vs_cueq"] = res["mace_off_m_cueq"] / res[kind]
            res[f"{kind}_speedup_vs_e3nn"] = res["mace_off_m_e3nn"] / res[kind]
    (RES / "boostmd_time.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1), flush=True)
    return 0


def _ala_dipeptide():
    lines = (ROOT / "data" / "datasets" / "spice_dipeptides" / "ala_dipeptide_cid141892.sdf").read_text().splitlines()
    n = int(lines[3][:3])
    sym, xyz = [], []
    for ln in lines[4:4 + n]:
        p = ln.split()
        xyz.append([float(p[0]), float(p[1]), float(p[2])])
        sym.append(p[3])
    return [("ala_dipeptide", sym, np.array(xyz))]


def stage_stab(args, dev) -> int:
    """V3: alanine dipeptide (CID 141892, not in training), BoostMD with the reference every 10 steps (their Fig. 3),
    Langevin 300 K, 1 fs, --ps long; records whether it stays finite, its temperature and bond lengths."""
    from frost.teachers import load_calc

    mols = _ala_dipeptide()
    mb = MolBatch(load_calc("off-medium", str(dev), "float32", enable_cueq=True), mols, dev)
    model = load_trained("boost", dev, args.ckpt)
    z = torch.as_tensor([SPECIES.index(int(q)) for q in mb.atoms.get_atomic_numbers()], device=dev)
    attrs = torch.nn.functional.one_hot(z, len(SPECIES)).float()
    n = mb.x0.shape[0]
    rng = torch.Generator(device="cpu").manual_seed(args.seed)
    sig = torch.sqrt(KB * args.temp / mb.m * ACC)
    x, v = mb.x0.clone(), sig * torch.randn(n, 3, generator=rng, dtype=torch.float64).to(dev)
    c1 = math.exp(-args.friction)
    bonds = (mb.x0[mb.cj] - mb.x0[mb.ci]).norm(dim=1) < 1.7
    d0 = (mb.x0[mb.cj] - mb.x0[mb.ci]).norm(dim=1)[bonds]

    def forces(step, x, ref):
        if step % 10 == 0:
            e_at, f, feats = mb(x)
            return f, (x.clone(), feats.float(), e_at.float())
        keep = (x[mb.cj] - x[mb.ci]).norm(dim=1) < R_MAX
        b = {"x": x.float(), "x_ref": ref[0].float(), "h_ref": ref[1], "e_ref": ref[2], "attrs": attrs,
             "edge_index": torch.stack((mb.cj[keep], mb.ci[keep])), "batch": mb.mol, "n_graphs": 1}
        _, f = predict("boost", model, b, create_graph=False)
        return f.detach().double(), ref

    f, ref = forces(0, x, None)
    temps, worst, t0 = [], 0.0, time.perf_counter()
    steps = round(args.ps * 1e3)
    ok = True
    for s in range(1, steps + 1):
        v = v + 0.5 * f / mb.m * ACC
        x = x + 0.5 * v
        v = c1 * v + math.sqrt(1 - c1 * c1) * sig * torch.randn(n, 3, generator=rng, dtype=torch.float64).to(dev)
        x = x + 0.5 * v
        f, ref = forces(s, x, ref)
        v = v + 0.5 * f / mb.m * ACC
        if s % 1000 == 0:
            if not torch.isfinite(x).all():
                ok = False
                break
            temps.append(float((mb.m * v * v).sum() / ACC) / (3 * n * KB))
            d = (x[mb.cj] - x[mb.ci]).norm(dim=1)[bonds]
            worst = max(worst, float(((d - d0).abs() / d0).max()))
    res = {"ps": s / 1e3, "finite": ok, "T_mean": float(np.mean(temps)) if temps else None,
           "T_sd": float(np.std(temps)) if temps else None, "max_bond_strain": worst,
           "ms_per_step": 1e3 * (time.perf_counter() - t0) / s,
           "V3_pass": bool(ok and s == steps and worst < 0.3 and temps and abs(np.mean(temps) - args.temp) < 15)}
    (RES / "boostmd_stab.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res), flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("md", "train", "test", "time", "stab"))
    ap.add_argument("--batch", type=int, default=64, help="md: molecules per MACE call")
    ap.add_argument("--limit", type=int, default=0, help="md: first N molecules only (smoke test)")
    ap.add_argument("--temp", type=float, default=300.0)
    ap.add_argument("--friction", type=float, default=0.01, help="per fs")
    ap.add_argument("--equil-ps", type=float, default=2.0)
    ap.add_argument("--ps", type=float, default=10.0)
    ap.add_argument("--ref-every", type=int, default=100, help="steps between reference frames")
    ap.add_argument("--gap-max", type=int, default=10, help="frames kept after each reference (N)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", choices=("boost", "xsplus"), default="boost", help="train: BoostMD XS or their XS+")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--pairs-per-epoch", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--max-hours", type=float, default=0.0)
    ap.add_argument("--ckpt", choices=("best", "last"), default="best")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = torch.device(args.device)
    return {"md": stage_md, "train": stage_train, "test": stage_test, "time": stage_time,
            "stab": stage_stab}[args.stage](args, dev)


if __name__ == "__main__":
    raise SystemExit(main())
