"""MTS-Allegro (Fu, Musaelian, Johansson, Jaakkola, Kozinsky, arXiv 2310.13756), rebuilt: their code is not public.

Their method, as the paper states it (section 2, Algorithm 2, Tables 2-4; read 2026-11-06):
- two Allegro models predict one energy, E = E_inner + E_outer;
- inner: "an efficient model with fewer parameters and interaction layers and a small radial cutoff" (4 A, 1 layer,
  half the width of the outer model); outer: the full Allegro model (6 A, 2 layers);
- training is sequential with one loss: 20 epochs of E = E_inner alone, then E = E_inner + E_outer, with the outer
  model's output block initialised to zero so it starts by contributing nothing;
- loss lambda_E ||(E_ML - E) / N||^2 + lambda_F (1 / 3N) ||F_ML - F||^2, lambda_E 1, lambda_F 2; Adam, lr 0.005,
  ReduceLROnPlateau (patience 5, factor 0.5), stop at lr 1e-6, batch 1;
- r-RESPA: the inner model every 0.5 fs step, the outer every N steps (N 2-8; 4 recommended: 2.6x at 192 atoms);
- data: Cheng et al. (PNAS 2019) revPBE0-D3 water, 1,593 frames of 192 atoms, 1,000 train / 100 validation / rest test.
Their Table 1 (energy MAE meV/atom; force MAE, RMSE meV/A): Allegro 2.4 / 40.3 / 77.4, MTS-Allegro 1.6 / 35.5 / 76.9,
MTS inner alone 12.5 / 72.5 / 118.3. Reproducing these first is what makes this rebuild a validated replication (tier
T2 of the fair-comparison skill) rather than an unvalidated one.

Built only from public pieces of nequip-allegro (``allegro.model.AllegroModel``, ``nequip.data.from_ase`` and its
transforms, ``nequip.utils.global_state.set_global_state``); the training loop is ours. Allegro 0.8's builder names
its widths differently from the 2023 code: the mapping used here (``ARCH``) keeps each stated depth, width, cutoff,
l_max and multiplicity; where the 2023 two-body MLP widened layer by layer ([128, 256, 512, 1024]) a uniform hidden
width (512) is used.

Stages (frost-nequip environment, GPU):
- ``train --model allegro|mts``: train and checkpoint (resumable), selecting on the validation split only;
- ``test``: Table 1's metrics on the test split for both, and for the inner model alone;
- ``md``: section 4's NVE protocol (0.5 fs inner step, outer N in 2-8, 100 ps after an equilibration with the full
  model, drift fitted after 20 ps), and the speed of every arm against the full Allegro model every step.

Data: ``data/datasets/water_cheng2019/water_1593_eVA.xyz`` (Materials Cloud 2018.0020/v1, CC BY 4.0, converted from
Bohr and Hartree). Output: ``data/baselines/mts_allegro/<model>/`` (checkpoints, log) and
``data/results/e14/mts_allegro_<stage>.json``.

Usage (guarded launcher; GPU)
  MAX_HOURS=24 LANE=gpu1 scripts/safe_run.sh mtsa_allegro .venvs/frost-nequip/bin/python \\
      experiments/e14_baselines/mts_allegro.py train --model allegro
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "datasets" / "water_cheng2019" / "water_1593_eVA.xyz"
BASE = ROOT / "data" / "baselines" / "mts_allegro"
RESULTS = ROOT / "data" / "results" / "e14"
TYPES = ["H", "O"]
LAMBDA_E, LAMBDA_F = 1.0, 2.0
INNER_ONLY_EPOCHS = 20

# Tables 2 and 3 of the paper, in allegro 0.8's argument names
ARCH = {
    "outer": {"r_max": 6.0, "num_layers": 2, "radial_chemical_embed_dim": 128, "scalar_embed_mlp_hidden_layers_depth": 2,
              "scalar_embed_mlp_hidden_layers_width": 512, "num_scalar_features": 1024,
              "allegro_mlp_hidden_layers_depth": 2, "allegro_mlp_hidden_layers_width": 1024, "num_tensor_features": 32,
              "readout_mlp_hidden_layers_depth": 1, "readout_mlp_hidden_layers_width": 128},
    "inner": {"r_max": 4.0, "num_layers": 1, "radial_chemical_embed_dim": 64, "scalar_embed_mlp_hidden_layers_depth": 2,
              "scalar_embed_mlp_hidden_layers_width": 256, "num_scalar_features": 512,
              "allegro_mlp_hidden_layers_depth": 2, "allegro_mlp_hidden_layers_width": 512, "num_tensor_features": 32,
              "readout_mlp_hidden_layers_depth": 1, "readout_mlp_hidden_layers_width": 64},
}
ZERO_INIT = "model.func.edge_readout.mlp_module.mlp.2.weight"  # the outer model's output block (last readout layer)


def load_frames():
    from ase.io import read

    frames = read(DATA, ":")
    rng = np.random.default_rng(0)
    order = rng.permutation(len(frames))
    return frames, {"train": order[:1000], "val": order[1000:1100], "test": order[1100:]}


def to_data(atoms, r_max, device):
    from nequip.data import from_ase
    from nequip.data.transforms import (
        ChemicalSpeciesToAtomTypeMapper,
        NeighborListTransform,
    )

    d = from_ase(atoms)
    d = ChemicalSpeciesToAtomTypeMapper(model_type_names=TYPES,
                                        chemical_species_to_atom_type_map={t: t for t in TYPES})(d)
    d = NeighborListTransform(r_max=r_max)(d)
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in d.items()}


def build(part: str, stats: dict, outer_zero: bool = False):
    from allegro.model import AllegroModel

    a = ARCH[part]
    model = AllegroModel(
        seed=0 if part == "inner" else 1, model_dtype="float32", type_names=TYPES, l_max=2, parity=True,
        radial_chemical_embed={"_target_": "allegro.nn.TwoBodyBesselScalarEmbed", "num_bessels": 8},
        avg_num_neighbors=stats["neighbours"][a["r_max"]], per_type_energy_scales=stats["force_rms"],
        per_type_energy_shifts=0.0 if outer_zero else stats["energy_per_atom"], **a)
    if outer_zero:
        params = dict(model.named_parameters())
        with torch.no_grad():
            params[ZERO_INIT].zero_()
    return model


def dataset(frames, split, device):
    """Per frame: the 4 A and 6 A graphs, the reference energy and forces (eV, eV/A), on the device."""
    from nequip.data import AtomicDataDict

    out = []
    for i in split:
        a = frames[int(i)]
        out.append({"d4": to_data(a, 4.0, device), "d6": to_data(a, 6.0, device),
                    "E": torch.tensor(a.get_potential_energy(), device=device),
                    "F": torch.as_tensor(a.get_forces(), dtype=torch.float32, device=device), "n": len(a)})
    for row in out:
        for g in ("d4", "d6"):
            row[g].pop(AtomicDataDict.TOTAL_ENERGY_KEY, None)
            row[g].pop(AtomicDataDict.FORCE_KEY, None)
    return out


def stats_of(train):
    e = torch.stack([r["E"] / r["n"] for r in train])
    f = torch.cat([r["F"].reshape(-1) for r in train])
    from nequip.data import AtomicDataDict

    nb = {r_max: float(np.mean([r[g][AtomicDataDict.EDGE_INDEX_KEY].shape[1] / r["n"] for r in train]))
          for r_max, g in ((4.0, "d4"), (6.0, "d6"))}
    return {"energy_per_atom": float(e.mean()), "force_rms": float(f.pow(2).mean().sqrt()), "neighbours": nb}


def predict(models: dict, row, use_outer: bool):
    from nequip.data import AtomicDataDict

    if "allegro" in models:
        o = models["allegro"](dict(row["d6"]))
        return o[AtomicDataDict.TOTAL_ENERGY_KEY].reshape(()), o[AtomicDataDict.FORCE_KEY]
    o = models["inner"](dict(row["d4"]))
    e, f = o[AtomicDataDict.TOTAL_ENERGY_KEY].reshape(()), o[AtomicDataDict.FORCE_KEY]
    if use_outer:
        p = models["outer"](dict(row["d6"]))
        e, f = e + p[AtomicDataDict.TOTAL_ENERGY_KEY].reshape(()), f + p[AtomicDataDict.FORCE_KEY]
    return e, f


def loss_of(e, f, row):
    return LAMBDA_E * ((e - row["E"]) / row["n"]).pow(2) + LAMBDA_F * (f - row["F"]).pow(2).mean()


def evaluate(models, rows, use_outer):
    for m in models.values():
        m.eval()
    de, df, total = [], [], 0.0
    for row in rows:
        e, f = predict(models, row, use_outer)
        de.append(float((e.detach() - row["E"]).abs()) / row["n"])
        df.append((f.detach() - row["F"]).reshape(-1))
        total += float(loss_of(e.detach(), f.detach(), row))
    df = torch.cat(df)
    return {"loss": total / len(rows), "energy_mae_meV_per_atom": 1e3 * float(np.mean(de)),
            "force_mae_meV_per_A": 1e3 * float(df.abs().mean()), "force_rmse_meV_per_A": 1e3 * float(df.pow(2).mean().sqrt())}


def train(args, dev):
    frames, split = load_frames()
    tr, va = dataset(frames, split["train"], dev), dataset(frames, split["val"], dev)
    stats = stats_of(tr)
    out = BASE / args.model
    out.mkdir(parents=True, exist_ok=True)
    if args.model == "allegro":
        models = {"allegro": build("outer", stats).to(dev)}
    else:
        models = {"inner": build("inner", stats).to(dev), "outer": build("outer", stats, outer_zero=True).to(dev)}
    groups = [{"params": list(models[k].parameters()), "name": k} for k in models]
    if args.model == "mts":  # stage 1 trains the inner model alone; the outer group joins at stage 2
        opt = torch.optim.Adam([groups[0]], lr=args.lr)
    else:
        opt = torch.optim.Adam(groups, lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=5, factor=0.5)
    log, epoch, best = [], 0, math.inf
    ck = out / "last.pt"
    if ck.exists():  # resume
        state = torch.load(ck, map_location=dev, weights_only=False)
        for k, m in models.items():
            m.load_state_dict(state["models"][k])
        epoch, best, log = state["epoch"], state["best"], state["log"]
        if args.model == "mts" and epoch >= INNER_ONLY_EPOCHS and len(opt.param_groups) == 1:
            opt.add_param_group({**groups[1], "lr": state["lr"]})
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
    t_start = time.perf_counter()
    gen = torch.Generator().manual_seed(epoch)
    while epoch < args.max_epochs and opt.param_groups[0]["lr"] >= 1e-6:
        if args.model == "mts" and epoch == INNER_ONLY_EPOCHS and len(opt.param_groups) == 1:
            opt.add_param_group({**groups[1], "lr": opt.param_groups[0]["lr"]})
        use_outer = args.model != "mts" or epoch >= INNER_ONLY_EPOCHS
        for m in models.values():
            m.train()
        t0, run = time.perf_counter(), 0.0
        for i in torch.randperm(len(tr), generator=gen).tolist():
            e, f = predict(models, tr[i], use_outer)
            loss = loss_of(e, f, tr[i])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run += float(loss.detach())
        val = evaluate(models, va, use_outer)
        sched.step(val["loss"])
        epoch += 1
        row = {"epoch": epoch, "train_loss": run / len(tr), "lr": opt.param_groups[0]["lr"], "seconds": time.perf_counter() - t0,
               "stage": "inner only" if not use_outer else "joint", **{f"val_{k}": v for k, v in val.items()}}
        log.append(row)
        print(json.dumps(row), flush=True)
        if use_outer and val["loss"] < best:
            best = val["loss"]
            torch.save({k: m.state_dict() for k, m in models.items()}, out / "best.pt")
        torch.save({"models": {k: m.state_dict() for k, m in models.items()}, "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": epoch, "best": best, "log": log, "lr": opt.param_groups[0]["lr"]},
                   out / "last.tmp")
        (out / "last.tmp").replace(ck)
        if args.max_hours and (time.perf_counter() - t_start) / 3600 > args.max_hours:
            print("stopping for the time budget; resume continues from last.pt", flush=True)
            break
    (out / "stats.json").write_text(json.dumps({"stats": stats, "arch": ARCH, "epochs": epoch, "best_val_loss": best}, indent=1))
    return 0


def test(args, dev):
    frames, split = load_frames()
    te = dataset(frames, split["test"], dev)
    report = {"paper_table1": {"allegro": [2.4, 40.3, 77.4], "mts": [1.6, 35.5, 76.9], "mts_inner": [12.5, 72.5, 118.3]},
              "n_test": len(te)}
    for ckpt in ("last", "best"):
        for name in ("allegro", "mts"):
            if not (BASE / name / f"{ckpt}.pt").exists():
                continue
            models = load_models(name, saved_stats(name), dev, ckpt)
            report[f"{name}_{ckpt}"] = evaluate(models, te, use_outer=True)
            if name == "mts":
                report[f"mts_inner_{ckpt}"] = evaluate({"inner": models["inner"]}, te, use_outer=False)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "mts_allegro_test.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return 0


ACC = 1.0 / 103.642697  # eV / (A amu) -> A / fs^2
KB = 8.617333262e-5


class Part:
    """One Allegro model on a Verlet list (cutoff + skin), rebuilt by nequip's own neighbour list when any atom has
    moved more than half the skin; positions stay unwrapped so the stored cell shifts stay valid between rebuilds.
    Pairs between the cutoff and cutoff + skin contribute exactly zero (nequip's polynomial cutoff is masked at 1)."""

    def __init__(self, model, r_max, atoms, device, skin=1.0):
        self.model, self.r_max, self.skin, self.device = model, r_max, skin, device
        self.atoms = atoms.copy()
        self.rebuilds = 0
        self.ref = None

    def _build(self, x):
        from nequip.data import AtomicDataDict

        self.atoms.positions = x.detach().cpu().numpy()
        d = to_data(self.atoms, self.r_max + self.skin, self.device)
        d.pop(AtomicDataDict.TOTAL_ENERGY_KEY, None)
        d.pop(AtomicDataDict.FORCE_KEY, None)
        self.base, self.ref = d, x.detach().clone()
        self.rebuilds += 1

    def __call__(self, x):
        from nequip.data import AtomicDataDict

        if self.ref is None or float((x - self.ref).norm(dim=1).max()) > 0.5 * self.skin:
            self._build(x)
        d = dict(self.base)
        d[AtomicDataDict.POSITIONS_KEY] = x.detach().to(self.base[AtomicDataDict.POSITIONS_KEY].dtype)
        o = self.model(d)
        return float(o[AtomicDataDict.TOTAL_ENERGY_KEY]), o[AtomicDataDict.FORCE_KEY].detach().double()


def saved_stats(name: str) -> dict:
    """The normalisation a trained model was built with (stats.json, written at the end of training): a model rebuilt
    with other statistics would not be the trained model."""
    stats = json.loads((BASE / name / "stats.json").read_text())["stats"]
    stats["neighbours"] = {float(k): v for k, v in stats["neighbours"].items()}
    return stats


class AllegroCalculator:
    """Our trained full Allegro model as an ASE calculator, for ``teacher_server.py``'s ``mtsallegro:<name>`` kind (the
    teacher Frost runs against on water). A fresh nequip neighbour list per call, as nequip's own ASE calculator does."""

    def __new__(cls, name: str = "allegro", device: str = "cuda"):
        from ase.calculators.calculator import Calculator, all_changes
        from nequip.utils.global_state import set_global_state

        set_global_state(allow_tf32=False)  # as main() does; teacher_server imports this class without main()
        dev = torch.device(device)
        models = load_models(name, saved_stats(name), dev)

        class _Calc(Calculator):
            implemented_properties = ("energy", "forces")

            def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
                from nequip.data import AtomicDataDict

                super().calculate(atoms, properties, system_changes)
                rows = {"d6": to_data(self.atoms, 6.0, dev), "d4": to_data(self.atoms, 4.0, dev)}
                for g in rows.values():
                    g.pop(AtomicDataDict.TOTAL_ENERGY_KEY, None)
                    g.pop(AtomicDataDict.FORCE_KEY, None)
                e, f = predict(models, rows, use_outer=True)
                self.results = {"energy": float(e.detach()), "forces": f.detach().double().cpu().numpy()}

        return _Calc()


def load_models(name, stats, dev, ckpt: str = "last"):
    """``last``: the model at the paper's stopping rule (lr below 1e-6), the default; ``best``: the epoch of lowest
    validation loss. Their loss is ~all force (lambda_E (dE / N)^2 is ~1e-5 against lambda_F <dF^2> ~1e-2), so
    the two have equal forces while ``best`` (epoch 82 / 98) carried a 4.4 meV/atom energy offset that the last
    epochs had removed (1.6 meV/atom)."""
    raw = torch.load(BASE / name / f"{ckpt}.pt", map_location=dev, weights_only=False)
    state = raw["models"] if ckpt == "last" else raw
    if name == "allegro":
        models = {"allegro": build("outer", stats).to(dev)}
    else:
        models = {"inner": build("inner", stats).to(dev), "outer": build("outer", stats, outer_zero=True).to(dev)}
    for k, m in models.items():
        m.load_state_dict(state[k])
        m.eval()
    return models


def md(args, dev):
    """NVE energy conservation and speed, the paper's section 4 protocol: 0.5 fs inner step, outer N; 100 ps after a
    Langevin equilibration with the full model; drift fitted after the first 20 ps (meV/atom/ps); frames kept."""
    frames, split = load_frames()
    start = frames[int(split["test"][0])]
    if args.tile > 1:
        start = start.repeat((args.tile,) * 3)
    n = len(start)
    if args.h_mass != 1.008:  # hydrogen mass is not stated in the paper: tested both ways (unreleased-baseline skill)
        ms = start.get_masses()
        ms[start.get_atomic_numbers() == 1] = args.h_mass
        start.set_masses(ms)
    m = torch.as_tensor(start.get_masses(), dtype=torch.float64, device=dev).unsqueeze(1)
    out = RESULTS / f"mts_allegro_md_{args.tag}.json"
    report = {"atoms": n, "dt_fs": args.dt, "ps": args.ps, "h_mass": args.h_mass, "their_start": args.their_start,
              "arms": {}}
    full = load_models("allegro", saved_stats("allegro"), dev)
    pair = load_models("mts", saved_stats("mts"), dev) if (BASE / "mts" / "best.pt").exists() else None
    eq_path = BASE / f"equilibrated_{n}.pt"
    if args.their_start:
        # Their section 4, as written: "initialize the water system using a structure from the test dataset and a
        # temperature of 300 K, run energy minimization, and then simulate for 100 ps in the NVE ensemble". LAMMPS
        # order (velocity create, minimize, run): minimisation moves positions only, so NVE starts at the energy
        # minimum carrying 300 K of kinetic energy (equipartition then settles near half). Minimised with the full
        # model (FIRE, fmax 0.01 eV/A: their tolerance is not stated); velocity create's defaults: Maxwell-Boltzmann,
        # zero total momentum, scaled to exactly 300 K.
        from ase.optimize import FIRE

        at = start.copy()
        at.calc = AllegroCalculator("allegro", device=str(dev))
        FIRE(at, logfile=None).run(fmax=0.01, steps=5000)
        rng = torch.Generator(device="cpu").manual_seed(args.seed)
        x0 = torch.as_tensor(at.get_positions(), dtype=torch.float64, device=dev)
        v0 = torch.randn(n, 3, generator=rng, dtype=torch.float64).to(dev) * torch.sqrt(KB * 300.0 / m * ACC)
        v0 = v0 - (m * v0).sum(0, keepdim=True) / m.sum()
        t_now = float((m * v0 * v0).sum() / ACC) / ((3 * n - 3) * KB)
        v0 = v0 * math.sqrt(300.0 / t_now)
        report["minimised_energy_eV"] = float(at.get_potential_energy())
    elif eq_path.exists():
        st = torch.load(eq_path, map_location=dev)
        x0, v0 = st["x"], st["v"]
    else:  # BAOAB at 300 K, friction 1/ps, with the full Allegro model
        rng = torch.Generator(device="cpu").manual_seed(0)
        x = torch.as_tensor(start.get_positions(), dtype=torch.float64, device=dev)
        v = (torch.randn(n, 3, generator=rng, dtype=torch.float64).to(dev) * torch.sqrt(KB * 300.0 / m * ACC))
        f_full = Part(full["allegro"], 6.0, start, dev)
        _, f = f_full(x)
        g, dt = 1e-3, args.dt
        c1 = math.exp(-g * dt)
        for _ in range(round(args.equil_ps * 1e3 / dt)):
            v = v + 0.5 * dt * f / m * ACC
            x = x + 0.5 * dt * v
            v = c1 * v + math.sqrt(1 - c1 * c1) * torch.sqrt(KB * 300.0 / m * ACC) * torch.randn(n, 3, generator=rng, dtype=torch.float64).to(dev)
            x = x + 0.5 * dt * v
            _, f = f_full(x)
            v = v + 0.5 * dt * f / m * ACC
        x0, v0 = x, v
        torch.save({"x": x0, "v": v0}, eq_path)
    if args.equil_only:
        return 0
    v_every, x_every = max(1, round(2.0 / args.dt)), max(1, round(20.0 / args.dt))
    keep = {}
    want = [a.strip() for a in args.arms.split(",") if a.strip()]
    arms = ([("allegro", 1)] if "allegro" in want else []) + ([(f"mts{k}", int(k)) for k in want if k.isdigit()] if pair else [])
    for name, k in arms:
        x, v = x0.clone(), v0.clone()
        if name == "allegro":
            parts = {"inner": Part(full["allegro"], 6.0, start, dev)}
        else:
            parts = {"inner": Part(pair["inner"], 4.0, start, dev), "outer": Part(pair["outer"], 6.0, start, dev)}
        e_in, f_in = parts["inner"](x)
        e_out, f_out = parts["outer"](x) if "outer" in parts else (0.0, torch.zeros_like(x))
        dt, steps = args.dt, round(args.ps * 1e3 / args.dt)
        energies, vs, xs = [], [], []
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        step = 0
        while step < steps:
            v = v + 0.5 * k * dt * f_out / m * ACC
            for _ in range(k):
                v = v + 0.5 * dt * f_in / m * ACC
                x = x + dt * v
                e_in, f_in = parts["inner"](x)
                v = v + 0.5 * dt * f_in / m * ACC
                step += 1
                if step % v_every == 0:
                    vs.append(v.float().cpu())
                if step % x_every == 0:
                    xs.append(x.float().cpu())
            if "outer" in parts:
                e_out, f_out = parts["outer"](x)
            v = v + 0.5 * k * dt * f_out / m * ACC
            kin = float(0.5 * (m * v * v).sum() / ACC)
            energies.append((step * dt, (e_in + e_out + kin) / n * 1e3))  # meV/atom
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        t = np.array([a for a, _ in energies]) / 1e3
        e = np.array([b for _, b in energies])
        sel = t >= 20.0
        slope = float(np.polyfit(t[sel], e[sel], 1)[0]) if sel.sum() > 10 else float("nan")
        T = float(np.mean([float(0.5 * (m.cpu() * vv.double() ** 2).sum()) for vv in vs[-100:]]) * 2 / (3 * n * KB) / ACC)
        report["arms"][name] = {"k": k, "ms_per_fs": 1e3 * wall / (steps * dt), "drift_meV_per_atom_per_ps": slope,
                                "energy_range_meV_per_atom": float(e.max() - e.min()), "mean_T_late": T,
                                "rebuilds": {p: q.rebuilds for p, q in parts.items()}}
        print(name, json.dumps(report["arms"][name]), flush=True)
        keep[f"{name}__v"], keep[f"{name}__x"] = np.stack([a.numpy() for a in vs]), np.stack([a.numpy() for a in xs])
        out.write_text(json.dumps(report, indent=1))
    if "allegro" in report["arms"]:  # otherwise speedups are formed when the arms' reports are joined
        base = report["arms"]["allegro"]["ms_per_fs"]
        for a in report["arms"].values():
            a["speedup"] = base / a["ms_per_fs"]
    out.write_text(json.dumps(report, indent=1))
    if args.frames:
        keep.update({"numbers": start.get_atomic_numbers(), "cell": start.cell.array, "masses": start.get_masses(),
                     "v_fs": np.array(v_every * args.dt), "x_fs": np.array(x_every * args.dt)})
        np.savez(RESULTS / f"frames_mts_allegro_{args.tag}.npz", **keep)
    print(json.dumps({k: (a.get("speedup"), a["drift_meV_per_atom_per_ps"]) for k, a in report["arms"].items()}), flush=True)
    return 0


def traj(args, dev):
    """A 300 K water trajectory with the trained full Allegro model, in ``TrajectoryWriter``'s layout (positions
    unwrapped, velocities in ASE units, every 1 fs), for Frost's student labels and start frames on the same teacher:
    Langevin 1/ps, 0.5 fs steps, from the equilibrated state the md stage writes (or a fresh equilibration)."""
    from ase import units

    from frost.trajectory import TrajectoryWriter

    frames, split = load_frames()
    atoms = frames[int(split["test"][0])].copy()
    n = len(atoms)
    m = torch.as_tensor(atoms.get_masses(), dtype=torch.float64, device=dev).unsqueeze(1)
    full = load_models("allegro", saved_stats("allegro"), dev)
    part = Part(full["allegro"], 6.0, atoms, dev)
    eq = torch.load(BASE / f"equilibrated_{n}.pt", map_location=dev)
    x, v = eq["x"], eq["v"]
    rng = torch.Generator(device="cpu").manual_seed(args.seed)
    dt, g = args.dt, 1e-3
    c1 = math.exp(-g * dt)
    sig = torch.sqrt(KB * 300.0 / m * ACC)
    e, f = part(x)
    out = ROOT / "data" / "trajectories" / "water192allegro_T300.h5"
    attrs = {"cell": atoms.cell.array, "numbers": atoms.get_atomic_numbers(), "pbc": np.array([True] * 3),
             "temperature_K": 300.0, "timestep_fs": 1.0, "cutoff": 6.0, "skin": 1.0, "seed": args.seed,
             "teacher": "allegro-water (mts_allegro.py)", "generator": "mts_allegro.py traj"}
    every = max(1, round(1.0 / dt))
    with TrajectoryWriter(out, n_atoms=n, cutoff=6.0, skin=1.0, attrs=attrs) as writer:
        for s in range(1, round(args.ps * 1e3 / dt) + 1):
            v = v + 0.5 * dt * f / m * ACC
            x = x + 0.5 * dt * v
            v = c1 * v + math.sqrt(1 - c1 * c1) * sig * torch.randn(n, 3, generator=rng, dtype=torch.float64).to(dev)
            x = x + 0.5 * dt * v
            e, f = part(x)
            v = v + 0.5 * dt * f / m * ACC
            if s % every == 0:
                xn = x.detach().cpu().numpy()
                atoms.set_positions(xn)
                atoms.wrap()
                temp = float((m * v * v).sum() / ACC) / (3 * n * KB)
                v_ase = (v.detach().cpu().numpy() / units.fs).astype(np.float32)  # A/fs -> A per ASE time unit
                writer.write(s // every - 1, xn, atoms, e, temp, v_ase)
    print(f"-> {out}", flush=True)
    return 0


def split(args, dev):
    """Their Figure 2(b, c) on our rebuild: how large the inner and outer forces are and how fast each changes, along
    ``--frames-n`` consecutive 1 fs frames of the full-model trajectory (``traj``). The outer force is what MTS holds for
    N inner steps, so its power at the O-H stretch (~118 THz here: period 8.46 fs) is what an impulse every N x 0.5 fs
    can resonate with. Bands: < 25 THz (translation, libration), 25-75 (bend), 75-160 (stretch). Output
    ``mts_allegro_split.json``."""
    from frost.trajectory import TrajectoryReader

    stats = saved_stats("mts")
    models = load_models("mts", stats, dev)
    rd = TrajectoryReader(ROOT / "data" / "trajectories" / "water192allegro_T300.h5")
    frames, _ = load_frames()
    atoms = frames[0].copy()
    atoms.set_cell(rd.cell)
    atoms.set_atomic_numbers(rd.numbers)
    t0, n = args.frames_start, args.frames_n
    fin, fout = [], []
    for t in range(t0, t0 + n):
        atoms.set_positions(np.asarray(rd.positions[t], dtype=np.float64))
        atoms.wrap()
        row = {"d6": to_data(atoms, 6.0, dev), "d4": to_data(atoms, 4.0, dev)}
        from nequip.data import AtomicDataDict

        for g in row.values():
            g.pop(AtomicDataDict.TOTAL_ENERGY_KEY, None)
            g.pop(AtomicDataDict.FORCE_KEY, None)
        oi = models["inner"](dict(row["d4"]))
        oo = models["outer"](dict(row["d6"]))
        fin.append(oi[AtomicDataDict.FORCE_KEY].detach().double().cpu().numpy())
        fout.append(oo[AtomicDataDict.FORCE_KEY].detach().double().cpu().numpy())
    rd.close()
    z = np.asarray(atoms.get_atomic_numbers())
    out = {"frames": n, "start": t0, "dt_fs": 1.0, "species": {}}
    freq = np.fft.rfftfreq(n, d=1.0e-15) / 1e12  # THz
    bands = {"0-25": (0, 25), "25-75": (25, 75), "75-160": (75, 160), "160+": (160, 1e9)}
    for name, sel in (("all", np.ones_like(z, bool)), ("H", z == 1), ("O", z == 8)):
        rec = {}
        for label, arr in (("inner", np.asarray(fin)), ("outer", np.asarray(fout))):
            a = arr[:, sel, :]  # (T, n_sel, 3)
            rms = float(np.sqrt((a * a).mean()))
            dif = float(np.sqrt((np.diff(a, axis=0) ** 2).mean()))  # change per 1 fs
            x = a - a.mean(axis=0, keepdims=True)
            w = np.hanning(n)[:, None, None]
            pw = (np.abs(np.fft.rfft(x * w, axis=0)) ** 2).mean(axis=(1, 2))
            tot = pw.sum()
            frac = {b: float(pw[(freq >= lo) & (freq < hi)].sum() / tot) for b, (lo, hi) in bands.items()}
            rec[label] = {"rms_meV_A": 1e3 * rms, "change_per_fs_meV_A": 1e3 * dif, "power_fraction": frac,
                          "band_rms_meV_A": {b: 1e3 * rms * float(np.sqrt(f)) for b, f in frac.items()},
                          "peak_THz": float(freq[1:][np.argmax(pw[1:])])}
        out["species"][name] = rec
        i, o = rec["inner"], rec["outer"]
        print(f"{name}: inner rms {i['rms_meV_A']:.0f} meV/A (change {i['change_per_fs_meV_A']:.0f}/fs) | outer rms "
              f"{o['rms_meV_A']:.0f} (change {o['change_per_fs_meV_A']:.0f}/fs) | outer power fractions "
              + " ".join(f"{b} {f:.3f}" for b, f in o["power_fraction"].items())
              + f" | outer stretch-band rms {o['band_rms_meV_A']['75-160']:.1f} meV/A (inner "
              f"{i['band_rms_meV_A']['75-160']:.0f})", flush=True)
    res = RESULTS / "mts_allegro_split.json"
    tmp = res.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, indent=1))
    tmp.replace(res)
    print("->", res)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("train", "test", "md", "traj", "split"))
    ap.add_argument("--arms", default="allegro,2,4,6,8", help="md: allegro (the full model every step) and outer intervals N")
    ap.add_argument("--equil-only", action="store_true", help="md: equilibrate, save the state, and stop")
    ap.add_argument("--dt", type=float, default=0.5, help="md: inner step, fs (the paper's 0.5)")
    ap.add_argument("--ps", type=float, default=100.0, help="md: NVE length per arm (the paper's 100)")
    ap.add_argument("--equil-ps", type=float, default=10.0)
    ap.add_argument("--tile", type=int, default=1, help="md: tile^3 copies of the 192-atom box (2 gives 1,536)")
    ap.add_argument("--tag", default="nve")
    ap.add_argument("--frames", action="store_true")
    ap.add_argument("--seed", type=int, default=0, help="traj: Langevin seed")
    ap.add_argument("--frames-start", type=int, default=5000, help="split: first trajectory frame")
    ap.add_argument("--their-start", action="store_true",
                    help="md: their stated start (minimised structure, 300 K velocities, straight into NVE)")
    ap.add_argument("--h-mass", type=float, default=1.008, help="md: hydrogen mass, amu (not stated in the paper)")
    ap.add_argument("--frames-n", type=int, default=2048, help="split: consecutive 1 fs frames")
    ap.add_argument("--model", choices=("allegro", "mts"), default="mts")
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--max-epochs", type=int, default=400)
    ap.add_argument("--max-hours", type=float, default=0.0, help="stop cleanly after this long (resume continues)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    from nequip.utils.global_state import set_global_state

    set_global_state(allow_tf32=False)
    dev = torch.device(args.device)
    return {"train": train, "test": test, "md": md, "traj": traj, "split": split}[args.stage](args, dev)


if __name__ == "__main__":
    raise SystemExit(main())
