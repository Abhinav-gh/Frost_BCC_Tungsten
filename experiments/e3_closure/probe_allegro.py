"""Allegro closure test: does an edge-local cache delete force, as in MACE layer 2?

The design document (corollary 3.5) treats Allegro as having zero closure at
every depth, because it passes no learned state between atoms. Reading
`allegro/nn/_allegro.py` shows something narrower. Each tensor product takes

    x1  the edge's own tensor track, carried from the previous layer, and
    x2  an environment embedding: a weighted sum over ALL edges that share the
        edge's centre atom, with weights computed from those edges' latents.

So there is no coupling between different centre atoms, but every edge is
coupled to every other edge around its own atom. A cache that stores only an
edge's dependence on its own geometry cannot see those neighbours change. For
MACE layer 2 the equivalent blind spot deleted 36% of the force at zero
displacement (plan section 1.5). This measures whether Allegro's within-atom
coupling does the same.

Method, over every edge at once (never a single edge or atom): displace all
atoms randomly, then substitute all five tensor products in one of three ways.
Each product is exactly linear in x1 and in x2 separately (plan section 1.6),
so the first-order expansion is exact without any Jacobian:

    first       g0 + TP(dx1, x2_0) + TP(x1_0, dx2)   exact to first order
    stale_env   g0 + TP(dx1, x2_0)                   environment held stale:
                                                     what an edge-local cache gives
    zeroth      g0                                   naive reference

The sharp signal is the error at zero displacement. `first` must be zero there
up to float precision. If `stale_env` sits on a floor that does not shrink with
displacement, the within-atom environment path carries force and an edge-local
cache deletes it.

Cell: 64-atom diamond Si, rattled by 0.05 A so forces are not zero by symmetry.
Allegro-L needs about 35 GB on GPU for the 512-atom cell, more than is free on a
shared A6000. Displacements that change the edge list are skipped and reported,
since the reference and current rows would no longer be the same edges.

Usage (through the guarded launcher)
  scripts/safe_run.sh probe_allegro .venvs/frost-nequip/bin/python \\
      experiments/e3_closure/probe_allegro.py --device cuda
"""

from __future__ import annotations

import argparse
import inspect
import json
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "data" / "models" / "nequip" / "Allegro-MP-L-0.1.nequip.zip"
OUT = ROOT / "data" / "results" / "e3" / "allegro_closure.json"

DISPLACEMENTS = [0.0, 1e-4, 1e-3, 5e-3, 0.01, 0.02]
CONFIGS = ("first", "stale_env", "zeroth")


class BilinearSubstituter:
    """Replace one Allegro tensor product's output with a cached surrogate."""

    def __init__(self, module: torch.nn.Module):
        self.module = module
        self.mode = "record"
        self.ref: tuple | None = None
        self._busy = False
        self._handle = module.register_forward_hook(self._hook, with_kwargs=True)

    def _hook(self, mod, args, kwargs, output):
        if self._busy:
            return None
        x1, x2, rest = args[0], args[1], args[2:]
        if self.mode == "record":
            self.ref = (x1.detach().clone(), x2.detach().clone(), output.detach().clone())
            return None
        if self.mode == "exact" or self.ref is None:
            return None
        x10, x20, g0 = self.ref
        if x1.shape != x10.shape or x2.shape != x20.shape:
            raise RuntimeError("edge set changed between reference and current")
        if self.mode == "zeroth":
            return g0
        self._busy = True
        try:
            out = g0 + mod(x1 - x10, x20, *rest, **kwargs)
            if self.mode == "first":
                out = out + mod(x10, x2 - x20, *rest, **kwargs)
            elif self.mode != "stale_env":
                raise ValueError(self.mode)
        finally:
            self._busy = False
        return out

    def close(self):
        self._handle.remove()


def load(device: str):
    from nequip.data import AtomicDataDict, from_ase
    from nequip.data import transforms as T  # noqa: N812
    from nequip.model import ModelFromPackage, ModelTypeNamesFromPackage

    names = ModelTypeNamesFromPackage(str(PKG))
    model = ModelFromPackage(package_path=str(PKG))["sole_model"].eval().to(device)

    mp = inspect.signature(T.ChemicalSpeciesToAtomTypeMapper).parameters
    kw = {}
    if "model_type_names" in mp:
        kw["model_type_names"] = names
    if "chemical_species_to_atom_type_map" in mp:
        kw["chemical_species_to_atom_type_map"] = {n: n for n in names}
    mapper = T.ChemicalSpeciesToAtomTypeMapper(**kw)
    np_ = inspect.signature(T.NeighborListTransform).parameters
    nl = T.NeighborListTransform(**({"r_max": 6.0} if "r_max" in np_ else {next(iter(np_)): 6.0}))

    def evaluate(atoms):
        data = from_ase(atoms)
        for t in (mapper, nl):
            data = t(data)
        data = AtomicDataDict.to_(data, torch.device(device))
        edges = data[AtomicDataDict.EDGE_INDEX_KEY].detach().clone()
        out = model(data)
        forces = out[AtomicDataDict.FORCE_KEY].detach().cpu().double().numpy()
        return forces, edges

    return model, evaluate


def rms(f: np.ndarray) -> float:
    return float(np.sqrt((f**2).sum(1).mean()))


def main() -> int:
    from ase.build import bulk

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--cells", type=int, default=2)
    ap.add_argument("--per-layer", action="store_true",
                    help="hold the environment stale in one tensor product at a time")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    model, evaluate = load(args.device)
    subs = [BilinearSubstituter(m) for n, m in model.named_modules()
            if any(n.endswith(f"tps.{k}") for k in range(16))]

    rng = np.random.default_rng(0)
    base = bulk("Si", "diamond", a=5.43, cubic=True).repeat((args.cells,) * 3)
    base.set_positions(base.get_positions() + rng.normal(0, 0.05, base.positions.shape))
    p0 = base.get_positions()
    direction = rng.normal(size=p0.shape)
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)

    rows: list[dict] = []
    per_layer: list[dict] = []
    print(f"Allegro-MP-L, Si{len(base)}, {len(subs)} tensor products substituted, "
          f"every atom displaced")
    print(f"{'dr (A)':>8} | " + " ".join(f"{c:>11}" for c in CONFIGS) + "   (meV/A)")
    for dr in DISPLACEMENTS:
        t0 = time.perf_counter()
        moved = base.copy()
        moved.set_positions(p0 + (dr / np.sqrt(2)) * direction)

        for s in subs:
            s.mode = "record"
        _, edges_ref = evaluate(base)
        for s in subs:
            s.mode = "exact"
        f_exact, edges_cur = evaluate(moved)
        if edges_cur.shape != edges_ref.shape or not torch.equal(edges_cur, edges_ref):
            print(f"{dr:8.4f} | edge list changed; skipped")
            continue

        errs = {}
        for c in CONFIGS:
            for s in subs:
                s.mode = c
            f, _ = evaluate(moved)
            errs[c] = rms(f - f_exact) * 1e3
        rows.append({"displacement_A": dr, "exact_force_rms_meV_per_A": rms(f_exact) * 1e3,
                     **{f"err_{c}_meV_per_A": v for c, v in errs.items()}})
        print(f"{dr:8.4f} | " + " ".join(f"{errs[c]:11.4f}" for c in CONFIGS)
              + f"   [{time.perf_counter() - t0:.0f}s]", flush=True)

    if args.per_layer:
        # At zero displacement the stale value equals the exact value, so any
        # error is the deleted gradient through that layer's environment. Every
        # edge is still substituted in the chosen layer.
        for s in subs:
            s.mode = "record"
        evaluate(base)
        for s in subs:
            s.mode = "exact"
        f0, _ = evaluate(base)
        f0_rms = rms(f0) * 1e3
        print("\nper layer: environment stale in ONE tensor product, all edges, zero displacement")
        for k, sk in enumerate(subs):
            for s in subs:
                s.mode = "exact"
            sk.mode = "stale_env"
            f, _ = evaluate(base)
            e = rms(f - f0) * 1e3
            per_layer.append({"tp": k, "err_meV_per_A": e, "force_fraction": e / f0_rms})
            print(f"  tps.{k}: {e:9.3f} meV/A  ({e / f0_rms:.1%} of the force)", flush=True)

    for s in subs:
        s.close()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"n_atoms": len(base), "rows": rows, "per_layer": per_layer}, indent=2))

    zero = [r for r in rows if r["displacement_A"] == 0.0]
    if zero:
        z = zero[0]
        frac = z["err_stale_env_meV_per_A"] / z["exact_force_rms_meV_per_A"]
        print(f"\nat zero displacement: first {z['err_first_meV_per_A']:.4f}, "
              f"stale_env {z['err_stale_env_meV_per_A']:.4f} meV/A "
              f"({frac:.1%} of the force), zeroth {z['err_zeroth_meV_per_A']:.4f}")
        verdict = ("FLOOR: the within-atom environment path carries force; an "
                   "edge-local cache deletes it" if z["err_stale_env_meV_per_A"] > 1.0
                   else "NO FLOOR: the environment path carries negligible force")
        print(verdict)
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
