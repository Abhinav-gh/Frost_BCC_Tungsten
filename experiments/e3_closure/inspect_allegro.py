"""Inspect Allegro's tensor products before testing whether they can be cached.

Allegro is the design's primary target because it passes no learned state
between atoms, so each edge's features should depend only on geometry around
that edge's own centre atom (Frost.pdf corollary 3.5). The MACE layer-2 probe
showed what goes wrong when a cached edge silently depends on other edges'
state: the force path through that state is deleted. Before running the same
all-edge test on Allegro, this script establishes three facts it depends on:

  1. what each tensor product ("tps.k", a Contracter) takes as input, and
  2. whether each is linear in each tensor input -- the property that lets a
     first-order cache be written exactly without materialising a Jacobian;
  3. the baseline energy and force scale on the same Si512 cell.

Every check runs on all edges of the cell at once, never on a single edge or a
single atom.

Output: data/results/e3/allegro_structure.json

Usage (GPU, through the guarded launcher; running Allegro-L on CPU is what
once pushed the shared server to a load average of 658)
  scripts/safe_run.sh inspect_allegro .venvs/frost-nequip/bin/python \
      experiments/e3_closure/inspect_allegro.py --device cuda
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "data" / "models" / "nequip" / "Allegro-MP-L-0.1.nequip.zip"
OUT = ROOT / "data" / "results" / "e3" / "allegro_structure.json"


def build(pkg: Path, atoms, device: str, r_max: float = 6.0):
    from nequip.data import AtomicDataDict, from_ase
    from nequip.data import transforms as T  # noqa: N812
    from nequip.model import ModelFromPackage, ModelTypeNamesFromPackage

    names = ModelTypeNamesFromPackage(str(pkg))
    model = ModelFromPackage(package_path=str(pkg))["sole_model"].eval()

    mp = inspect.signature(T.ChemicalSpeciesToAtomTypeMapper).parameters
    kw = {}
    if "model_type_names" in mp:
        kw["model_type_names"] = names
    if "chemical_species_to_atom_type_map" in mp:
        kw["chemical_species_to_atom_type_map"] = {n: n for n in names}
    mapper = T.ChemicalSpeciesToAtomTypeMapper(**kw)

    np_ = inspect.signature(T.NeighborListTransform).parameters
    nl_kw = {"r_max": r_max} if "r_max" in np_ else {next(iter(np_)): r_max}
    nl = T.NeighborListTransform(**nl_kw)

    data = from_ase(atoms)
    for t in (mapper, nl):
        data = t(data)
    return model.to(device), AtomicDataDict.to_(data, torch.device(device)), AtomicDataDict


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--cells", type=int, default=2,
        help="diamond Si cells per side; every edge of the cell is still used. "
        "4 (512 atoms) needs ~35 GB on GPU for Allegro-L, 2 (64 atoms) about 4.5",
    )
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    from ase.build import bulk

    atoms = bulk("Si", "diamond", a=5.43, cubic=True).repeat((args.cells,) * 3)
    model, data, ADD = build(PKG, atoms, args.device)
    n_edges = int(data[ADD.EDGE_INDEX_KEY].shape[1])
    n_atoms = len(atoms)
    print(f"Allegro-MP-L on Si{n_atoms}: {n_edges} edges", flush=True)

    captured: dict = {}
    hooks = []
    for name, mod in model.named_modules():
        if any(name.endswith(f"tps.{k}") for k in range(16)):

            def mk(n):
                def hook(m, args, kwargs, output):
                    # keep non-tensor arguments too: Contracter.forward needs an
                    # integer scatter_dim_size alongside its tensors
                    captured[n] = (
                        m,
                        [a.detach() if torch.is_tensor(a) else a for a in args],
                        dict(kwargs),
                        output.detach(),
                    )

                return hook

            hooks.append(mod.register_forward_hook(mk(name), with_kwargs=True))
    out = model(data)
    for h in hooks:
        h.remove()

    forces = out[ADD.FORCE_KEY].detach()
    force_rms = float(forces.norm(dim=1).pow(2).mean().sqrt()) * 1e3
    energy = float(out[ADD.TOTAL_ENERGY_KEY])
    print(f"E/N = {energy / n_atoms:.4f} eV, |F|rms = {force_rms:.1f} meV/A\n", flush=True)
    del out, forces
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    report = {
        "n_atoms": n_atoms,
        "n_edges": n_edges,
        "energy_eV": energy,
        "force_rms_meV_per_A": force_rms,
        "tps": {},
    }
    gen = torch.Generator().manual_seed(0)
    for name in sorted(captured):
        mod, args_, kwargs_, o = captured[name]
        ins = [a for a in args_ if torch.is_tensor(a)]
        try:
            sig = str(inspect.signature(mod.forward))
        except (TypeError, ValueError):
            sig = "?"
        entry = {
            "class": type(mod).__name__,
            "forward": sig,
            "input_shapes": [list(t.shape) for t in ins],
            "output_shape": list(o.shape),
            "rows_equal_edges": [int(t.shape[0]) == n_edges for t in ins],
            "linearity_residual": [],
        }
        # linearity in each tensor input, tested on every row at once:
        # f(x + y) - f(x) - f(y) should vanish if f is linear in that input
        with torch.no_grad():
            base = mod(*args_, **kwargs_)
            scale = float(base.abs().max()) or 1.0
            for k, x in enumerate(args_):
                if not torch.is_tensor(x) or not x.is_floating_point():
                    entry["linearity_residual"].append(None)
                    continue
                y = torch.randn(x.shape, generator=gen, dtype=x.dtype).to(x.device)
                a = list(args_)
                b = list(args_)
                c = list(args_)
                a[k] = x + y
                c[k] = y
                resid = mod(*a, **kwargs_) - mod(*b, **kwargs_) - mod(*c, **kwargs_)
                entry["linearity_residual"].append(float(resid.abs().max()) / scale)
        report["tps"][name] = entry
        lin = ", ".join(
            "n/a" if r is None else f"{r:.1e}" for r in entry["linearity_residual"]
        )
        print(f"{name}: {entry['class']}{sig}")
        print(f"    inputs {entry['input_shapes']} -> {entry['output_shape']}")
        print(f"    one row per edge: {entry['rows_equal_edges']}")
        print(
            f"    linearity residual per input: {lin}   (~0 means linear)", flush=True
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
