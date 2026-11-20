"""Empirically test Proposition 3.4 (zero closure at layer 1) on real MACE.

The proposition says the layer-1 edge quantity is a pure function of that one
edge's geometry, because the only node state entering it is the species
embedding, which is constant under the dynamics.  If that is true then:

  * conv_tp input 0 at interaction 0 is bit-identical between two different
    geometries of the same system, and
  * perturbing a single atom changes interaction-0 conv_tp rows only on the
    edges incident to that atom -- zero closure -- while interaction-1 rows
    change on a wide halo, which is closure saturation (Theorem 3.2) in action.

This is the cheapest possible check of the paper's pivotal claim and it should
be run before any cache code is written.

Run:  source scripts/env.sh mace && python e1/probe_zero_closure.py
"""

from __future__ import annotations

import numpy as np
import torch

from frost.systems import si512

TOL = 1e-10


def capture(calc, atoms):
    """Return {site: (inputs, output)} for the two conv_tp modules."""
    model = calc.models[0]
    rec: dict[str, tuple] = {}

    def hook(name):
        def fn(mod, inputs, output):
            rec[name] = (
                [t.detach().clone() for t in inputs if torch.is_tensor(t)],
                output.detach().clone(),
            )

        return fn

    mods = dict(model.named_modules())
    handles = [
        mods[n].register_forward_hook(hook(n))
        for n in ("interactions.0.conv_tp", "interactions.1.conv_tp")
    ]
    atoms = atoms.copy()
    atoms.calc = calc
    atoms.get_potential_energy()
    for h in handles:
        h.remove()
    return rec


def main() -> int:
    from mace.calculators import mace_mp

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    calc = mace_mp(model="medium", device=dev, default_dtype="float64")

    rng = np.random.default_rng(0)
    base = si512()
    # thermal-scale rattle: 0.05 A is a 300 K vibrational amplitude
    warm = base.copy()
    warm.set_positions(base.get_positions() + rng.normal(0, 0.05, base.positions.shape))

    r_base = capture(calc, base)
    r_warm = capture(calc, warm)

    print("=== A. is the layer-1 node-state input constant under motion? ===")
    for site in ("interactions.0.conv_tp", "interactions.1.conv_tp"):
        ib, iw = r_base[site][0], r_warm[site][0]
        if ib[0].shape != iw[0].shape:
            print(
                f"  {site}: edge count changed ({ib[0].shape} vs {iw[0].shape}); "
                "compare on a fixed neighbour list for a clean answer"
            )
            continue
        d = (ib[0] - iw[0]).abs().max().item()
        verdict = (
            "CONSTANT (species embedding only)" if d < TOL else "VARIES with geometry"
        )
        print(f"  {site} input0 max|delta| = {d:.3e}  -> {verdict}")

    print("\n=== B. single-atom perturbation: how far does staleness reach? ===")
    atoms = si512()
    n = len(atoms)
    r0 = capture(calc, atoms)
    moved = 17
    pert = atoms.copy()
    p = pert.get_positions()
    p[moved] += np.array([0.01, 0.0, 0.0])  # 0.01 A, one timestep of thermal motion
    pert.set_positions(p)
    r1 = capture(calc, pert)

    # edge list is rebuilt inside the calculator; recover it for attribution
    from mace.data import AtomicData, config_from_atoms
    from mace.tools import AtomicNumberTable, torch_geometric

    z_table = AtomicNumberTable([int(z) for z in calc.models[0].atomic_numbers])
    cfg = config_from_atoms(atoms)
    d = AtomicData.from_config(cfg, z_table=z_table, cutoff=float(calc.r_max))
    loader = torch_geometric.dataloader.DataLoader([d], batch_size=1, shuffle=False)
    batch = next(iter(loader)).to(dev)
    src, dst = batch.edge_index[0].cpu().numpy(), batch.edge_index[1].cpu().numpy()
    E = len(src)
    print(f"  atoms={n}  edges={E}  mean degree={E / n:.1f}")

    incident = np.where((src == moved) | (dst == moved))[0]
    for site, layer in (("interactions.0.conv_tp", 1), ("interactions.1.conv_tp", 2)):
        a, b = r0[site][1], r1[site][1]
        if a.shape != b.shape:
            print(f"  {site}: shape changed, skipping")
            continue
        delta = (a - b).abs().amax(dim=1).cpu().numpy()
        thresh = 1e-9 * max(1.0, float(a.abs().max()))
        changed = np.where(delta > thresh)[0]
        frac = len(changed) / E
        # which atoms own a changed edge
        touched = (
            np.unique(np.concatenate([src[changed], dst[changed]]))
            if len(changed)
            else []
        )
        print(
            f"  layer {layer}: {len(changed):>6}/{E} edge rows changed "
            f"({frac:6.2%}), touching {len(touched):>4}/{n} atoms"
            f"   [incident edges = {len(incident)}]"
        )

    print(
        "\n  Prediction: layer 1 changes only the ~n_bar incident edges (zero closure,"
    )
    print("  Prop. 3.4); layer 2 changes a large halo (closure saturation, Thm. 3.2).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
