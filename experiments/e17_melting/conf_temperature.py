"""Configurational temperature of every arm's configurations under the teacher's potential (gaps.md, melting: the
k-dependent T_m shift that no other clause sees).

    PYTHONPATH=src python experiments/e17_melting/conf_temperature.py data/results/e13/frames_cuslab_T1250_<tag>.npz \
        [--every 5] [--probes 2] [--skip-ps 2]

For each arm's saved positions (every --every-th frame after --skip-ps), the teacher's energy gradient and a Hutchinson
estimate of its Hessian trace (frost.confT, double backward through MACE with plain e3nn kernels, float32: float64 needed
>35 GB at 2,880 atoms and ran out of memory beside another job), and from them
kT_conf = <|grad U|^2> / <laplacian U> with a block standard error. Canonical configurations at the thermostat's T read T;
a held force that biases the configurational distribution reads otherwise. The teacher arm is the control: it must read
T within its error. Per-frame cells (NPT) are used when the frames file has them.
Output: confT_<frames stem>.json next to the frames file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from mace import data as mdata
from mace.tools import torch_geometric, utils

from frost.confT import conf_temperature, grad_and_trace
from frost.teachers import load_calc


def energy_fn_for(model, atoms: Atoms, dev: str):
    z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
    cfg = mdata.config_from_atoms(atoms)
    ad = mdata.AtomicData.from_config(cfg, z_table=z_table, cutoff=float(model.r_max), heads=["Default"])
    data = torch_geometric.Batch.from_data_list([ad]).to(dev).to_dict()
    for k, v in list(data.items()):
        if torch.is_tensor(v) and torch.is_floating_point(v):
            data[k] = v.float()

    def energy(x):
        d = dict(data)
        d["positions"] = x
        return model(d, compute_force=False, training=True)["energy"].sum()  # training=True keeps the graph

    return energy, data["positions"].detach()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frames")
    ap.add_argument("--teacher", default="mp-medium")
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--probes", type=int, default=2)
    ap.add_argument("--skip-ps", type=float, default=2.0)
    ap.add_argument("--temperature", type=float, default=0.0, help="K; 0: from the file name (_T<K>_)")
    args = ap.parse_args()
    dev = "cuda"
    torch.set_default_dtype(torch.float32)
    path = Path(args.frames)
    temp = args.temperature or float(path.stem.split("_T")[1].split("_")[0])
    d = np.load(path)
    numbers, x_fs = d["numbers"], float(d["x_fs"])
    model = load_calc(args.teacher, dev, "float32", enable_cueq=False).models[0].float().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    gen = torch.Generator().manual_seed(0)
    out = {"frames": str(path), "teacher": args.teacher, "T_K": temp, "every": args.every, "probes": args.probes, "arms": {}}
    for key in sorted(k for k in d.files if k.endswith("__x")):
        arm = key[:-3]
        xs = d[key]
        cells = d[f"{arm}__cells"] if f"{arm}__cells" in d.files else None
        first = int(round(args.skip_ps * 1e3 / x_fs))
        g2s, trs = [], []
        for f in range(first, len(xs), args.every):
            cell = cells[f] if cells is not None else d["cell"]
            atoms = Atoms(numbers=numbers, positions=xs[f].astype(np.float64), cell=cell, pbc=True)
            energy, x0 = energy_fn_for(model, atoms, dev)
            g2, tr, _ = grad_and_trace(energy, x0, probes=args.probes, generator=gen)
            g2s.append(g2)
            trs.append(tr)
            del energy, x0
            torch.cuda.empty_cache()
        t, se = conf_temperature(g2s, trs)
        out["arms"][arm] = {"T_conf_K": t, "se_K": se, "frames": len(g2s), "dT_K": t - temp}
        print(f"{arm:10s} T_conf {t:7.1f} +- {se:4.1f} K ({t - temp:+6.1f} K against {temp:.0f} K, {len(g2s)} frames)",
              flush=True)
    res = path.parent / f"confT_{path.stem.removeprefix('frames_')}.json"
    res.write_text(json.dumps(out, indent=1))
    print("->", res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
