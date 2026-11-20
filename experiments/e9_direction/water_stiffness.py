"""How much of a water residual's staleness is a stiffness error along the bonds? (offline, before any runtime change)

Water's held residual R = F_T - F_S goes stale by ~60 meV/A in 4 fs (MACE-OFF23 small against a small MACE
student), because R oscillates with the O-H stretch: a student whose bond stiffness is off by a fraction delta
leaves a residual delta k_OH dr, and the certificate's hydrogen budget needs delta ~ 0.35%. If the error is mostly
a stiffness error along the fast internal coordinates, the runtime can hold R as a local harmonic model instead
of a constant:

    R(x) ~ R(x_n) + sum_b dK_b (r_b(x) - r_b(x_n)) e_b(x),

with r_b the length of bond b, e_b its unit vector (signed per atom), and dK fitted online by least squares from
the residual changes the refreshes already measure (no extra teacher call). This measures, on teacher frames,
the staleness over k fs of (a) the constant hold and (b) the harmonic hold with one dK per bond type (O-H), fitted
on windows other than the one scored, and (c) the same with dK fitted per bond from its own history.

Output: data/results/e9/water_stiffness_<tag>.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="water216off_T300")
    ap.add_argument("--student", default="data/models/students/water216off_T300_s35c16e1.pt")
    ap.add_argument("--starts", default="8000,8500,9000,9500")
    ap.add_argument("--window", type=int, default=24, help="fs per window (frames are 1 fs apart)")
    ap.add_argument("--ks", default="2,4,6,8,12")
    ap.add_argument("--tag", default="v1")
    args = ap.parse_args()

    from frost.neighbours import edges_for
    from frost.student import energy_and_forces
    from frost.students import load_student
    from frost.teacher import frame_atoms, mace_forces
    from frost.teachers import load_calc, teacher_of

    dev = "cuda"
    with h5py.File(ROOT / "data" / "trajectories" / f"{args.run}.h5", "r") as fh:
        numbers = np.asarray(fh.attrs["numbers"])
        cell = np.asarray(fh.attrs["cell"], dtype=np.float64)
        teacher_name = teacher_of(fh.attrs)
        starts = [int(s) for s in args.starts.split(",")]
        frames = {s: np.asarray(fh["positions"][s:s + args.window + 1]) for s in starts}
    calc = load_calc(teacher_name, dev, "float32", enable_cueq=False)
    student = load_student(ROOT / args.student, dev)
    types = student.types(torch.as_tensor(numbers, device=dev))
    cutoff = float(student.config["cutoff"])
    o = np.nonzero(numbers == 8)[0]
    h = np.nonzero(numbers == 1)[0]
    # O-H bonds from the first frame (each H to its nearest O, minimum image)
    x0 = frames[starts[0]][0]
    d = x0[h][:, None, :] - x0[o][None, :, :]
    inv = np.linalg.inv(cell)
    d -= np.round(d @ inv) @ cell
    bond_o = o[np.argmin((d ** 2).sum(-1), axis=1)]
    bonds = np.stack([h, bond_o], 1)  # (n_bonds, 2): H, O
    masses = np.where(numbers == 1, 1.008, 15.999)

    def residual(x):
        atoms = frame_atoms(numbers, x, cell)
        f_t, _, _ = mace_forces(calc, atoms, dev)
        xw, ei, ej, sh, _ = edges_for(atoms.get_positions(), cell, cutoff, dev)
        _, f_s = energy_and_forces(student, xw, types, ei, ej, sh)
        return f_t - f_s.detach().double().cpu().numpy()

    def bond_geom(x):
        v = x[bonds[:, 0]] - x[bonds[:, 1]]  # unwrapped frames: no image needed within a molecule
        r = np.linalg.norm(v, axis=1)
        return r, v / r[:, None]

    data = {}
    for s, fr in frames.items():
        rs = np.stack([residual(x) for x in fr])  # (T+1, n, 3)
        geo = [bond_geom(x) for x in fr]
        data[s] = (rs, np.stack([g[0] for g in geo]), np.stack([g[1] for g in geo]))

    def harmonic(r_n, dr, e, dk):
        """sum_b dK_b dr_b e_b on H (+) and O (-), as forces (n, 3)."""
        out = np.zeros(len(numbers) * 3).reshape(-1, 3)
        f = (dk * dr)[:, None] * e
        np.add.at(out, bonds[:, 0], f)
        np.add.at(out, bonds[:, 1], -f)
        return out

    def accel_rms(df):  # meV/(A amu), RMS over atoms, the certificate's units
        return float(np.sqrt(((df / masses[:, None]) ** 2).sum(-1).mean()) * 1e3)

    out = {"run": args.run, "student": args.student, "n_bonds": len(bonds), "ks": {}}
    for k in (int(v) for v in args.ks.split(",")):
        rows = {"constant": [], "global_dK": [], "per_bond_dK": []}
        for s in starts:
            # fit dK on the other windows: least squares of the bond-projected residual change on dr
            num = den = 0.0
            for s2 in starts:
                if s2 == s:
                    continue
                rs2, r2, e2 = data[s2]
                for t in range(0, args.window - k + 1, k):
                    drs = rs2[t + k] - rs2[t]
                    proj = ((drs[bonds[:, 0]] - drs[bonds[:, 1]]) * e2[t]).sum(-1) / 2.0
                    dr = r2[t + k] - r2[t]
                    num += float((proj * dr).sum())
                    den += float((dr * dr).sum())
            dk_global = num / max(den, 1e-30)
            rs, r, e = data[s]
            hist_num = np.zeros(len(bonds))
            hist_den = np.zeros(len(bonds))
            for t in range(0, args.window - k + 1, k):
                stale = rs[t + k] - rs[t]
                dr = r[t + k] - r[t]
                rows["constant"].append(accel_rms(stale))
                rows["global_dK"].append(accel_rms(stale - harmonic(None, dr, e[t], np.full(len(bonds), dk_global))))
                dk_b = np.where(hist_den > 0, hist_num / np.maximum(hist_den, 1e-30), dk_global)
                rows["per_bond_dK"].append(accel_rms(stale - harmonic(None, dr, e[t], dk_b)))
                proj = ((stale[bonds[:, 0]] - stale[bonds[:, 1]]) * e[t]).sum(-1) / 2.0
                hist_num += proj * dr
                hist_den += dr * dr
        out["ks"][k] = {m: float(np.mean(v)) for m, v in rows.items()} | {"dK_global_eV_A2": dk_global}
        print(f"k {k:2d} fs: staleness RMS meV/(A amu) constant {out['ks'][k]['constant']:.2f} | global dK "
              f"{out['ks'][k]['global_dK']:.2f} | per-bond dK {out['ks'][k]['per_bond_dK']:.2f} "
              f"(dK {dk_global:+.2f} eV/A^2)", flush=True)
    res = ROOT / "data" / "results" / "e9"
    res.mkdir(parents=True, exist_ok=True)
    (res / f"water_stiffness_{args.tag}.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
