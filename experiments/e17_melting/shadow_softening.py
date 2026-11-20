"""Does holding the residual soften the potential more in the liquid than in the crystal? (gaps.md, melting: holding the
residual 20+ steps lowers the slab's T_m by 10-19 K while every clause passes, and neither the whole-slab configurational
temperature nor the per-phase kinetic temperature sees it.)

Frost integrates the residual R = F_T - F_S by impulses (r-RESPA): a kick R(x0) h/2 at an interval's start, k student
steps, a kick R(x1) h/2 at its end, h = k dt. For this Strang splitting the modified (shadow) Hamiltonian is, to second
order (velocity Verlet's, with the student's exact flow as the drift),

    H~ = H - (h^2 / 24 m) |R|^2 - (h^2 / 12 m) F_S . R + (h^2 / 12 m^2) p^T H_R p,     H_R = -grad R,

checked on the split harmonic oscillator (stiffness k_s + k_r, exact flow of k_s, kicks of k_r): the map's exact
invariant has q/p coefficient ratio (k_s + k_r) - h^2 (k_r^2 / 4 + k_s k_r / 3) + O(h^4), which these coefficients give and
the swapped pair (1/12, 1/24) does not. A weakly thermostatted run samples exp(-H~ / kT), whose configurational marginal
adds, per atom,

    dU = -(h^2 / 24 m) |R|^2 - (h^2 / 12 m) F_S . R + (h^2 / 12) <v^T H_R v>.

To first order in free-energy perturbation each phase's chemical potential moves by its mean dU, so the melting point
moves by dT_m = -(<dU>_liq - <dU>_cry) / ds_m, ds_m = L / T_m. The last term needs the residual's curvature along the
velocity: from frame pairs x0, x1 = x0 + v tau, s_i = (1/2) (R_i(x0) - R_i(x1)) . (x1_i - x0_i) ~ (tau^2 / 2) (v^T H_R v)_i,
so it is (h^2 / 6 tau^2) s_i (ballistic over tau; at tau = 20 fs and a 93 fs period, ~15% low).

This evaluates every term per atom on saved frames: the teacher's and the student's per-atom energies and forces at
each frame (R, F_S; s from pairs a lag apart, its total checked against the energy form U_R(x1) - U_R(x0) + R(x0) . dx),
atoms labelled liquid or crystal by the in-plane order of their z slice (the pinning wave vectors), and dT_m predicted for
h = 10, 20 and 31 fs (fixed 10, fixed 20, the certified arm's mean k). Measured (gaps.md, `melting_cuslab_all40.json`):
fixed 10 +2.3 +- 1.9 K, fixed 20 -15.7 +- 1.8, certified -18.6 +- 1.7 (block errors, which understate the seed spread).
The teacher arm's frames give the first-order estimate (unbiased configurations); the other arms' frames check it.

    PYTHONPATH=src python experiments/e17_melting/shadow_softening.py data/results/e13/frames_cuslab_T1250_confT_s7.npz \
        [--arms teacher,fixed20] [--lags 1,2] [--skip-ps 5] [--every 1]
Output: shadow_<frames stem>.json next to the frames file.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from mace import data as mdata
from mace.tools import torch_geometric, utils

from frost.neighbours import edges_for
from frost.pinning import block_se, inplane_kvectors
from frost.students import load_student
from frost.teachers import load_calc

KB_EV = 8.617333262e-5


def phase_labels(x: np.ndarray, cell: np.ndarray, kvecs: np.ndarray, n_bins: int, hi: float, lo: float) -> np.ndarray:
    """Per atom: 1 crystal, -1 liquid, 0 interface, from the in-plane order |sum_j exp(i k . r_j)| / n of its z slice."""
    fz = np.remainder(x[:, 2] / cell[2, 2], 1.0)
    b = np.minimum((fz * n_bins).astype(np.int64), n_bins - 1)
    phase = x @ kvecs.T
    q = np.zeros(n_bins)
    for s in range(n_bins):
        m = b == s
        if m.sum() >= 10:
            q[s] = np.mean(np.abs(np.exp(1j * phase[m]).sum(0))) / m.sum()
    qa = q[b]
    return np.where(qa > hi, 1, np.where(qa < lo, -1, 0))


class Teacher:
    """The teacher's per-atom energies and forces (float32, plain e3nn kernels, positions wrapped into the cell)."""

    def __init__(self, name: str, dev: str):
        self.model = load_calc(name, dev, "float32", enable_cueq=False).models[0].float().eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.z_table = utils.AtomicNumberTable([int(z) for z in self.model.atomic_numbers])
        self.dev = dev

    def __call__(self, numbers, x, cell):
        atoms = Atoms(numbers=numbers, positions=x, cell=cell, pbc=True)
        atoms.wrap()
        cfg = mdata.config_from_atoms(atoms)
        ad = mdata.AtomicData.from_config(cfg, z_table=self.z_table, cutoff=float(self.model.r_max), heads=["Default"])
        data = torch_geometric.Batch.from_data_list([ad]).to(self.dev).to_dict()
        for k, v in list(data.items()):
            if torch.is_tensor(v) and torch.is_floating_point(v):
                data[k] = v.float()
        out = self.model(data, compute_force=True, training=False)
        return (out["node_energy"].detach().double().cpu().numpy(), out["forces"].detach().double().cpu().numpy())


class Student:
    """The student's per-atom energies (``site_energies``) and conservative forces on the run's own edge list."""

    def __init__(self, path: str, numbers, dev: str):
        self.model = load_student(path, dev).float().eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.types = self.model.types(torch.as_tensor(np.asarray(numbers), device=dev))
        self.dev = dev

    def __call__(self, x, cell):
        xo, i, j, sh, _ = edges_for(x, cell, float(self.model.cutoff), device=self.dev, dtype=torch.float32)
        xo = xo.requires_grad_(True)
        e = self.model.site_energies(xo, self.types, i, j, sh)
        (g,) = torch.autograd.grad(e.sum(), xo)
        return e.detach().double().cpu().numpy(), (-g).detach().double().cpu().numpy()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frames")
    ap.add_argument("--student", default="data/models/students/cuslab_T1300_f45c16.pt")
    ap.add_argument("--teacher", default="mp-medium")
    ap.add_argument("--arms", default="teacher,fixed20,fixed10,adaptive")
    ap.add_argument("--lags", default="1,2")
    ap.add_argument("--skip-ps", type=float, default=5.0, help="the pinning runs' own equilibration skip")
    ap.add_argument("--every", type=int, default=1, help="use every n-th x0 frame")
    ap.add_argument("--bins", type=int, default=24)
    ap.add_argument("--cells-xy", default="6,6")
    ap.add_argument("--latent-ev", type=float, default=0.074, help="latent heat per atom (the teacher's fit)")
    ap.add_argument("--tm-k", type=float, default=1261.3, help="the teacher's melting point")
    args = ap.parse_args()
    dev = "cuda"
    torch.set_default_dtype(torch.float32)
    path = Path(args.frames)
    d = np.load(path)
    numbers = d["numbers"]
    masses = d["masses"].astype(np.float64)
    x_fs = float(d["x_fs"])
    kvecs = inplane_kvectors(d["cell"], tuple(int(v) for v in args.cells_xy.split(",")))
    lags = [int(v) for v in args.lags.split(",")]
    teacher, student = Teacher(args.teacher, dev), Student(args.student, numbers, dev)
    ds_m = args.latent_ev / args.tm_k
    out = {"frames": str(path), "student": args.student, "teacher": args.teacher, "x_fs": x_fs, "lags": lags,
           "ds_m_eV_per_K": ds_m, "arms": {}}
    res_path = path.parent / f"shadow_{path.stem.removeprefix('frames_')}.json"
    for arm in args.arms.split(","):
        if f"{arm}__x" not in d.files:
            print(f"{arm}: not in {path.name}")
            continue
        xs, cells = d[f"{arm}__x"].astype(np.float64), d[f"{arm}__cells"]
        first = int(round(args.skip_ps * 1e3 / x_fs))
        need = sorted({j + lag for j in range(first, len(xs) - max(lags), args.every) for lag in (0, *lags)})
        t0 = time.time()
        cache, forces_s = {}, {}
        for n, f in enumerate(need):
            et, ft = teacher(numbers, xs[f], cells[f])
            es, fs_ = student(xs[f], cells[f])
            cache[f] = (et - es, ft - fs_)
            forces_s[f] = fs_
            if n == 0 or (n + 1) % 200 == 0:
                print(f"{arm}: {n + 1}/{len(need)} frames, {time.time() - t0:.0f} s", flush=True)
        rows = {}
        for lag in lags:
            sl, sc, si, tot_trap, tot_energy, dr_l, dr_c = [], [], [], [], [], [], []
            r2_l, r2_c, fr_l, fr_c = [], [], [], []
            for j in range(first, len(xs) - max(lags), args.every):
                (u0, r0), (u1, r1) = cache[j], cache[j + lag]
                dx = xs[j + lag] - xs[j]
                s = 0.5 * ((r0 - r1) * dx).sum(1)  # eV per atom
                lab = phase_labels(xs[j], cells[j], kvecs, args.bins, 0.45, 0.2)
                sl.append(s[lab == -1].mean())
                sc.append(s[lab == 1].mean())
                si.append(s[lab == 0].mean() if (lab == 0).any() else np.nan)
                tot_trap.append(s.mean())
                tot_energy.append(((u1 - u0).sum() + (r0 * dx).sum()) / len(s))
                fs0 = forces_s[j]
                r2 = (r0 * r0).sum(1) / masses  # |R|^2 / m, eV^2 / (A^2 amu)
                fr = (fs0 * r0).sum(1) / masses
                r2_l.append(r2[lab == -1].mean())
                r2_c.append(r2[lab == 1].mean())
                fr_l.append(fr[lab == -1].mean())
                fr_c.append(fr[lab == 1].mean())
                dr = np.sqrt(((r1 - r0) ** 2).sum(1) / 3.0)  # eV/A per component
                dr_l.append(dr[lab == -1].mean())
                dr_c.append(dr[lab == 1].mean())
            sl, sc = np.array(sl), np.array(sc)
            diff = sl - sc
            tau = lag * x_fs
            # (a held constant force would move T_m by -(<s>_liq - <s>_cry) / 3 / ds_m; Frost kicks, see impulse_shadow)
            pred = -(diff.mean() / 3.0) / ds_m
            row = {"tau_fs": tau, "pairs": len(sl),
                   "s_liq_meV": 1e3 * sl.mean(), "s_liq_se": 1e3 * block_se(sl),
                   "s_cry_meV": 1e3 * sc.mean(), "s_cry_se": 1e3 * block_se(sc),
                   "s_int_meV": 1e3 * float(np.nanmean(si)),
                   "liq_minus_cry_meV": 1e3 * diff.mean(), "liq_minus_cry_se": 1e3 * block_se(diff),
                   "total_trapezoid_meV": 1e3 * float(np.mean(tot_trap)),
                   "total_energy_form_meV": 1e3 * float(np.mean(tot_energy)),
                   "dR_liq_meV_per_A": 1e3 * float(np.mean(dr_l)), "dR_cry_meV_per_A": 1e3 * float(np.mean(dr_c)),
                   "dTm_K_if_constant_held_force": pred,
                   }
            # the impulse scheme's shadow term per phase at h (fs): conv turns h^2 / m (fs^2 / amu) times eV^2 / A^2
            # into eV; the curvature term from s over tau
            conv = 1.0 / 103.6427
            r2_l, r2_c, fr_l, fr_c = map(np.array, (r2_l, r2_c, fr_l, fr_c))
            shadow = {}
            for h in (10.0, 20.0, 31.0):
                du_l = conv * h * h * (-r2_l / 24.0 - fr_l / 12.0) + (h * h / (6.0 * tau * tau)) * sl
                du_c = conv * h * h * (-r2_c / 24.0 - fr_c / 12.0) + (h * h / (6.0 * tau * tau)) * sc
                dd = du_l - du_c
                shadow[str(int(h))] = {
                    "dU_liq_meV": 1e3 * du_l.mean(), "dU_cry_meV": 1e3 * du_c.mean(),
                    "terms_liq_meV": [1e3 * conv * h * h * (-r2_l.mean() / 24.0), 1e3 * conv * h * h * (-fr_l.mean() / 12.0),
                                      1e3 * (h * h / (6.0 * tau * tau)) * sl.mean()],
                    "terms_cry_meV": [1e3 * conv * h * h * (-r2_c.mean() / 24.0), 1e3 * conv * h * h * (-fr_c.mean() / 12.0),
                                      1e3 * (h * h / (6.0 * tau * tau)) * sc.mean()],
                    "dTm_K": -dd.mean() / ds_m, "dTm_se": block_se(dd) / ds_m}
                print(f"{arm:9s}   impulse shadow at h {h:.0f} fs (tau {tau:.0f}): dU liq {1e3 * du_l.mean():+.4f}, cry "
                      f"{1e3 * du_c.mean():+.4f} meV/atom -> dT_m {shadow[str(int(h))]['dTm_K']:+.2f} +- "
                      f"{shadow[str(int(h))]['dTm_se']:.2f} K (terms liq {shadow[str(int(h))]['terms_liq_meV']}, cry "
                      f"{shadow[str(int(h))]['terms_cry_meV']})", flush=True)
            row["impulse_shadow"] = shadow
            row["R_rms_liq_meV_per_A"] = 1e3 * float(np.sqrt(np.mean(r2_l * masses.mean() / 3.0)))
            row["R_rms_cry_meV_per_A"] = 1e3 * float(np.sqrt(np.mean(r2_c * masses.mean() / 3.0)))
            rows[str(lag)] = row
            print(f"{arm:9s} tau {tau:4.0f} fs: s liq {row['s_liq_meV']:+.3f} +- {row['s_liq_se']:.3f}, cry "
                  f"{row['s_cry_meV']:+.3f} +- {row['s_cry_se']:.3f} meV/atom (liq - cry {row['liq_minus_cry_meV']:+.3f} "
                  f"+- {row['liq_minus_cry_se']:.3f}); total trapezoid {row['total_trapezoid_meV']:+.3f}, energy form "
                  f"{row['total_energy_form_meV']:+.3f}; |dR| liq {row['dR_liq_meV_per_A']:.1f} cry "
                  f"{row['dR_cry_meV_per_A']:.1f} meV/A",
                  flush=True)
        out["arms"][arm] = rows
        tmp = res_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(out, indent=1))
        tmp.replace(res_path)
    print("->", res_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
