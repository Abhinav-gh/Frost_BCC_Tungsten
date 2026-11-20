"""E7: profile the residual R = F_T - F_S end to end, per species and per frequency.

The multirate loop holds R constant between teacher calls. Water refuses and
silica breaks at k = 16, and a staleness scalar cannot say why. This measures,
on consecutive 1 fs frames of an E1 trajectory, with the teacher and a trained
student evaluated on every frame:

- **Per-species staleness** of the held residual at k = 1, 2, 4, 8, 16, in force
  and in acceleration units. Which atoms carry the error?
- **The residual's spectrum** per species, against the teacher force's. Is the
  residual fast because it tracks a vibration the student misses?
- **Three ways to carry R across an outer step**, scored the same way:
    held     R(t0 + j) = R(t0), what the loop does now;
    linear   R extrapolated in time from the previous outer step;
    affine   R(t0 + j) = R(t0) + A_i (x_i(t0 + j) - x_i(t0)), with a 3x3 A_i per
             atom fitted by ridge least squares from the previous outer steps'
             teacher calls only, as an online loop could.
  The affine form is the hypothesis: if the student's error is a local
  stiffness mismatch (the O-H stretch in water), R is linear in each atom's
  displacement and the affine carry removes most of the staleness.

Every statistic covers every atom of every frame in the windows.

Output: data/results/e7/residual_profile_<run>.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TRAJ = ROOT / "data" / "trajectories"
OUT_DIR = ROOT / "data" / "results" / "e7"
KS = (1, 2, 4, 8, 16)


def rms_rows(x: torch.Tensor) -> float:
    """RMS of per-atom vector norms over every leading index."""
    return float(x.pow(2).sum(-1).mean().sqrt())


CARRIES = ("held", "linear", "affine", "bonded")


def fit_map(dR, dX, ridge: float):
    """Per-atom ridge fit of dR = A dX over anchors: (M, atoms, 3) -> (atoms, 3, 3)."""
    rx = torch.einsum("mai,maj->aij", dR, dX)
    xx = torch.einsum("mai,maj->aij", dX, dX)
    lam = ridge * xx.diagonal(dim1=1, dim2=2).mean(-1, keepdim=True).unsqueeze(-1) + 1e-12
    eye = torch.eye(3, dtype=dR.dtype, device=dR.device)
    return torch.linalg.solve(xx + lam * eye, rx.transpose(1, 2)).transpose(1, 2)


def carry_errors(R, X, B, k: int, history: int, ridge: float):
    """Mean staleness over outer steps of length k for each carry.

    R, X: (frames, atoms, 3) residual and unwrapped positions within one window.
    B: the same positions relative to each atom's bonded partners (the bond
    coordinate); equal to X for atoms without partners.
    Returns summed squared errors and counts, per atom, for every carry.
    """
    n = R.shape[0]
    sq = {name: torch.zeros(R.shape[1], dtype=R.dtype, device=R.device) for name in CARRIES}
    count = 0
    first = history * k
    for t0 in range(first, n - k, k):
        anchors = [t0 - m * k for m in range(1, history + 1)]
        dR = torch.stack([R[a] - R[t0] for a in anchors])  # (M, atoms, 3)
        A = fit_map(dR, torch.stack([X[a] - X[t0] for a in anchors]), ridge)
        Ab = fit_map(dR, torch.stack([B[a] - B[t0] for a in anchors]), ridge)
        slope = (R[t0] - R[t0 - k]) / k
        for j in range(1, k + 1):
            truth = R[t0 + j]
            sq["held"] += (truth - R[t0]).pow(2).sum(-1)
            sq["linear"] += (truth - R[t0] - j * slope).pow(2).sum(-1)
            pred = R[t0] + torch.einsum("aij,aj->ai", A, X[t0 + j] - X[t0])
            sq["affine"] += (truth - pred).pow(2).sum(-1)
            pred_b = R[t0] + torch.einsum("aij,aj->ai", Ab, B[t0 + j] - B[t0])
            sq["bonded"] += (truth - pred_b).pow(2).sum(-1)
        count += k
    return sq, count


def bond_partners(numbers: np.ndarray, x0: np.ndarray, cell: np.ndarray, bond: float = 1.25):
    """For every atom, the atoms covalently bonded to it within ``bond`` A (minimum image)."""
    from frost.neighbours import edges_for

    _, i, j, _, _ = edges_for(x0, cell, bond, dtype=torch.float64)
    partners = [[] for _ in numbers]
    for a, b in zip(i.tolist(), j.tolist(), strict=True):
        partners[a].append(b)
    return partners


def bond_coordinates(X: torch.Tensor, partners, cell_t: torch.Tensor) -> torch.Tensor:
    """Each atom's position relative to the mean of its bonded partners (minimum image)."""
    B = X.clone()
    inv = torch.linalg.inv(cell_t)
    for a, ps in enumerate(partners):
        if not ps:
            continue
        d = X[:, a:a + 1, :] - X[:, ps, :]  # (frames, partners, 3)
        d = d - torch.round(d @ inv) @ cell_t
        B[:, a, :] = d.mean(1)
    return B


def main() -> int:
    import h5py
    from mace.calculators import mace_mp

    from frost.neighbours import edges_for
    from frost.student import energy_and_forces
    from frost.students import load_student
    from frost.teacher import frame_atoms, mace_forces

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="water216_T300")
    ap.add_argument("--student", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--starts", default="4000,7000")
    ap.add_argument("--window", type=int, default=160, help="consecutive 1 fs frames per window")
    ap.add_argument("--history", type=int, default=4, help="previous outer steps for the affine fit")
    ap.add_argument("--ridge", type=float, default=1e-2)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    dev = args.device

    with h5py.File(TRAJ / f"{args.run}.h5", "r") as fh:
        numbers = np.asarray(fh.attrs["numbers"], dtype=np.int64)
        cell = np.asarray(fh.attrs["cell"], dtype=np.float64)
        starts = [int(s) for s in args.starts.split(",")]
        windows = [np.asarray(fh["positions"][s:s + args.window], dtype=np.float64) for s in starts]

    calc = mace_mp(model="medium", device=dev, default_dtype="float32", enable_cueq=True)
    student = load_student(args.student, dev)
    types = student.types(torch.as_tensor(numbers, device=dev))
    cutoff = float(student.config["cutoff"])
    masses = torch.as_tensor(frame_atoms(numbers, windows[0][0], cell).get_masses(),
                             dtype=torch.float64, device=dev)
    inv_cell = np.linalg.inv(cell)
    species = sorted(set(numbers.tolist()))

    per_k = {k: {name: torch.zeros(len(numbers), dtype=torch.float64, device=dev)
                 for name in CARRIES} for k in KS}
    partners = bond_partners(numbers, windows[0][0], cell)
    cell_t = torch.as_tensor(cell, device=dev)
    counts = dict.fromkeys(KS, 0)
    spectra_r, spectra_f, spectra_s = [], [], []
    force_rms = []
    for frames in windows:
        R, F, X = [], [], []
        x_prev, x_unwrapped = None, None
        for pos in frames:
            if x_prev is None:
                x_unwrapped = pos.copy()
            else:  # minimum-image step, so X is continuous across the box
                d = pos - x_prev
                d -= np.rint(d @ inv_cell) @ cell
                x_unwrapped = x_unwrapped + d
            x_prev = pos
            atoms = frame_atoms(numbers, pos, cell)
            f_t, _, _ = mace_forces(calc, atoms, dev)
            xw, i, j, sh, _ = edges_for(atoms.get_positions(), cell, cutoff, dev)
            _, f_s = energy_and_forces(student, xw, types, i, j, sh)
            F.append(torch.as_tensor(f_t, device=dev))
            R.append(torch.as_tensor(f_t, device=dev) - f_s.detach().double())
            X.append(torch.as_tensor(x_unwrapped, device=dev))
        R, F, X = torch.stack(R), torch.stack(F), torch.stack(X)
        B = bond_coordinates(X, partners, cell_t)
        force_rms.append(rms_rows(F))
        for k in KS:
            sq, c = carry_errors(R, X, B, k, args.history, args.ridge)
            for name in sq:
                per_k[k][name] += sq[name]
            counts[k] += c
        # spectra: power of each atom's residual and force, summed over components
        spectra_r.append(torch.fft.rfft(R - R.mean(0), dim=0).abs().pow(2).sum(-1))
        spectra_f.append(torch.fft.rfft(F - F.mean(0), dim=0).abs().pow(2).sum(-1))
        S = F - R  # the student force on the same frames
        spectra_s.append(torch.fft.rfft(S - S.mean(0), dim=0).abs().pow(2).sum(-1))

    freq = np.fft.rfftfreq(args.window, d=1.0) * 1e3  # THz, from 1 fs frames
    power_r = torch.stack(spectra_r).mean(0)  # (freqs, atoms)
    power_f = torch.stack(spectra_f).mean(0)
    power_s = torch.stack(spectra_s).mean(0)
    numbers_t = torch.as_tensor(numbers, device=dev)
    report = {"run": args.run, "student": student.config, "window_fs": args.window,
              "starts": starts, "history": args.history, "ridge": args.ridge,
              "teacher_force_rms_meV_per_A": 1e3 * float(np.mean(force_rms)), "species": {}}
    for z in [*species, "all"]:
        mask = torch.ones_like(numbers_t, dtype=torch.bool) if z == "all" else numbers_t == z
        m = masses[mask]
        entry = {"n_atoms": int(mask.sum()), "mass_amu": float(m.mean())}
        for k in KS:
            c = counts[k] * int(mask.sum())
            for name in CARRIES:
                sq = per_k[k][name][mask]
                entry.setdefault(f"{name}_force_meV_per_A", {})[k] = 1e3 * float((sq.sum() / c).sqrt())
                entry.setdefault(f"{name}_accel_meV_per_A_amu", {})[k] = 1e3 * float(
                    ((sq / m.pow(2)).sum() / c).sqrt())
        pr = power_r[:, mask].mean(1).cpu().numpy()
        pf = power_f[:, mask].mean(1).cpu().numpy()
        entry["residual_peak_THz"] = float(freq[1:][np.argmax(pr[1:])])
        entry["force_peak_THz"] = float(freq[1:][np.argmax(pf[1:])])
        entry["residual_power_above_40THz"] = float(pr[freq > 40].sum() / pr[1:].sum())
        entry["force_power_above_40THz"] = float(pf[freq > 40].sum() / pf[1:].sum())
        entry["spectrum_THz"] = freq[::2].tolist()
        entry["residual_spectrum"] = (pr[::2] / pr[1:].max()).tolist()
        entry["force_spectrum"] = (pf[::2] / pf[1:].max()).tolist()
        ps = power_s[:, mask].mean(1).cpu().numpy()
        # absolute power above every physical mode: if the student's floor is far above
        # the teacher's, the residual's fast part is student roughness, not physics
        entry["floor_above_150THz"] = {"teacher": float(pf[freq > 150].mean()),
                                       "student": float(ps[freq > 150].mean()),
                                       "residual": float(pr[freq > 150].mean())}
        report["species"][str(z)] = entry

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"residual_profile_{args.run}.json"
    out.write_text(json.dumps(report, indent=2))
    for z, e in report["species"].items():
        a = e["held_accel_meV_per_A_amu"]
        b = e["bonded_accel_meV_per_A_amu"]
        print(f"{args.run} Z={z:>3} m={e['mass_amu']:5.1f} | accel held/linear/affine/bonded at k=2: "
              f"{a[2]:.2f}/{e['linear_accel_meV_per_A_amu'][2]:.2f}/{e['affine_accel_meV_per_A_amu'][2]:.2f}"
              f"/{b[2]:.2f} k=8: {a[8]:.2f}/{e['linear_accel_meV_per_A_amu'][8]:.2f}/"
              f"{e['affine_accel_meV_per_A_amu'][8]:.2f}/{b[8]:.2f} | residual peak "
              f"{e['residual_peak_THz']:.0f} THz,"
              f" >40 THz {e['residual_power_above_40THz']:.2f} (force {e['force_power_above_40THz']:.2f})",
              flush=True)
    print(f"-> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
