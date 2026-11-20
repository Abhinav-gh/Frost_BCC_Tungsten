"""Screw dislocation cores in a BCC [111] dipole cell, frame by frame (2026-11-18, further_expansions_beyond.md 7.5).

The cell is ``frost.systems.bcc_screw_dipole``'s (x [11-2], y [-110], z [111]): every atom keeps its index from the perfect
oriented crystal, so its z-displacement from its perfect site is known. Atoms sit in [111] columns (one atom per b along
z); a column's displacement is the mean over its atoms (the whole crystal's drift removed first, trap 32). Around each
smallest triangle of neighbouring columns (oriented counter-clockwise) the Burgers circuit is the sum of the three edge
differences, each wrapped to (-b/2, b/2]: +b or -b where a core sits inside, 0 elsewhere. An edge to the nearest image
of a column takes the cell's z-tilt (the image sits one tilt lower per cell). A core's position is the mean of its
triangles' centroids, unwrapped in time.

    python experiments/e20_mechanics/screw_cores.py data/trajectories/<run>.h5 [--every 20]
    python experiments/e20_mechanics/screw_cores.py data/results/e13/frames_<run>_<tag>.npz [--every 1]
Output: data/results/e20/screw_cores_<stem>.json (per source or arm: cores found per frame, the + and - core paths,
their net glide and mean square displacement along x).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "results" / "e20"


def setup(cell: np.ndarray, n_atoms: int):
    """The perfect oriented crystal (bcc_screw_dipole's atom order), each atom's column, column positions, the
    counter-clockwise smallest triangles of columns, and b."""
    from ase.lattice.cubic import BodyCenteredCubic

    lx, ly, lz = cell[0, 0], cell[1, 1], cell[2, 2]
    for a in np.linspace(2.5, 4.0, 15001):
        reps = np.array([lx / (a * np.sqrt(6)), ly / (a * np.sqrt(2)), lz / (a * np.sqrt(3) / 2)])
        if np.abs(reps - np.round(reps)).max() < 1e-3:
            break
    else:
        raise SystemExit("cell is not an oriented BCC [11-2] x [-110] x [111] cell")
    nx, ny, nz = (int(v) for v in np.round(reps))
    p0 = BodyCenteredCubic(directions=[[1, 1, -2], [-1, 1, 0], [1, 1, 1]], size=(nx, ny, nz), symbol="W",
                           latticeconstant=a, pbc=True).get_positions()
    if len(p0) != n_atoms:
        raise SystemExit(f"the perfect crystal has {len(p0)} atoms, the cell {n_atoms}")
    frac = p0[:, :2] / [lx, ly]
    frac -= np.floor(frac + 1e-6)  # one column at x = 0 and x = lx after rounding would be two (2026-11-18)
    cols, col_of = np.unique(np.round(frac * [lx, ly], 2), axis=0, return_inverse=True)
    col_of = col_of.ravel()
    d = cols[None, :, :] - cols[:, None, :]
    d[..., 0] -= lx * np.round(d[..., 0] / lx)
    d[..., 1] -= ly * np.round(d[..., 1] / ly)
    r = np.linalg.norm(d, axis=-1)
    np.fill_diagonal(r, np.inf)
    adj = r < 1.15 * r[r > 0.1].min()
    tris = set()
    for i in range(len(cols)):
        nb = np.flatnonzero(adj[i])
        for j in nb:
            for k in nb:
                if j < k and adj[j, k]:
                    tris.add(tuple(sorted((i, int(j), int(k)))))
    tris = np.array(sorted(tris))
    # counter-clockwise, by the minimum-image edge vectors
    e1 = d[tris[:, 0], tris[:, 1]]
    e2 = d[tris[:, 0], tris[:, 2]]
    cw = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0] < 0
    tris[cw] = tris[cw][:, [0, 2, 1]]
    centroids = cols[tris[:, 0]] + (d[tris[:, 0], tris[:, 1]] + d[tris[:, 0], tris[:, 2]]) / 3.0
    return p0, col_of, cols, tris, centroids, a * np.sqrt(3) / 2


def circulation(pos: np.ndarray, p0, col_of, cols, tris, cell, b) -> np.ndarray:
    """The Burgers circuit (A) around every triangle for one frame's positions (n, 3).

    An atom wrapped across the cell's x or y face (the runtime's frames hold such atoms; the teacher's trajectory does
    not) has its z moved by that face's tilt, which is not a multiple of b and corrupts its column's mean (2026-11-18: it
    put spurious +-b pairs along the y face of every runtime frame). Each atom's image offset is read from its in-plane
    distance to its perfect site and undone before its displacement is taken."""
    lx, ly, t1, t2 = cell[0, 0], cell[1, 1], cell[0, 2], cell[1, 2]
    n1 = np.round((pos[:, 0] - p0[:, 0]) / lx)
    n2 = np.round((pos[:, 1] - p0[:, 1]) / ly)
    z = pos[:, 2] - n1 * t1 - n2 * t2
    dz = z - p0[:, 2]
    dz = dz - dz.mean()
    colz = np.bincount(col_of, weights=dz, minlength=len(cols)) / np.bincount(col_of, minlength=len(cols))
    circ = np.zeros(len(tris))
    for u, v in ((0, 1), (1, 2), (2, 0)):
        i, j = tris[:, u], tris[:, v]
        sx = np.round((cols[j, 0] - cols[i, 0]) / lx)
        sy = np.round((cols[j, 1] - cols[i, 1]) / ly)
        e = (colz[j] - sx * t1 - sy * t2) - colz[i]
        circ += (e + b / 2) % b - b / 2
    return circ


def frame_cores(circ: np.ndarray, centroids: np.ndarray, lx: float, ly: float, b: float, prev=None,
                link: float = 6.0) -> dict:
    """Core positions {+1: (x, y), -1: (x, y)} for one frame, from clusters of hit triangles (2026-11-18).

    A hop's transition, or a core spread along its 16 A line, shows as a triplet of neighbouring triangles (two of its
    sign and one of the other, net +-b), so a core is a cluster: hit triangles (|circ| > b/2) linked within ``link`` A
    (minimum image), its net circulation the sum, its position the circulation-weighted mean of its centroids
    (sum s_i c_i / sum s_i). Net-zero clusters (a local +-b pair) are not cores. With two clusters of one sign the one
    nearest the previous position is taken."""
    hit = np.flatnonzero(np.abs(circ) > 0.5 * b)
    out = {}
    if not len(hit):
        return out
    c = centroids[hit]
    q = np.round(circ[hit] / b)
    n = len(hit)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            dd = c[j] - c[i]
            dd[0] -= lx * np.round(dd[0] / lx)
            dd[1] -= ly * np.round(dd[1] / ly)
            if np.hypot(*dd) < link:
                parent[find(i)] = find(j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    for members in groups.values():
        net = int(q[members].sum())
        if net not in (1, -1):
            continue
        ref = c[members[0]]
        dd = c[members] - ref
        dd[:, 0] -= lx * np.round(dd[:, 0] / lx)
        dd[:, 1] -= ly * np.round(dd[:, 1] / ly)
        pos = ref + (q[members, None] * dd).sum(0) / net
        if net in out and prev is not None and prev.get(net) is not None:
            def dist(a):
                e = a - prev[net]
                e[0] -= lx * np.round(e[0] / lx)
                e[1] -= ly * np.round(e[1] / ly)
                return np.hypot(*e)
            if dist(pos) >= dist(out[net]):
                continue
        out[net] = pos
    return out


def track(frames: np.ndarray, cell, dt_fs: float, every: int):
    p0, col_of, cols, tris, centroids, b = setup(cell, frames.shape[1])
    lx, ly = cell[0, 0], cell[1, 1]
    counts, paths = [], {+1: [], -1: []}
    for f in range(0, len(frames), every):
        circ = circulation(frames[f], p0, col_of, cols, tris, cell, b)
        row = {}
        last = {sg: (paths[sg][-1] if paths[sg] else None) for sg in (+1, -1)}
        found = frame_cores(circ, centroids, lx, ly, b, last)
        for sign in (+1, -1):
            row[sign] = 1 if sign in found else 0
            if sign in found:
                pos = np.array(found[sign], dtype=float)
                prev = paths[sign][-1] if paths[sign] else None
                if prev is not None:  # unwrap in time
                    step = pos - prev
                    step[0] -= lx * np.round(step[0] / lx)
                    step[1] -= ly * np.round(step[1] / ly)
                    pos = prev + step
                paths[sign].append(pos)
            else:
                paths[sign].append(paths[sign][-1] if paths[sign] else np.array([np.nan, np.nan]))
        counts.append((row[+1], row[-1]))
    out = {"b": b, "frames": len(counts), "dt_fs": dt_fs * every, "counts_plus_minus": counts}
    for sign, name in ((+1, "plus"), (-1, "minus")):
        p = np.array(paths[sign])
        x = p[:, 0] - p[0, 0]
        lags = np.arange(1, max(2, len(x) // 4))
        msd = [float(np.mean((x[k:] - x[:-k]) ** 2)) for k in lags]
        slope = float(np.polyfit(lags * dt_fs * every / 1e3, msd, 1)[0]) if len(lags) > 2 else None
        y = p[:, 1] - p[0, 1]
        # hops (2026-11-18): a short random walk's MSD slope is noisy, its hop count is not; steps between positions
        # 100 fs apart longer than 1 A are hops (one Peierls hop is 2.6 A), their summed length the path walked
        stride = max(1, int(round(100.0 / (dt_fs * every))))
        q = p[::stride]
        st = np.diff(q, axis=0)
        r = np.hypot(st[:, 0], st[:, 1]) if len(st) else np.zeros(0)
        hops, path_a = int((r > 1.0).sum()), float(r.sum())
        msd_xy = [float(np.mean((x[k:] - x[:-k]) ** 2 + (y[k:] - y[:-k]) ** 2)) for k in lags]
        slope_xy = float(np.polyfit(lags * dt_fs * every / 1e3, msd_xy, 1)[0]) if len(lags) > 2 else None
        out[name] = {"start": p[0].tolist(), "end": p[-1].tolist(), "net_x": float(x[-1]),
                     "net_y": float(y[-1]), "net_r": float(np.hypot(x[-1], y[-1])),
                     "rms_x": float(np.sqrt(np.mean(x ** 2))),
                     "D_x_A2_per_ps": slope / 2.0 if slope is not None else None,
                     # in-plane diffusivity (glide on the three {110} planes moves a core in y too)
                     "D_xy_A2_per_ps": slope_xy / 4.0 if slope_xy is not None else None,
                     "hops_per_100ps": hops * 1e5 / max(1.0, (len(p) - 1) * dt_fs * every), "path_A": path_a}
    good = sum(1 for c in counts if c[0] >= 1 and c[1] >= 1)
    out["frames_with_both_cores"] = good
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source")
    ap.add_argument("--every", type=int, default=20, help="use every Nth stored frame")
    args = ap.parse_args()
    path = Path(args.source)
    report = {"source": str(path)}
    if path.suffix == ".h5":
        import h5py

        with h5py.File(path, "r") as fh:
            cell = np.asarray(fh.attrs["cell"], dtype=np.float64)
            x = np.asarray(fh["positions"][:: args.every])
            dt = float(fh.attrs.get("dt_fs", 1.0))
        report["teacher_trajectory"] = track(x, cell, dt * args.every, 1)
    else:
        d = np.load(path)
        cell = np.asarray(d["cell"], dtype=np.float64)
        for arm in sorted({k.split("__")[0] for k in d.files if k.endswith("__x")}):
            report[arm] = track(np.asarray(d[f"{arm}__x"], dtype=np.float64), cell, float(d["x_fs"]), args.every)
    for name, r in report.items():
        if isinstance(r, dict):
            print(f"{name}: both cores in {r['frames_with_both_cores']} of {r['frames']} frames | "
                  f"hops per 100 ps + {r['plus']['hops_per_100ps']:.0f} / - {r['minus']['hops_per_100ps']:.0f} | path + "
                  f"{r['plus']['path_A']:.0f} / - {r['minus']['path_A']:.0f} A | net + ({r['plus']['net_x']:+.1f}, "
                  f"{r['plus']['net_y']:+.1f}) - ({r['minus']['net_x']:+.1f}, {r['minus']['net_y']:+.1f}) A", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"screw_cores_{path.stem}.json"
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, indent=1, default=float))
    tmp.replace(out)
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
