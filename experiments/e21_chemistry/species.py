"""Species and reaction events of every arm of a reactive run (further_expansions_beyond.md 7.4: bonds that break).

    PYTHONPATH=src python experiments/e21_chemistry/species.py data/results/e13/frames_<run>_<tag>.npz [...]

Per frame: a bond graph (minimum image, element-pair cutoffs below), molecules as its connected components, each named by
its formula (Hill order); per arm: species counts against time, the fuel's surviving fraction (and a first-order rate
constant fitted to it), final yields, and bond events (a bond counts as formed or broken only when the change persists
``persist`` frames, so a bond vibrating across the cutoff is not an event). Several frames files (seeds) pool per arm.
Output: species_<first file's stem>.json next to it.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

CUT = {frozenset(("C", "H")): 1.35, frozenset(("O", "H")): 1.25, frozenset(("C", "O")): 1.75,
       frozenset(("C", "C")): 1.85, frozenset(("O", "O")): 1.75, frozenset(("H",)): 1.00,
       frozenset(("N", "H")): 1.30, frozenset(("C", "N")): 1.75, frozenset(("N", "O")): 1.70, frozenset(("N",)): 1.60}
SYMBOL = {1: "H", 6: "C", 7: "N", 8: "O"}


def formula(symbols) -> str:
    c = Counter(symbols)
    head = [k for k in ("C", "H") if "C" in c and c[k]]  # Hill order: C, then H, then the rest alphabetically
    order = head + sorted(k for k in c if k not in head)
    return "".join(f"{k}{c[k] if c[k] > 1 else ''}" for k in order)


def bonds(x: np.ndarray, box: np.ndarray, sym: list, cut: np.ndarray) -> np.ndarray:
    d = x[None, :, :] - x[:, None, :]
    d -= box * np.round(d / box)
    r = np.linalg.norm(d, axis=-1)
    b = r < cut
    np.fill_diagonal(b, False)
    return b


def analyse(xs: np.ndarray, box: np.ndarray, numbers: np.ndarray, x_fs: float, fuel: str, persist: int = 3) -> dict:
    sym = [SYMBOL[int(z)] for z in numbers]
    n = len(sym)
    cut = np.array([[CUT.get(frozenset((a, c)), 0.0) for c in sym] for a in sym])
    counts, graphs = [], []
    for x in xs:
        b = bonds(np.asarray(x, dtype=np.float64), box, sym, cut)
        graphs.append(b)
        _, lab = connected_components(csr_matrix(b), directed=False)
        counts.append(Counter(formula([sym[i] for i in np.flatnonzero(lab == k)]) for k in range(lab.max() + 1)))
    # persistent bond events: a pair's state must hold for `persist` frames on each side of the change
    state = np.stack([g[np.triu_indices(n, 1)] for g in graphs])  # (frames, pairs)
    formed = broken = 0
    for t in range(persist, len(state) - persist + 1):
        before, after = state[t - persist:t], state[t:t + persist]
        stable_before, stable_after = before.all(0) | (~before).all(0), after.all(0) | (~after).all(0)
        changed = stable_before & stable_after & (before[-1] != after[0])
        formed += int((changed & after[0]).sum())
        broken += int((changed & ~after[0]).sum())
    ps = (len(xs) - 1) * x_fs / 1e3
    times = np.arange(len(xs)) * x_fs / 1e3
    fuel_left = np.array([c.get(fuel, 0) for c in counts], dtype=np.float64)
    frac = fuel_left / max(fuel_left[0], 1.0)
    keep = frac > 0.05
    k_rate = float(-np.polyfit(times[keep], np.log(frac[keep]), 1)[0]) if keep.sum() > 3 and frac[keep].min() < 0.95 else 0.0
    species = sorted({k for c in counts for k in c})
    return {"ps": ps, "frames": len(xs), "fuel_fraction": frac[:: max(1, len(frac) // 200)].tolist(),
            "fuel_rate_per_ps": k_rate, "final": dict(counts[-1]), "species_seen": species,
            "bonds_formed": formed, "bonds_broken": broken, "events_per_ps": (formed + broken) / ps if ps else 0.0}


def main(argv) -> int:
    fuel = "CH4"
    per_arm: dict[str, list] = {}
    for f in argv:
        d = np.load(f)
        box = np.diag(d["cell"]).astype(np.float64)
        for k in d.files:
            if k.endswith("__x"):
                per_arm.setdefault(k[:-3], []).append(analyse(d[k], box, d["numbers"], float(d["x_fs"]), fuel))
    out = {"files": argv, "fuel": fuel, "arms": {}}
    for arm, rs in per_arm.items():
        rates = np.array([r["fuel_rate_per_ps"] for r in rs])
        ev = np.array([r["events_per_ps"] for r in rs])
        finals = Counter()
        for r in rs:
            finals.update(r["final"])
        out["arms"][arm] = {"runs": len(rs), "fuel_rate_per_ps": float(rates.mean()),
                            "fuel_rate_sd": float(rates.std()), "events_per_ps": float(ev.mean()),
                            "events_sd": float(ev.std()),
                            "final_mean": {k: v / len(rs) for k, v in finals.most_common(12)}, "per_run": rs}
    res = Path(argv[0]).parent / f"species_{Path(argv[0]).stem.removeprefix('frames_')}.json"
    res.write_text(json.dumps(out, indent=1))
    for arm, row in out["arms"].items():
        top = ", ".join(f"{k} {v:.1f}" for k, v in list(row["final_mean"].items())[:6])
        print(f"{arm:10s} {row['runs']} runs | {fuel} rate {row['fuel_rate_per_ps']:.4f} +- {row['fuel_rate_sd']:.4f}/ps | "
              f"bond events {row['events_per_ps']:.1f} +- {row['events_sd']:.1f}/ps | final: {top}")
    print("->", res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
