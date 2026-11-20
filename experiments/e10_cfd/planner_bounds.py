"""The planner's deployment test for a learned CFD component, from measurements alone.

Before anything is trained, two cheap measurements bound what a learned model on an n-grid
can win against the classical solver at equal accuracy:
- the **classical frontier**: tau and wall clock of every classical configuration (grid,
  time step, filter, closure), reduced to its Pareto envelope;
- the **bound B(n)**: the initial-condition ceiling (``ceiling.py``, ``ceiling3d.py``) or,
  for tracers, the Markov bound (``tracer_frontier.py``): what perfect dynamics on the
  n-grid state reach.

The most a learned model on n can win, even with a free network, is

    W_max(n) = C(B(n)) / P(n),

where C(a) is the classical cost of reaching accuracy a (the envelope, log-cost
interpolated in tau) and P(n) the cost of the n-grid physics alone. W_max assumes the
model closes the whole gap from the n-grid physics S(n) to B(n); no trained correction
here or in the literature has closed more than F_BEST = 0.30 of it, so the decision uses

    W_f(n) = C(S(n) + f (B(n) - S(n))) / P(n),   f = F_BEST,

still with a free network. The network is not free: its cost per unit time N is profiled
once (the every-step hybrid's cost less the physics), and the decision uses

    W_f,net(n) = C(S(n) + f (B(n) - S(n))) / (P(n) + N / k),   k = K_HOLD = 16,

which assumes the hold costs no accuracy (optimistic). Below 1.5 the planner refuses: it
keeps the classical solver. Trained hybrids are scored the same way (W = C(tau) / cost,
and their closed share f) to check the prediction.

Output: data/results/e10/planner_bounds.json

Usage (CPU, reads results only)
  PYTHONPATH=src .venvs/frost-mhd/bin/python experiments/e10_cfd/planner_bounds.py
"""

from __future__ import annotations

import json
import math
import sys
from itertools import pairwise
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from w1_eval import RES  # noqa: E402

REFUSE_BELOW = 1.5
K_HOLD = 16  # Frost's usual certified hold: the network's cost per unit time is divided by it
F_BEST = 0.30  # the largest share of a bound's gap a trained correction has closed (W3 32^3; Dresdner's ~2x is the exp filter's)


def envelope(points):
    """Pareto envelope of (cost, tau): cheaper first, each point more accurate than all cheaper ones."""
    out = []
    for cost, tau, name in sorted(points):
        if not out or tau > out[-1][1]:
            out.append((cost, tau, name))
    return out


def classical_cost(env, a: float):
    """Cost of reaching tau >= a on the envelope (log-linear in tau); None past its end."""
    if a <= env[0][1]:
        return env[0][0]
    for (c0, t0, _), (c1, t1, _) in pairwise(env):
        if t0 < a <= t1:
            return math.exp(math.log(c0) + (a - t0) / (t1 - t0) * (math.log(c1) - math.log(c0)))
    return None


def fmt(x):
    return "beyond the frontier" if x is None else f"{x:.2f}"


def load(name):
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else None


def w1(batch: str):
    fr = load(f"frontier_exp_fine{batch}.json")["rows"]
    pts = [(r["ms_per_0.007"], r["tau_mean"], k) for k, r in fr.items() if r["finite"]]
    ceil = {**load("ceiling.json"), **load("ceiling_mid.json")}
    bound = {}
    for r in ceil.values():
        bound[r["n_ic"]] = max(bound.get(r["n_ic"], 0.0), r["tau"])
    own = {n: min((c, t) for c, t, k in pts if k.startswith(f"spectral{n}exp")) for n in bound}
    base, phys = {n: v[0] for n, v in own.items()}, {n: v[1] for n, v in own.items()}
    sp = load("w1split_split_k1.json")
    hyb = []
    if sp and batch in sp["timing"]:
        ms = sp["timing"][batch]["ms_per_0.007"]
        hyb = [(k, 64, sp["accuracy"][k]["tau_mean"], ms[k]) for k in ms if k.startswith("split:")]
    return {"workload": f"W1 velocity, 2D Kolmogorov Re 1000, batch {batch}", "unit": "ms per 0.007",
            "points": pts, "bound": bound, "base": base, "physics_tau": phys, "hybrids": hyb, "dim": 2}


def w3(batch: str):
    fr = load("frontier3d_base.json")["rows"]
    pts = [(r["ms_per_0.04"][batch], r["tau_mean"], k) for k, r in fr.items() if r["finite"]]
    ceil = load("ceiling3d.json")
    bound, censored = {}, {}
    for r in ceil.values():
        if r["tau"] >= bound.get(r["n_ic"], 0.0):
            bound[r["n_ic"]] = r["tau"]
            censored[r["n_ic"]] = r["tau_se"] == 0.0  # every trajectory reached the 6.0 window end
    own = {n: min((c, t) for c, t, k in pts if k.startswith(f"{n}:1:none")) for n in bound}
    base, phys = {n: v[0] for n, v in own.items()}, {n: v[1] for n, v in own.items()}
    hyb = []
    for tag in ("w3_48epd_k1", "w3_48epd_k16"):
        d = load(f"w3split_{tag}.json")
        if d and batch in d["timing"]:
            ms = d["timing"][batch]["ms_per_0.04"]
            hyb += [(f"{tag}:{k}", 48, d["accuracy"][k]["tau_mean"], ms[k]) for k in ms if k.startswith("split:")]
    return {"workload": f"W3 velocity, 3D forced isotropic turbulence, batch {batch}", "unit": "ms per 0.04",
            "points": pts, "bound": bound, "base": base, "physics_tau": phys, "hybrids": hyb, "censored": censored,
            "dim": 3}


def tracers(tags=("gate", "fill")):
    """W1-T. The bound on an n-grid is the best measured on it: the Markov run (sharp projection
    every interval) or the n-grid spectral scheme, which beats it (its filter is gentler). A
    learned scheme that infers sub-grid filaments could pass it, so it is a reference, not a proof."""
    rows = {}
    for tag in tags:
        d = load(f"tracer_frontier_{tag}.json")
        if d is not None:
            rows.update({k: r for k, r in d["rows"].items() if not k.startswith("reference")})
    if not rows:
        return None

    def size(k):
        return int(k[len(k.rstrip("0123456789")):])

    out = []
    for admissible, label in ((lambda r: True, "any scheme"), (lambda r: r["min"] >= -1e-4, "positive schemes")):
        pts = [(r["ms_per_0.014"], r["tau_k21"], k) for k, r in rows.items() if "ms_per_0.014" in r and admissible(r)]
        sizes = sorted({size(k) for k in rows if k.startswith("markov")})
        bound, base, phys = {}, {}, {}
        for n in sizes:
            on_n = [(r["tau_k21"], k) for k, r in rows.items() if size(k) == n]
            bound[n] = max(t for t, _ in on_n)
            # the scheme a learned one corrects (``li_pp`` sits on limited WENO-Z), else the cheapest finite volume
            fv = [(c, t) for c, t, k in pts if size(k) == n and not k.startswith(("spectral", "markov"))]
            own = [(c, t) for c, t, k in pts if k == f"weno5z_pp{n}"]
            if own or fv:
                base[n], phys[n] = (own or [min(fv)])[0]
        hyb = []
        d = load("tracer_learned_li1.json")
        if d is not None:
            hyb = [(name, 64, r["tau_k21"], r["ms_per_0.014"]) for name, r in d["rows"].items()]
        out.append({"workload": f"W1-T tracers ({label}), 64 fields, tau on |k| <= 21", "unit": "ms per 0.014",
                    "points": pts, "bound": {n: bound[n] for n in base}, "base": base, "physics_tau": phys,
                    "hybrids": hyb, "dim": 2})
    return out


def report(w):
    env = envelope(w["points"])
    rows = {"workload": w["workload"], "unit": w["unit"],
            "envelope": [{"config": k, "cost": c, "tau": t} for c, t, k in env], "grids": {}, "hybrids": {}}
    print(f"\n== {w['workload']} ({w['unit']})")
    print("   classical envelope: " + ", ".join(f"{k} {t:.2f} @ {c:.3g}" for c, t, k in env))
    for n in sorted(w["bound"]):
        b, p, s0 = w["bound"][n], w["base"][n], w["physics_tau"][n]
        c = classical_cost(env, b)
        wmax = None if c is None else c / p
        cf = classical_cost(env, s0 + F_BEST * (b - s0))
        wf = None if cf is None else cf / p
        cens = w.get("censored", {}).get(n, False)
        costs = [cost for _, m, _, cost in w["hybrids"] if m == n]
        net = max(costs) - p if costs else None  # the every-step hybrid is the dearest
        if net is None and w["hybrids"]:  # profiled on another grid: a convolution's cost scales with the cells
            m = w["hybrids"][0][1]
            net = (max(cost for _, mm, _, cost in w["hybrids"] if mm == m) - w["base"][m]) * (n / m) ** w["dim"]
        wnet = None if (cf is None or net is None) else cf / (p + net / K_HOLD)
        decide = wnet if wnet is not None else wf
        rows["grids"][n] = {"bound": b, "bound_censored": cens, "physics_tau": s0, "physics_cost": p,
                            "classical_cost_at_bound": c, "w_max": wmax, "w_f": wf, "f": F_BEST,
                            "network_per_unit": net, "w_f_net": wnet,
                            "decision": "refuse" if (decide is not None and decide < REFUSE_BELOW) else "candidate"}
        print(f"   n {n:4d}: physics {s0:.2f} @ {p:.3g} | bound {b:.2f}{' (censored)' if cens else ''} | "
              f"W_max {fmt(wmax)} | W at f={F_BEST:.2f} {fmt(wf)}"
              + (f" | network {net:.3g}{'' if costs else ' (scaled)'}, W at f, k={K_HOLD} {fmt(wnet)}"
                 if net is not None else "")
              + f" -> {rows['grids'][n]['decision']}")
    for name, n, tau, cost in w["hybrids"]:
        c = classical_cost(env, tau)
        f = (tau - w["physics_tau"][n]) / (w["bound"][n] - w["physics_tau"][n])
        rows["hybrids"][name] = {"n": n, "tau": tau, "cost": cost, "classical_cost": c, "w": c / cost, "f": f}
        print(f"   trained {name:32s} tau {tau:.2f} @ {cost:.3g} | closed f {f:+.2f} | classical {c:.3g}, W {c / cost:.2f}")
    return rows


def main() -> int:
    out = [report(w1("16")), report(w1("256")), report(w3("16")), report(w3("64"))]
    for w in tracers() or []:
        out.append(report(w))
    (RES / "planner_bounds.json").write_text(json.dumps(out, indent=1))
    print(f"\nwrote {RES / 'planner_bounds.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
