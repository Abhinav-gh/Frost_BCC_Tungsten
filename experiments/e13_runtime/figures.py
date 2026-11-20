"""The paper's ablation figures, drawn from result files only (no runs).

1. ``qos.pdf``: wrong decisions on the 36 recorded 50 ps runs, Frost's certificate against Green and Rumba
   (``data/results/e9/qos_headtohead.json``, energy truth).
2. ``k_sweep_cu.pdf``: Cu with the descriptor student beyond the resonance prior: 50 ps NVE energy gain and
   temperature shift against the outer interval, with the prior's k and the 1 meV/atom line.
3. ``kernels.pdf``: the two kernel fixes on both arms: the teacher's force call (base, edge-force) per system,
   and the student's call (eager graph, compiled, compiled with detached parameters).
4. ``competitors.pdf``: speedup and fidelity of the reruns that have landed (MACE-NEP alone, NEP as Frost's
   student, Frost), Cu, paired 50 ps.

Output: docs/paper/figs/*.pdf
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "data" / "results"
OUT = ROOT / "docs" / "paper" / "figs"


def load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def qos() -> None:
    d = load(R / "e9" / "qos_headtohead.json")
    if not d:
        return
    truth = d.get("energy") or d[next(iter(d))]
    mons = truth["monitors"]
    rows = [(name, m["errors"]) for name, m in mons.items() if "errors" in m]
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.barh([r[0].replace("_", " ") for r in rows], [r[1] for r in rows], color="#4a7ab5")
    ax.set_xlabel(f"wrong decisions on {truth['n_runs']} runs (truth: {truth['truth']})")
    fig.tight_layout()
    fig.savefig(OUT / "qos.pdf")
    plt.close(fig)


def k_sweep() -> None:
    rows = []
    for k in (24, 32, 40, 48):
        name = "fidelity_cu500_T300_nve_f45_k24_50ps.json" if k == 24 else f"fidelity_cu500_T300_nve_f45_k{k}_50ps.json"
        d = load(R / "e6" / name)
        if d and d.get("frost"):
            tr = d["frost"]["energy_trace_meV_per_atom_1ps"]
            rows.append((k, tr[-1], d["comparison"]["temperature_shift_K"]))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(4.5, 3))
    ax.plot([r[0] for r in rows], [r[1] for r in rows], "o-", label="energy gain, 50 ps")
    ax.axhline(1.0, color="gray", ls="--", lw=1)
    ax.axvline(24, color="#c44", ls=":", lw=1, label="resonance prior (0.22 T)")
    ax2 = ax.twinx()
    ax2.plot([r[0] for r in rows], [r[2] for r in rows], "s--", color="#888", label="T shift")
    ax.set_xlabel("outer interval k (fs)")
    ax.set_ylabel("meV/atom")
    ax2.set_ylabel("K")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "k_sweep_cu.pdf")
    plt.close(fig)


def kernels() -> None:
    d = load(R / "e13" / "teacher_profile_edge.json")
    if not d:
        return
    systems = list(d)
    base = [d[s]["variants"]["base"]["ms"] for s in systems]
    edge = [d[s]["variants"]["edge"]["ms"] for s in systems]
    fig, ax = plt.subplots(figsize=(5, 3))
    x = range(len(systems))
    ax.bar([i - 0.2 for i in x], base, 0.4, label="teacher, autograd to positions")
    ax.bar([i + 0.2 for i in x], edge, 0.4, label="teacher, edge-force output")
    ax.set_xticks(list(x), [s.split("_")[0] for s in systems])
    ax.set_ylabel("ms per force call")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "kernels.pdf")
    plt.close(fig)


def competitors() -> None:
    nep = load(R / "e14" / "mace_nep_cu500_T300_paired50.json")
    st = load(R / "e6" / "fidelity_cu500_T300_nve_nepstudent_k24_50ps.json")
    fr = load(R / "e6" / "fidelity_cu500_T300_nve_v2_f45_k24.json")
    if not (nep and st and fr):
        return
    own = nep["vdos_teacher_own_spread"]
    # absolute cost per simulated fs: the speedups of these runs are against different teacher builds
    rows = [("NEP alone (GPUMD, with output)", nep["vdos_l1_vs_teacher"], nep["gpumd_ms_per_step"]),
            ("Frost, NEP as student (TorchNEP)", st["comparison"]["vdos_l1_relative"],
             1e3 * st["wall_seconds"]["frost"] / st["fs"]),
            ("Frost, descriptor student", fr["comparison"]["vdos_l1_relative"],
             1e3 * fr["wall_seconds"]["frost"] / fr["fs"])]
    fig, ax = plt.subplots(figsize=(5, 3))
    for name, v, s in rows:
        ax.scatter(s, v, s=40)
        ax.annotate(name, (s, v), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.axhline(own, color="gray", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("ms per simulated fs (one A6000, 500 Cu atoms)")
    ax.set_ylabel("VDOS L1 against the teacher (dashed: its own spread)")
    fig.tight_layout()
    fig.savefig(OUT / "competitors_cu.pdf")
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for fn in (qos, k_sweep, kernels, competitors):
        fn()
    print(sorted(p.name for p in OUT.glob("*.pdf")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
