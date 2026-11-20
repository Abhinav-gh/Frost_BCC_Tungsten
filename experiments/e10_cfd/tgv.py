"""The Taylor-Green vortex at Re 1600: the standard decaying-turbulence LES benchmark (DeBonis, AIAA 2013-0382; the
spectral DNS of van Rees et al., J. Comput. Phys. 2011), here as a CFD family for Frost.

- Flow: periodic [0, 2 pi)^3, u = (sin x cos y cos z, -cos x sin y cos z, 0), U0 = L = 1, nu = 1 / 1600, no forcing. A
  single-mode start, exact on every grid: the smooth vortices break down into turbulence, dissipation peaks near t = 9.
- DNS: ``frost.cfd.spectral3d`` at ``--n-dns`` (2/3 rule), kinetic energy E(t) and dissipation nu <|omega|^2> recorded
  every ``--record-every`` coarse steps of 0.04.
- Methods, each from the same start on its own grid: classical LES (``n:m:closure:c``, as ``stats3d.py``) and learned
  closures (``model:k``, ``split3d.rollout``: Frost's hold k), timed per 0.04 time units at batch 1 (one TGV).
- Score (the benchmark's own): the total dissipation rate -dE/dt of the resolved kinetic energy against the DNS's, L2 over
  t in [0, T] relative to the DNS; the peak's value and time; and E(t).

Output: data/results/e10/tgv_<tag>.json

Usage (guarded launcher; GPU, JAX env)
  scripts/safe_run.sh tgv env -u LD_LIBRARY_PATH -u CUDA_HOME PYTHONPATH=src .venvs/frost-mhd/bin/python \\
      experiments/e10_cfd/tgv.py --n-dns 384 --sub 8 --tag re1600
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from frost.cfd import spectral3d as s3
from frost.cfd import split3d
from frost.cfd.spectral3d import Grid3

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from w3split import load_model  # noqa: E402

RES = ROOT / "data" / "results" / "e10"
MODELS = ROOT / "data" / "models" / "cfd"
DT_C = 0.04


def taylor_green(grid: Grid3):
    x = 2.0 * np.pi * np.arange(grid.n) / grid.n
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    u = np.stack([np.sin(X) * np.cos(Y) * np.cos(Z), -np.cos(X) * np.sin(Y) * np.cos(Z), np.zeros_like(X)])
    return s3.to_hat(grid, jnp.asarray(u[None], dtype=jnp.float32))


def score(t, e, t_dns, e_dns, eps_dns_direct):
    """-dE/dt of a method's resolved energy against the DNS's dissipation (both on the method's record times)."""
    rate = -np.gradient(e, t)
    ref = np.interp(t, t_dns, eps_dns_direct)
    i_pk, i_ref = int(np.argmax(rate)), int(np.argmax(ref))
    return {"eps_l2_rel": float(np.sqrt(np.mean((rate - ref) ** 2)) / np.sqrt(np.mean(ref**2))),
            "peak_value": float(rate[i_pk]), "peak_value_rel_err": float(rate[i_pk] / ref[i_ref] - 1.0),
            "peak_time": float(t[i_pk]), "peak_time_err": float(t[i_pk] - t[i_ref]),
            "energy_l2_rel": float(np.sqrt(np.mean((e - np.interp(t, t_dns, e_dns)) ** 2))
                                   / np.sqrt(np.mean(np.interp(t, t_dns, e_dns) ** 2)))}


def run_dns(args, nu: float, n_rec: int, t_rec, out: dict):
    """Our own spectral DNS of the single-mode start (2/3 rule), energy and nu <|omega|^2> on the record times."""
    fine = Grid3(args.n_dns, nu=nu, power=0.0)
    dt = DT_C / args.sub

    @partial(jax.jit, static_argnames=("n",))
    def dns_advance(u, n: int):
        return jax.lax.fori_loop(0, n, lambda _, x: s3.step(fine, x, dt), u)

    u = taylor_green(fine)
    e_dns, eps_dns = [float(s3.energy_of(fine, u)[0])], [float(s3.dissipation(fine, u)[0])]
    t0 = time.perf_counter()
    for _ in range(n_rec):
        u = dns_advance(u, args.sub * args.record_every)
        e_dns.append(float(s3.energy_of(fine, u)[0]))
        eps_dns.append(float(s3.dissipation(fine, u)[0]))
    dns_s = time.perf_counter() - t0
    e_dns, eps_dns = np.asarray(e_dns), np.asarray(eps_dns)
    rate_dns = -np.gradient(e_dns, t_rec)
    i = int(np.argmax(eps_dns))
    out["dns"] = {"E": e_dns.tolist(), "eps": eps_dns.tolist(), "t": t_rec.tolist(), "seconds": dns_s,
                  "eps_peak": float(eps_dns[i]), "eps_peak_time": float(t_rec[i]),
                  "rate_vs_eps_l2_rel": float(np.sqrt(np.mean((rate_dns - eps_dns) ** 2)) / np.sqrt(np.mean(eps_dns**2)))}
    print(f"DNS {args.n_dns}^3: E0 {e_dns[0]:.4f} | dissipation peak {eps_dns[i]:.5f} at t {t_rec[i]:.2f} "
          f"(published ~0.0127 near t 9) | -dE/dt against nu<w^2> {out['dns']['rate_vs_eps_l2_rel']:.3f} (resolution) | "
          f"{dns_s:.0f} s", flush=True)
    del u
    return e_dns, eps_dns


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-dns", type=int, default=384)
    ap.add_argument("--sub", type=int, default=8, help="DNS steps per 0.04")
    ap.add_argument("--T", type=float, default=20.0)
    ap.add_argument("--record-every", type=int, default=5, help="coarse steps of 0.04 between records")
    ap.add_argument("--classical", default="32:1:none:0,32:1:smag:0.17,48:1:none:0,48:1:smag:0.17,48:1:cl:1,"
                                           "64:2:none:0,64:2:smag:0.17,64:2:cl:1,96:3:smag:0.17")
    ap.add_argument("--learned", default="hit384_32k8sk:8,hit384_32k8sk:1")
    ap.add_argument("--timed-steps", type=int, default=200)
    ap.add_argument("--re", type=float, default=1600.0, help="Reynolds number, nu = 1 / Re")
    ap.add_argument("--reference", type=Path, default=None,
                    help="a published DNS time series instead of our DNS (Dairay et al. 2017, Zenodo 2577239: columns "
                         "t, E, -dE/dt, epsilon from the velocity gradients, ...); our DNS is skipped")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    nu = 1.0 / args.re
    n_rec = round(args.T / (DT_C * args.record_every))
    t_rec = DT_C * args.record_every * np.arange(n_rec + 1)
    out = {"nu": nu, "re": args.re, "T": args.T, "n_dns": args.n_dns, "rows": {},
           "reference": str(args.reference) if args.reference else None}

    if args.reference is not None:
        # their DNS (512^3 and up, Incompact3d), interpolated to our record times
        tab = np.loadtxt(args.reference, comments="#")
        e_dns, eps_dns = np.interp(t_rec, tab[:, 0], tab[:, 1]), np.interp(t_rec, tab[:, 0], tab[:, 3])
        i = int(np.argmax(eps_dns))
        out["dns"] = {"E": e_dns.tolist(), "eps": eps_dns.tolist(), "t": t_rec.tolist(), "published": True,
                      "eps_peak": float(eps_dns[i]), "eps_peak_time": float(t_rec[i])}
        print(f"published DNS at Re {args.re:g} ({args.reference.name}): E0 {e_dns[0]:.4f} | dissipation peak "
              f"{eps_dns[i]:.5f} at t {t_rec[i]:.2f}", flush=True)
    else:
        e_dns, eps_dns = run_dns(args, nu, n_rec, t_rec, out)

    def report(name, e, ms, extra):
        e = np.asarray(e)
        finite = bool(np.isfinite(e).all())
        row = {"ms_per_0.04": ms, "finite": finite, "E": e.tolist(), **extra}
        if finite:
            row.update(score(t_rec, e, t_rec, e_dns, eps_dns))
        out["rows"][name] = row
        RES.mkdir(parents=True, exist_ok=True)
        (RES / f"tgv_{args.tag}.json").write_text(json.dumps(out, indent=1))
        print(f"{name:28s} {ms:8.2f} ms | " + (f"dissipation L2 {row['eps_l2_rel']:.3f} | peak {row['peak_value']:.5f} "
              f"({100 * row['peak_value_rel_err']:+.1f}%) at t {row['peak_time']:.2f} ({row['peak_time_err']:+.2f}) | "
              f"E L2 {row['energy_l2_rel']:.3f}" if finite else "non-finite"), flush=True)

    def best_ms(fn, n_steps):
        fn().block_until_ready()  # compile
        best = 1e30
        for _ in range(3):
            t0 = time.perf_counter()
            fn().block_until_ready()
            best = min(best, time.perf_counter() - t0)
        return 1e3 * best / n_steps

    for run in (r for r in args.classical.split(",") if r):
        n, m, kind, c = run.split(":")
        n, m, c = int(n), int(m), float(c)
        g = Grid3(n, nu=nu, power=0.0, dealias="exp", filter_strength=6.0 / m)
        u0 = taylor_green(g)

        @partial(jax.jit, static_argnames=("steps",))
        def adv(u, steps: int, g=g, m=m, kind=kind, c=c):
            return jax.lax.fori_loop(0, steps, lambda _, x: split3d.physics_step(g, x, DT_C, (kind, c), "hydro", m), u)

        e, uu = [float(s3.energy_of(g, u0)[0])], u0
        for _ in range(n_rec):
            uu = adv(uu, args.record_every)
            e.append(float(s3.energy_of(g, uu)[0]))
        ms = best_ms(lambda: adv(u0, args.timed_steps), args.timed_steps)
        report(f"classical {run}", e, ms, {"n": n, "m": m, "closure": kind, "c": c})

    for run in (r for r in args.learned.split(",") if r):
        name, k = run.split(":")[:2]
        k = int(k)
        if not (MODELS / f"{name}.json").exists():
            print(f"learned {run}: no model, skipped", flush=True)
            continue
        model = load_model(MODELS / name)
        meta = json.loads((MODELS / f"{name}.json").read_text())
        g = Grid3(int(meta["grid"]), nu=nu, power=0.0, dealias=meta.get("dealias", "exp"))
        u0 = taylor_green(g)
        states = split3d.rollout(g, model, u0, k=k, dt=DT_C, n_records=n_rec, record_every=args.record_every)
        e = [float(s3.energy_of(g, u0)[0])] + [float(x) for x in np.asarray(jax.vmap(lambda s, g=g: s3.energy_of(g, s))(states)).ravel()]
        blocks = max(2, round(args.timed_steps / k))
        ms = best_ms(lambda: split3d.rollout(g, model, u0, k=k, dt=DT_C, n_records=1, record_every=blocks * k)[-1],
                     blocks * k)
        report(f"learned {run}", e[: n_rec + 1], ms, {"model": name, "k": k, "n": g.n})
    print(f"-> {RES / f'tgv_{args.tag}.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
