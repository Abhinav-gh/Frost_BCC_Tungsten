"""Frost on neural PDE surrogates (The Well, Ohana et al. NeurIPS 2024): certified multirate execution of an expensive
surrogate (the teacher) with a cheap one (the student), the same mechanism as Frost's MD runtime.

The released benchmark models are next-step predictors (context 4, ZScore-normalised, full-state output), rolled out
autoregressively exactly as ``the_well.benchmark.trainer.Trainer.rollout_model`` does at test time: normalise the
moving window, predict, denormalise, slide the window. This composes those public pieces and adds three schedules:

- ``teacher``: the expensive model every step (the baseline, their protocol);
- ``student``: the cheap model every step (the distilled-model-alone control, as MACE-NEP in MD);
- ``multirate k``: the teacher every k steps; in between the student plus the residual R = T(x) - S(x) measured at the
  last refresh and held (Frost's held residual, in state space instead of force space). A refresh costs T + S, so the
  cost per step is T / k + S, and the speedup is 1 / (c_S + 1 / k) with c_S the student's cost relative to the teacher.

The certificate's input is logged at every refresh: the residual's change since the last refresh, RMS over normalised
fields (the staleness, as in MD). Fidelity is The Well's own VRMSE against the simulation, per step and field, and
each schedule is compared with the teacher's rollout trajectory by trajectory (paired differences, 95% bootstrap
interval): the claim is "as good as running the teacher", not "as good as the simulation".

Usage (GPU; frost-mhd env)
  .venvs/frost-mhd/bin/python experiments/e15_well/multirate_surrogate.py --dataset MHD_64 --teacher UNetConvNext \\
      --student FNO --steps 40 --ks 2,4,8 --n-traj 8 --tag mhd64_cnext_fno
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from the_well.benchmark import models as well_models
from the_well.benchmark.metrics import VRMSE
from the_well.data import WellDataset
from the_well.data.data_formatter import DefaultChannelsFirstFormatter
from the_well.data.normalization import ZScoreNormalization

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data" / "datasets" / "the_well"
OUT = ROOT / "data" / "results" / "e15"
MODELS = {"FNO": well_models.FNO, "TFNO": well_models.TFNO, "UNetClassic": well_models.UNetClassic,
          "UNetConvNext": well_models.UNetConvNext}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--teacher", default="UNetConvNext", choices=sorted(MODELS))
    ap.add_argument("--student", default="FNO", choices=sorted(MODELS))
    ap.add_argument("--steps", type=int, default=40, help="rollout length (their test rollouts: up to the trajectory)")
    ap.add_argument("--ks", default="2,4,8", help="multirate intervals")
    ap.add_argument("--n-traj", type=int, default=8, help="test trajectories (spread over the test files)")
    ap.add_argument("--include", default="", help="comma list of test-file name filters (WellDataset include_filters)")
    ap.add_argument("--timing-reps", type=int, default=20)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    filters = [f for f in args.include.split(",") if f]

    common = dict(well_base_path=str(BASE), well_dataset_name=args.dataset, well_split_name="test",
                  include_filters=filters, n_steps_input=4)
    norm = WellDataset(**common, n_steps_output=1, use_normalization=True,
                       normalization_type=ZScoreNormalization).norm
    ds = WellDataset(**common, n_steps_output=args.steps, full_trajectory_mode=True, max_rollout_steps=args.steps)
    meta = ds.metadata
    fmt = DefaultChannelsFirstFormatter(meta)
    teacher = MODELS[args.teacher].from_pretrained(f"polymathic-ai/{args.teacher}-{args.dataset}").to(dev).eval()
    student = MODELS[args.student].from_pretrained(f"polymathic-ai/{args.student}-{args.dataset}").to(dev).eval()
    idx = np.unique(np.linspace(0, len(ds) - 1, min(args.n_traj, len(ds))).round().astype(int))
    print(f"{args.dataset}: {len(ds)} test trajectories, using {len(idx)}; teacher {args.teacher}, student {args.student}; "
          f"{args.steps} steps", flush=True)

    @torch.no_grad()
    def predict(model, mb):
        b = dict(mb)
        b["input_fields"] = norm.normalize_flattened(mb["input_fields"], "variable")
        if "constant_fields" in mb:
            b["constant_fields"] = norm.normalize_flattened(mb["constant_fields"], "constant")
        inputs, _ = fmt.process_input(b)
        y = model(*[x.to(dev) for x in inputs])
        y = fmt.process_output_channel_last(y)
        return norm.denormalize_flattened(y, "variable")  # (B, *spatial, C)

    def to_dev(batch):
        return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}

    def rollout(batch, k: int | None, which: str):
        """k None: one model every step (which = teacher | student); k: the multirate schedule."""
        mb = dict(batch)
        preds, stale, r, r_prev = [], [], None, None
        for i in range(args.steps):
            if k is None:
                y = predict(teacher if which == "teacher" else student, mb)
            elif i % k == 0:
                y_t, y_s = predict(teacher, mb), predict(student, mb)
                r = y_t - y_s
                if r_prev is not None:  # the certificate's input: the residual's drift over the interval
                    z = norm.normalize_flattened(r, "variable") - norm.normalize_flattened(r_prev, "variable")
                    stale.append(float(z.pow(2).mean().sqrt()))
                r_prev = r
                y = y_t
            else:
                y = predict(student, mb) + r
            y = fmt.process_output_expand_time(y)
            preds.append(y)
            mb["input_fields"] = torch.cat([mb["input_fields"][:, 1:], y], dim=1)
        return torch.cat(preds, dim=1), stale

    # costs: one call each, after warm-up, synchronised
    probe = to_dev(ds[int(idx[0])])
    probe = {k: (v[None] if torch.is_tensor(v) else v) for k, v in probe.items()}
    cost = {}
    for name, model in (("teacher", teacher), ("student", student)):
        for _ in range(3):
            predict(model, probe)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.timing_reps):
            predict(model, probe)
        torch.cuda.synchronize()
        cost[name] = 1e3 * (time.perf_counter() - t0) / args.timing_reps
    c_s = cost["student"] / cost["teacher"]
    print(f"per call: teacher {cost['teacher']:.2f} ms, student {cost['student']:.2f} ms (c_S {c_s:.3f})", flush=True)

    ks = [int(k) for k in args.ks.split(",") if k]
    schedules = [("teacher", None), ("student", None)] + [(f"multirate{k}", k) for k in ks]
    vr = {name: [] for name, _ in schedules}  # [traj] -> (steps,) VRMSE averaged over fields
    vr_field = {name: [] for name, _ in schedules}
    wall = {name: 0.0 for name, _ in schedules}
    stale_log = {name: [] for name, _ in schedules}
    for j in idx:
        batch = to_dev(ds[int(j)])
        batch = {k: (v[None] if torch.is_tensor(v) else v) for k, v in batch.items()}
        _, y_ref = fmt.process_input(batch)
        y_ref = y_ref.to(dev)[:, : args.steps]
        for name, k in schedules:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            pred, stale = rollout(batch, k, name)
            torch.cuda.synchronize()
            wall[name] += time.perf_counter() - t0
            e = torch.stack([VRMSE.eval(pred[:, t], y_ref[:, t], meta) for t in range(args.steps)])  # (steps, B, C)
            e = e.squeeze(1).float().cpu().numpy()
            vr[name].append(np.nanmean(e, axis=-1))
            vr_field[name].append(e)
            stale_log[name] += stale

    n_steps_total = len(idx) * args.steps
    rng = np.random.default_rng(0)
    teach = np.stack(vr["teacher"])  # (traj, steps)
    report = {"dataset": args.dataset, "teacher": args.teacher, "student": args.student, "steps": args.steps,
              "n_traj": len(idx), "cost_ms": cost, "c_student": c_s, "fields": list(meta.field_names) if hasattr(meta, "field_names") else None,
              "schedules": {}}
    windows = [(0, min(10, args.steps)), (0, args.steps)]
    for name, k in schedules:
        m = np.stack(vr[name])
        row = {"ms_per_step_wall": 1e3 * wall[name] / n_steps_total,
               "ms_per_step_model": (cost["teacher"] if name == "teacher" else cost["student"] if name == "student"
                                     else cost["teacher"] / k + cost["student"]),
               "vrmse_by_step": m.mean(0).tolist()}
        row["speedup_model"] = cost["teacher"] / row["ms_per_step_model"]
        row["speedup_wall"] = (1e3 * wall["teacher"] / n_steps_total) / row["ms_per_step_wall"]
        for lo, hi in windows:
            d = (m[:, lo:hi].mean(1) - teach[:, lo:hi].mean(1)) / teach[:, lo:hi].mean(1)  # relative, per trajectory
            boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
            row[f"vrmse_{lo + 1}_{hi}"] = float(m[:, lo:hi].mean())
            row[f"rel_vs_teacher_{lo + 1}_{hi}"] = {"mean": float(d.mean()),
                                                    "ci95": [float(x) for x in np.percentile(boot, [2.5, 97.5])]}
        if stale_log[name]:
            row["staleness_rms"] = {"mean": float(np.mean(stale_log[name])), "max": float(np.max(stale_log[name]))}
        report["schedules"][name] = row
        rel = row[f"rel_vs_teacher_1_{args.steps}"]
        print(f"{name:12s} {row['ms_per_step_model']:7.2f} ms/step model ({row['speedup_model']:.2f}x), wall "
              f"{row['ms_per_step_wall']:.2f} ({row['speedup_wall']:.2f}x) | VRMSE 1-10 {row['vrmse_1_' + str(windows[0][1])]:.4f}, "
              f"1-{args.steps} {row['vrmse_1_' + str(args.steps)]:.4f} | vs teacher {100 * rel['mean']:+.1f}% "
              f"[{100 * rel['ci95'][0]:+.1f}, {100 * rel['ci95'][1]:+.1f}]"
              + (f" | staleness {row['staleness_rms']['mean']:.3f}" if "staleness_rms" in row else ""), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"well_{args.tag}.json").write_text(json.dumps(report, indent=1))
    print(f"-> {OUT / f'well_{args.tag}.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
