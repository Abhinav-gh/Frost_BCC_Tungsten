"""E2: cost decomposition. How much of the model can a cache actually remove?

Research question (RQ2). What fraction of MLIP inference cost is edge-local,
and therefore cacheable, per architecture?

This sets the ceiling. Frost removes tensor-product work on clean edges but
always re-runs aggregation, so even at a dirty fraction of zero the speedup is
bounded by

    S_max  =  1 / ((1 - theta) + c_axpy)

where ``theta`` is the share of forward+backward cost that caching removes and
``c_axpy`` is the cost of evaluating the affine cache (Frost.pdf 13.3).

Success / falsification (Frost.pdf 12.7.2)
    theta < 0.35 for MACE caps the speedup below 1.5x even at phi = 0.

Method: measure what removal saves, not what a profiler attributes.
Per-module timing hooks do not work here -- several MACE submodules are
TorchScript modules, which reject pre-hooks -- and profiler attribution of
kernels to modules is guesswork anyway. So each edge tensor product is swapped
for a stand-in that returns a precomputed tensor of the same shape, and the
whole forward+backward is timed with and without the swap. The difference is
exactly the cost a perfect cache would remove, measured end to end, including
the backward pass that MLIP inference cannot skip.

The model is driven directly rather than through ASE, whose own overhead
exceeds the model on these cells (plan section 1.3) and would dilute theta.

Two ceilings are reported:
  theta_L1   only layer 1 cached -- the guaranteed floor, zero closure
  theta_all  every edge tensor product cached -- needs attenuated closure

Usage
  python experiments/e2_cost/measure_cost.py --system si512
  python experiments/e2_cost/measure_cost.py --all
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "data" / "results" / "e2"
THETA_FLOOR = 0.35
C_AXPY = 0.02  # affine-cache evaluation, as a fraction of step cost (13.3)


class CachedStandIn(torch.nn.Module):
    """Returns a fixed tensor in place of a tensor product.

    The output keeps ``requires_grad`` so downstream autograd still builds the
    same backward graph it would with a live cache; only the tensor product's
    own forward and backward disappear.
    """

    def __init__(self, value: torch.Tensor):
        super().__init__()
        self.register_buffer("value", value.detach().clone())

    def forward(self, *args, **kwargs):
        return self.value.clone().requires_grad_(True)


def _capture(module) -> dict:
    box: dict = {}

    def hook(mod, inputs, output):
        box["out"] = output

    h = module.register_forward_hook(hook)
    return {"box": box, "handle": h}


def time_step(model, batch, repeats: int) -> float:
    """Median forward+backward (forces) in milliseconds."""
    data = batch.to_dict()
    for _ in range(3):  # warm-up: autotuning and allocator growth
        model(data, compute_force=True, training=False)
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        model(data, compute_force=True, training=False)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return float(np.median(times))


def measure(system: str, model_name: str, device: str, repeats: int) -> dict:
    from mace.calculators import mace_mp

    from frost.calibrate import _batch_for
    from frost.systems import E1_SYSTEMS

    atoms = E1_SYSTEMS[system]()
    calc = mace_mp(model=model_name, device=device, default_dtype="float32")
    model = calc.models[0]
    batch = _batch_for(calc, atoms, 6.0, device)[0]
    n_edges = int(batch.edge_index.shape[1])

    interactions = list(model.interactions)
    # capture each layer's tensor-product output so the stand-in is shape-exact
    caps = [_capture(block.conv_tp) for block in interactions]
    model(batch.to_dict(), compute_force=True, training=False)
    outputs = [c["box"]["out"] for c in caps]
    for c in caps:
        c["handle"].remove()

    originals = [block.conv_tp for block in interactions]

    def swap(layers):
        for k, block in enumerate(interactions):
            block.conv_tp = CachedStandIn(outputs[k]) if k in layers else originals[k]

    full = time_step(model, batch, repeats)
    swap({0})
    no_l1 = time_step(model, batch, repeats)
    swap(set(range(len(interactions))))
    no_all = time_step(model, batch, repeats)
    swap(set())  # restore

    theta_l1 = max(0.0, 1.0 - no_l1 / full)
    theta_all = max(0.0, 1.0 - no_all / full)
    return {
        "system": system,
        "model": model_name,
        "n_atoms": len(atoms),
        "n_edges": n_edges,
        "mean_degree": n_edges / len(atoms),
        "n_layers": len(interactions),
        "step_ms_full": full,
        "step_ms_layer1_cached": no_l1,
        "step_ms_all_cached": no_all,
        "theta_L1": theta_l1,
        "theta_all": theta_all,
        "S_max_L1": 1.0 / ((1.0 - theta_l1) + C_AXPY),
        "S_max_all": 1.0 / ((1.0 - theta_all) + C_AXPY),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--system", default="si512")
    p.add_argument("--all", action="store_true")
    p.add_argument("--model", default="medium")
    p.add_argument("--device", default="cuda")
    p.add_argument("--repeats", type=int, default=20)
    args = p.parse_args()

    from frost.systems import E1_SYSTEMS

    systems = list(E1_SYSTEMS) if args.all else [args.system]
    RESULTS.mkdir(parents=True, exist_ok=True)

    rows = []
    for s in systems:
        print(f"\n=== {s} ===", flush=True)
        try:
            row = measure(s, args.model, args.device, args.repeats)
        except Exception as exc:
            print(f"  [FAIL] {type(exc).__name__}: {exc}")
            continue
        rows.append(row)
        print(
            f"  {row['n_atoms']} atoms, {row['n_edges']} edges, "
            f"{row['n_layers']} interaction layers"
        )
        print(f"  step, exact                  {row['step_ms_full']:8.2f} ms")
        print(
            f"  step, layer-1 TP cached      {row['step_ms_layer1_cached']:8.2f} ms"
            f"   theta_L1  = {row['theta_L1']:.3f}   S_max {row['S_max_L1']:.2f}x"
        )
        print(
            f"  step, every TP cached        {row['step_ms_all_cached']:8.2f} ms"
            f"   theta_all = {row['theta_all']:.3f}   S_max {row['S_max_all']:.2f}x"
        )

    out = RESULTS / "cost_decomposition.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\n{len(rows)} rows -> {out}")
    if not rows:
        return 2
    t1 = float(np.median([r["theta_L1"] for r in rows]))
    ta = float(np.median([r["theta_all"] for r in rows]))
    print("\n" + "=" * 60)
    print(
        f"E2 median over {len(rows)} system(s): theta_L1 = {t1:.3f}, theta_all = {ta:.3f}"
    )
    print(f"   falsification floor: theta < {THETA_FLOOR}")
    print(
        f"   all layers cached : {'PASS' if ta >= THETA_FLOOR else 'FAIL'}"
        "  (needs attenuated closure, Gate 3)"
    )
    print(
        f"   layer 1 only      : {'PASS' if t1 >= THETA_FLOOR else 'FAIL'}"
        "  (the zero-closure floor)"
    )
    verdict = (
        "PASSES"
        if t1 >= THETA_FLOOR
        else "PASSES CONDITIONALLY"
        if ta >= THETA_FLOOR
        else "FAILS"
    )
    print(f"   >>> E2 {verdict} <<<")
    print("=" * 60)
    return 0 if ta >= THETA_FLOOR else 1


if __name__ == "__main__":
    raise SystemExit(main())
