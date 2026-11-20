"""Reuse typing on a real MLIP: which tensors a runtime may reuse, and what reuse would cost (R1).

new_directions.md R1 classifies every tensor a learned component produces by how the
simulator consumes it:
- **structural**: independent of the state, so reusable freely;
- **gradient-consumed**: on the path from positions to the energy whose gradient is
  the force, so reusing it deletes the adjoint-projected derivative D^T lambda
  (Theorem A);
- **value-consumed**: added to the state directly (CFD corrections), so reusable
  with error bounded by the input change.

For an MLIP the second class is the dangerous one. This walks every submodule output
of MACE-MP-0 medium on a rattled Si512 cell:
- forward hooks (public API) capture outputs and change nothing;
- the model's parameters are frozen with requires_grad_(False), so a tensor
  requires grad if and only if it depends on the positions;
- for each position-dependent tensor h, the force carried through it is measured
  with public autograd: lambda = dE/dh, then J^T lambda = d(h . lambda)/dx. This is
  exactly the force a reuse of h would delete.

The result is a reuse map of the whole model. v1 measured three entries by
substituting surrogates: 67.7% at layer 1 zeroth order, 36% for layer-2 neighbour
features, 15.4% for Allegro's environment. This measures every entry by the
identity of Theorem A, with no surrogate.

Output: data/results/e9/reuse_typing_mace.json

Usage (guarded launcher; GPU)
  scripts/safe_run.sh e9_reuse_typing .venvs/frost-mace/bin/python \\
      experiments/e9_direction/reuse_typing.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "results" / "e9" / "reuse_typing_mace.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cells", type=int, default=4)
    ap.add_argument("--rattle", type=float, default=0.05)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    from mace.calculators import mace_mp

    from frost.calibrate import _batch_for
    from frost.systems import si_scaling

    atoms = si_scaling(args.cells)
    atoms.positions += np.random.default_rng(0).normal(0.0, args.rattle, atoms.positions.shape)
    calc = mace_mp(model="medium", device="cuda", default_dtype="float32", enable_cueq=True)
    model = calc.models[0]
    for p in model.parameters():  # frozen: now "requires grad" means "depends on positions"
        p.requires_grad_(False)
    data = _batch_for(calc, atoms, float(model.r_max), "cuda")[0].to_dict()
    x = data["positions"].detach().clone().requires_grad_(True)
    data["positions"] = x

    captured: list[tuple[str, int, torch.Tensor]] = []

    def hook_for(name):
        def hook(_module, _args, output):
            outs = output if isinstance(output, (tuple, list)) else (output,)
            for i, t in enumerate(outs):
                if torch.is_tensor(t) and t.is_floating_point():
                    captured.append((name, i, t))
        return hook

    handles = [m.register_forward_hook(hook_for(n)) for n, m in model.named_modules() if n]
    out = model(data, compute_force=False, training=True)
    for h in handles:
        h.remove()
    energy = out["energy"].sum()
    (grad,) = torch.autograd.grad(energy, x, retain_graph=True)
    force = -grad
    force_rms = float(force.norm(dim=1).pow(2).mean().sqrt())

    rows = []
    for name, idx, t in captured:
        row = {"module": name, "output": idx, "shape": list(t.shape)}
        if not t.requires_grad:
            row["class"] = "structural"
            row["deleted_force_fraction"] = 0.0
        else:
            (lam,) = torch.autograd.grad(energy, t, retain_graph=True, allow_unused=True)
            if lam is None:
                row["class"] = "unused"
                row["deleted_force_fraction"] = 0.0
            else:
                (jt_lam,) = torch.autograd.grad((t * lam.detach()).sum(), x, retain_graph=True,
                                                allow_unused=True)
                frac = 0.0 if jt_lam is None else float(
                    jt_lam.norm(dim=1).pow(2).mean().sqrt()) / force_rms
                row["class"] = "gradient-consumed"
                row["deleted_force_fraction"] = frac
        rows.append(row)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    report = {"model": "MACE-MP-0 medium (cuEq)", "n_atoms": len(atoms),
              "force_rms_eV_per_A": force_rms, "tensors": rows}
    OUT.write_text(json.dumps(report, indent=1))
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["class"]] = counts.get(r["class"], 0) + 1
    print(f"{len(rows)} tensors: {counts} | force RMS {force_rms:.3f} eV/A")
    for r in rows:
        if r["class"] != "gradient-consumed" or r["deleted_force_fraction"] > 0.05 or \
                ("linear_up" in r["module"] or "node_embedding" in r["module"]):
            print(f"  {r['class']:18s} {r['deleted_force_fraction']:7.3f}  {r['module']}[{r['output']}]"
                  f" {r['shape']}")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
