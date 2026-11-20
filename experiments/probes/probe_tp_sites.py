"""Locate and instrument the edge tensor-product call sites in MACE.

This is the week-0 feasibility probe behind Gate 0 and the whole E1 protocol.
It answers three questions, all of which the plan depends on:

  1. Where does R(r) (x) Y(rhat) actually live in the forward pass, and can a
     plain forward hook see both its inputs and its output?
  2. Is the tensor product separable from the scatter-add, so Frost can cache
     between them?  (Both cuEquivariance and OpenEquivariance expose an unfused
     TensorProduct, but MACE's own module graph has to cooperate too.)
  3. Is dg/dr_ij reachable -- the Jacobian that Theorem 6.1 makes mandatory --
     and does it have the rank-3 structure Appendix D predicts?

Run:  source scripts/env.sh mace && python e1/probe_tp_sites.py
"""

from __future__ import annotations

import numpy as np
import torch
from e3nn import o3

from frost.systems import si512


def describe_modules(model) -> list[tuple[str, str]]:
    hits = []
    for name, mod in model.named_modules():
        cls = type(mod).__name__
        if isinstance(mod, o3.TensorProduct) or "TensorProduct" in cls:
            hits.append((name, cls))
    return hits


def main() -> int:
    from mace.calculators import mace_mp

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    calc = mace_mp(model="medium", device=dev, default_dtype="float64")
    model = calc.models[0]

    print("=== 1. tensor-product call sites ===")
    sites = describe_modules(model)
    for name, cls in sites:
        print(f"  {name:<48} {cls}")
    if not sites:
        print("  NONE FOUND -- the attribute names moved; grep mace/modules/blocks.py")
        return 1

    print("\n=== 2. hook capture (inputs + output) ===")
    captured: dict[str, dict] = {}

    def make_hook(name):
        def hook(mod, inputs, output):
            captured[name] = {
                "n_inputs": len(inputs),
                "in_shapes": [tuple(t.shape) for t in inputs if torch.is_tensor(t)],
                "out_shape": tuple(output.shape),
                "out_requires_grad": output.requires_grad,
            }

        return hook

    handles = [
        dict(model.named_modules())[n].register_forward_hook(make_hook(n))
        for n, _ in sites
    ]

    atoms = si512()
    atoms.calc = calc
    _ = atoms.get_potential_energy()
    forces = atoms.get_forces()
    for h in handles:
        h.remove()

    n_edges = None
    for name, rec in captured.items():
        print(
            f"  {name:<40} in={rec['in_shapes']} out={rec['out_shape']} "
            f"grad={rec['out_requires_grad']}"
        )
        if n_edges is None:
            n_edges = rec["out_shape"][0]

    print(
        f"\n  atoms={len(atoms)}  edges={n_edges}  "
        f"mean degree={n_edges / len(atoms):.1f}"
    )
    print(f"  |F|_max = {np.abs(forces).max():.4f} eV/A")

    print("\n=== 3. Jacobian dg/dr_ij: rank and cost ===")
    # Layer-1 edge feature map in isolation: g(r) = R(|r|) (x) Y(rhat).
    # Rank must be <= 3 because g maps R^3 -> R^n_irrep (Appendix D).
    irreps_sh = o3.Irreps.spherical_harmonics(3)
    sh = o3.SphericalHarmonics(irreps_sh, normalize=True, normalization="component")

    def g_of_r(r: torch.Tensor) -> torch.Tensor:
        d = torch.linalg.norm(r, dim=-1, keepdim=True)
        radial = torch.exp(
            -((d - torch.linspace(0.5, 6.0, 8, device=r.device, dtype=r.dtype)) ** 2)
        )
        y = sh(r)
        return (radial[:, :, None] * y[:, None, :]).reshape(r.shape[0], -1)

    r = torch.randn(64, 3, device=dev, dtype=torch.float64, requires_grad=True)
    J = torch.autograd.functional.jacobian(lambda x: g_of_r(x).sum(0), r)
    # J has shape (n_feat, n_edges, 3); per-edge slice is (n_feat, 3)
    ranks = [int(torch.linalg.matrix_rank(J[:, e, :]).item()) for e in range(8)]
    print(
        f"  per-edge Jacobian shape {tuple(J[:, 0, :].shape)}, "
        f"ranks over 8 edges: {ranks}"
    )
    print(f"  rank <= 3 as Appendix D predicts: {max(ranks) <= 3}")

    n_feat = J.shape[0]
    val_mb = n_edges * n_feat * 4 / 2**20
    jac_mb = n_edges * n_feat * 3 * 2 / 2**20  # bf16
    print(
        f"  toy cache footprint for this system: values {val_mb:.1f} MB (fp32) "
        f"+ Jacobian {jac_mb:.1f} MB (bf16) = {1 + 3 * 0.5:.1f}x"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
