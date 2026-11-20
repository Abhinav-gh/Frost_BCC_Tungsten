"""How much of a long-range teacher's call is the long-range part (further_expansions_beyond.md 6.8, tier 2 item 8).

A third level in Frost (the reciprocal-space part held k x m steps while the short-range teacher refreshes every k) can
save at most the long-range part's own cost. In MACELES (Cheng's latent Ewald summation inside MACE), the latent charges
come from the same MACE features as the energy, so the short-range network runs whenever the charges are needed; what a
third level could skip is the Ewald sum on those charges (forward and backward). This times, on one GPU in float32:
- the whole teacher call: MACELES-OFF small (the released model from github.com/ChengUCB/les_fit), energy and forces;
- the Ewald module alone (``model.les``) on the very inputs the call gave it, forward and backward with the charges and
  positions as leaves (captured with a forward hook: no behaviour of the model is changed);
on water boxes and the solvated peptide, at sizes where an Ewald sum's cost grows faster than the network's.

Run (GPU, frost-mace env plus the les overlay; ``les`` is not installed in the env itself):
    PYTHONPATH=src:.tools/les-overlay python experiments/e22_longrange/les_profile.py
Output: data/results/e22/les_profile.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from mace import data as mdata
from mace.tools import torch_geometric, utils

from frost.systems import ala2_water, water216

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "third_party" / "les_fit_models" / "MACELES-OFF_small_converted.model"
OUT = ROOT / "data" / "results" / "e22"


ALLOWED_TOP = {"torch", "e3nn", "mace", "les"}


def load_restricted(path, dev):
    """Load the released checkpoint with torch's restricted unpickler (weights_only): only the classes and functions the
    file names are allowed, and only if every one comes from torch, e3nn, mace or les (checked by reading the pickle's
    globals without executing it; the e3nn compiled-module source inside was read statically: tensor algebra only)."""
    import importlib

    names = torch.serialization.get_unsafe_globals_in_checkpoint(str(path))
    outside = [n for n in names if n.split(".")[0] not in ALLOWED_TOP]
    if outside:
        raise SystemExit(f"refusing {path}: globals outside {sorted(ALLOWED_TOP)}: {outside}")
    allowed = []
    for name in names:
        mod, attr = name.rsplit(".", 1)
        allowed.append(getattr(importlib.import_module(mod), attr))
    with torch.serialization.safe_globals(allowed):
        return torch.load(path, map_location=dev, weights_only=True)


def fresh_models(dev):
    """The released weights in two models built here, never the checkpoint's own compiled modules (their e3nn code was
    scripted on the authors' machine and keeps CPU constants: the first profile run failed on a device mismatch inside
    it). Both through MACE's public tools: the configuration from ``extract_config_mace_model`` and the class's own
    constructor (built on the CPU, then moved) with a state-dict copy (plain e3nn kernels); and
    ``mace.cli.convert_e3nn_cueq.run`` (cuEquivariance kernels, the strongest stack, as the teachers run). The checkpoint
    trained the long-range module with its defaults, which is what the class builds when no arguments are given."""
    from mace.cli.convert_e3nn_cueq import run as to_cueq
    from mace.tools.scripts_utils import extract_config_mace_model

    src = load_restricted(MODEL, "cpu").float().eval()
    cfg = extract_config_mace_model(src)
    # built on the CPU and then moved, as MACE's own converters do (a CUDA default device breaks e3nn's CPU-generator
    # normalisation constants at construction, the second profile run's failure)
    e3nn_model = type(src)(**cfg)
    res = e3nn_model.load_state_dict(src.state_dict(), strict=False)
    # this MACE version adds zero-pattern buffers to its symmetric contractions (built by the constructor from the
    # coupling coefficients); every other key must match exactly
    extra = [k for k in res.missing_keys if not k.endswith("_zeroed")]
    if extra or res.unexpected_keys:
        raise SystemExit(f"state dict mismatch: missing {extra}, unexpected {res.unexpected_keys}")
    models = {"e3nn": e3nn_model.to(dev).float().eval()}
    models["cueq"] = to_cueq(src, device=dev, return_model=True).float().eval()
    for m in models.values():
        for p in m.parameters():
            p.requires_grad_(False)
    return models


def batch_for(atoms, model, dev):
    z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
    cfg = mdata.config_from_atoms(atoms)
    ad = mdata.AtomicData.from_config(cfg, z_table=z_table, cutoff=float(model.r_max), heads=["Default"])
    batch = torch_geometric.Batch.from_data_list([ad]).to(dev)
    out = batch.to_dict()
    out["positions"] = out["positions"].float()
    for k in ("cell", "shifts", "unit_shifts", "node_attrs", "weight", "energy_weight", "forces_weight"):
        if k in out and torch.is_floating_point(out[k]):
            out[k] = out[k].float()
    return out


def cuda_time(fn, reps: int) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / reps


def profile_one(stack, model, label, atoms, captured, dev):
    """One stack on one system: the whole call (energy and forces) against the Ewald module alone on the inputs the
    call gave it (forward and backward with charges and positions as leaves)."""
    data = batch_for(atoms, model, dev)

    def full():
        d = dict(data)
        d["positions"] = data["positions"].clone().requires_grad_(True)
        return model(d, compute_force=True)["forces"]

    f = full()
    t_full = cuda_time(full, 10)
    kw = captured["kwargs"]

    def ewald():
        q = kw["latent_charges"].clone().requires_grad_(True)
        x = kw["positions"].clone().requires_grad_(True)
        res = model.les(**{**kw, "latent_charges": q, "positions": x})
        torch.autograd.grad(res["E_lr"].sum(), (q, x))

    def ewald_fwd():
        with torch.no_grad():
            model.les(**kw)

    t_ewald = cuda_time(ewald, 10)
    t_ewald_fwd = cuda_time(ewald_fwd, 10)
    e_lr = float(model.les(**kw)["E_lr"].sum())
    row = {"stack": stack, "system": label, "atoms": len(atoms), "full_ms": t_full, "ewald_fwd_bwd_ms": t_ewald,
           "ewald_fwd_ms": t_ewald_fwd, "fraction": t_ewald / t_full, "E_lr_eV": e_lr, "max_force": float(f.abs().max())}
    print(f"{stack:5s} {label}: full call {t_full:.2f} ms | Ewald {t_ewald:.2f} ms (forward {t_ewald_fwd:.2f}) = "
          f"{100 * row['fraction']:.1f}% | E_lr {e_lr:.3f} eV", flush=True)
    return row


def main() -> int:
    dev = "cuda"
    torch.set_default_dtype(torch.float32)
    models = fresh_models(dev)
    captured = {}

    def grab(module, args, kwargs, output):  # record the Ewald module's inputs (a hook: the call is unchanged)
        captured["kwargs"] = {k: (v.detach().clone() if isinstance(v, torch.Tensor) else v) for k, v in kwargs.items()}

    w = water216()
    systems = [("water216 (648 atoms)", w), ("ala2 in 454 waters (1,384 atoms)", ala2_water())]
    for rep in (2, 3, 4):
        big = w.repeat((rep, rep, rep))
        systems.append((f"water216 x{rep}^3 ({len(big):,} atoms)", big))
    rows = []
    # the strongest stack first (it decides); plain e3nn is context and runs out of memory past ~6,000 atoms (the
    # third profile run: 46 GB at 17,496); a box that does not fit is recorded as such, not fatal
    for (stack, model), (label, atoms) in [(sm, sy) for sm in (("cueq", models["cueq"]), ("e3nn", models["e3nn"]))
                                           for sy in systems]:
        if stack == "e3nn" and len(atoms) > 6000:
            rows.append({"stack": stack, "system": label, "atoms": len(atoms), "skipped": "plain e3nn past 6,000 atoms"})
            continue
        handle = model.les.register_forward_hook(grab, with_kwargs=True)
        try:
            row = profile_one(stack, model, label, atoms, captured, dev)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as err:
            if "out of memory" not in str(err).lower():
                raise
            row = {"stack": stack, "system": label, "atoms": len(atoms), "oom": True}
            print(f"{stack:5s} {label}: out of memory", flush=True)
        handle.remove()
        rows.append(row)
        torch.cuda.empty_cache()
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"model": str(MODEL.relative_to(ROOT)), "gpu": torch.cuda.get_device_name(0), "dtype": "float32",
           "when": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "rows": rows}
    (OUT / "les_profile.json").write_text(json.dumps(res, indent=1))
    print("->", OUT / "les_profile.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
