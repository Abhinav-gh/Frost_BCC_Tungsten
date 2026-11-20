"""A long-range MACE teacher (MACELES: Cheng's latent Ewald summation inside MACE) whose Ewald part can be held across
teacher calls: the third level of the hierarchy in further_expansions_beyond.md 6.8 (2026-11-15).

MACELES (``mace.modules.extensions.MACELES``) computes node energies and latent charges q(x) from the same MACE features
and adds E_lr = Ewald(q, x), a reciprocal-space sum over Gaussian charges whose cost grows ~N^2 (44.5% of the call at
17,496 atoms on cuEquivariance, ``les_profile.json``). Its force splits by the chain rule:

    F = F_sr - sum_j phi_j dq_j/dx + g,     phi_j = dE_lr/dq_j (the potential at j),  g = -dE_lr/dx |_q (the direct force).

Holding the whole long-range force would hold the charge path -sum phi dq/dx too, which changes as fast as the local
force (q follows every bond vibration). This holds only what the Ewald sum produces, the potential and the direct force,
evaluated at a refresh point (x0, q0):

    E~(x) = E_sr(x) + E_lr(x0, q0) + phi0 . (q(x) - q0) - g0 . (x - x0),

first order in (q - q0, x - x0). Its force F_sr - sum phi0_j dq_j/dx + g0 is exact at x0 (the chain rule above with
fresh phi and g), recomputes the charge path through the network on every call, and needs no Ewald sum: one network
forward and backward, so it can run inside a CUDA graph (the Ewald module syncs with the host). The error is second order:
the potential from the charges' change, G (q - q0), and the field's change along x - x0.

No library object is modified: ``HeldLongRange`` holds a MACELES and calls its submodules in the order its own forward
does (``frost.mace_compose``, as ``frost.gcm.neuralgcm_split.SplitModel`` composes NeuralGCM), with three modes: "exact" (MACELES's energy and
forces), "held" (E~ above) and "short" (the short-range energy alone). Energies and forces only (the positions gradient);
virials and edge forces are refused, because the held direct term depends on absolute positions, not on edge vectors.
"""

from __future__ import annotations

import importlib

import torch
from mace.modules.utils import get_outputs

from frost.mace_compose import node_energies

ALLOWED_TOP = {"torch", "e3nn", "mace", "les"}


def load_restricted(path, device="cpu"):
    """A released MACELES checkpoint through torch's restricted unpickler (weights_only): only the globals the file names
    are allowed, and only if every one comes from torch, e3nn, mace or les (read from the pickle without executing it)."""
    names = torch.serialization.get_unsafe_globals_in_checkpoint(str(path))
    outside = [n for n in names if n.split(".")[0] not in ALLOWED_TOP]
    if outside:
        raise RuntimeError(f"refusing {path}: globals outside {sorted(ALLOWED_TOP)}: {outside}")
    allowed = []
    for name in names:
        mod, attr = name.rsplit(".", 1)
        allowed.append(getattr(importlib.import_module(mod), attr))
    with torch.serialization.safe_globals(allowed):
        return torch.load(path, map_location=device, weights_only=True)


def fresh_maceles(path, device="cuda", stack="cueq"):
    """The released weights in a model built here with MACE's public tools, never the checkpoint's own scripted modules
    (they keep the authors' CPU constants): ``extract_config_mace_model`` and the class's constructor on the CPU with a
    state-dict copy (plain e3nn), then ``convert_e3nn_cueq.run`` for cuEquivariance kernels. float32, eval, frozen."""
    src = load_restricted(path, "cpu").float().eval()
    if stack == "cueq":
        from mace.cli.convert_e3nn_cueq import run as to_cueq

        model = to_cueq(src, device=device, return_model=True)
    else:
        from mace.tools.scripts_utils import extract_config_mace_model

        model = type(src)(**extract_config_mace_model(src))
        res = model.load_state_dict(src.state_dict(), strict=False)
        extra = [k for k in res.missing_keys if not k.endswith("_zeroed")]
        if extra or res.unexpected_keys:
            raise RuntimeError(f"state dict mismatch: missing {extra}, unexpected {res.unexpected_keys}")
        model = model.to(device)
    model = model.float().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


class HeldLongRange(torch.nn.Module):
    """A MACELES with its Ewald part switchable: ``mode`` "exact", "held" or "short" (module docstring). ``refresh`` and
    ``exact_and_refresh`` evaluate the Ewald part and hold its potential, direct force, charges and energy in place (the
    same storage every time, so a CUDA graph of the held call reads the new values)."""

    def __init__(self, model):
        super().__init__()
        if not hasattr(model, "les") or not hasattr(model, "les_readouts"):
            raise TypeError("HeldLongRange needs a MACELES (mace.modules.extensions.MACELES)")
        self.model = model
        self.mode = "exact"
        self.held: dict[str, torch.Tensor] = {}
        self.refreshes = 0

    @property
    def r_max(self):
        return self.model.r_max

    @property
    def atomic_numbers(self):
        return self.model.atomic_numbers

    @property
    def heads(self):
        return getattr(self.model, "heads", ["Default"])

    # ---- MACELES's forward, in its own order, up to the energies and charges (frost.mace_compose)
    def _features(self, data, charges: bool):
        return node_energies(self.model, data, charges=charges)

    def _ewald(self, q, x, data, cell):
        m = self.model
        cell_les = cell.clone()
        no_pbc = ~data["pbc"].to(device=cell.device).any(dim=-1)
        rows = no_pbc.repeat_interleave(3)
        cell_les[rows] = torch.zeros((int(rows.sum()), 3), dtype=cell_les.dtype, device=cell_les.device)
        out = m.les(latent_charges=q, positions=x, cell=cell_les.view(-1, 3, 3), batch=data["batch"],
                    compute_energy=True, compute_bec=False, bec_output_index=getattr(m, "bec_output_index", None))
        return out["E_lr"]

    def forward(self, data, training: bool = False, compute_force: bool = True, compute_virials: bool = False,
                compute_stress: bool = False, compute_displacement: bool = False, compute_hessian: bool = False,
                compute_edge_forces: bool = False, **unused):
        if compute_virials or compute_stress or compute_displacement or compute_hessian or compute_edge_forces:
            raise NotImplementedError("HeldLongRange gives energies and forces only (see the module docstring)")
        mode = self.mode
        ctx, e0, inter_e, node_e0, node_inter, q = self._features(data, charges=(mode != "short"))
        x = ctx.positions
        if mode == "exact":
            e_lr = self._ewald(q, x, data, ctx.cell)
        elif mode == "held":
            if ctx.num_graphs != 1:
                raise NotImplementedError("the held mode serves one configuration at a time")
            h = self.held
            e_lr = (h["e_lr"] + (h["phi"] * (q - h["q"])).sum() - (h["g"] * (x - h["x"])).sum()).reshape(1)
        elif mode == "short":
            e_lr = torch.zeros_like(inter_e)
        else:
            raise ValueError(f"unknown mode {mode!r}")
        forces, *_ = get_outputs(energy=inter_e + e_lr, positions=x, cell=ctx.cell, displacement=ctx.displacement,
                                 vectors=ctx.vectors, training=training, compute_force=compute_force,
                                 compute_virials=False, compute_stress=False)
        node_energy = node_e0.clone().double() + node_inter.clone().double()
        return {"energy": e0 + inter_e + e_lr, "node_energy": node_energy, "forces": forces, "les_energy": e_lr,
                "latent_charges": q, "edge_forces": None, "virials": None, "stress": None}

    # ---- refreshing the held part (eager: the Ewald module syncs with the host)
    def _store(self, new: dict) -> None:
        for k, v in new.items():
            if k in self.held and self.held[k].shape == v.shape:
                self.held[k].copy_(v)
            else:
                self.held[k] = v.clone()
        self.refreshes += 1

    def _ewald_leaves(self, q, x, data, cell):
        q_leaf = q.detach().clone().requires_grad_(True)
        x_leaf = x.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            e_lr = self._ewald(q_leaf, x_leaf, data, cell)
            phi, de_dx = torch.autograd.grad(e_lr.sum(), (q_leaf, x_leaf))
        return e_lr.detach().reshape(1), phi.detach(), (-de_dx).detach(), x_leaf.detach()

    def refresh(self, data) -> dict:
        """Evaluate the Ewald part at ``data``'s positions (one network forward for the charges, no backward through
        it) and hold it. Returns the held tensors."""
        d = dict(data)
        d["positions"] = data["positions"].detach().clone()
        with torch.no_grad():
            ctx, _, _, _, _, q = self._features(d, charges=True)
        e_lr, phi, g, x0 = self._ewald_leaves(q, ctx.positions, d, ctx.cell)
        self._store({"phi": phi, "g": g, "q": q.detach(), "x": x0, "e_lr": e_lr})
        return self.held

    def exact_and_refresh(self, data, training: bool = False) -> dict:
        """The exact energy and forces at ``data``'s positions in one network forward and backward (the held formula with
        the potential and direct force just computed, which is the chain rule), holding the Ewald part as it goes."""
        d = dict(data)
        d["positions"] = data["positions"].detach().clone()
        ctx, e0, inter_e, node_e0, node_inter, q = self._features(d, charges=True)
        x = ctx.positions
        e_lr0, phi, g, x0 = self._ewald_leaves(q, x, d, ctx.cell)
        self._store({"phi": phi, "g": g, "q": q.detach(), "x": x0, "e_lr": e_lr0})
        e_lr = (e_lr0 + (phi * (q - q.detach())).sum() - (g * (x - x0)).sum()).reshape(1)
        forces, *_ = get_outputs(energy=inter_e + e_lr, positions=x, cell=ctx.cell, displacement=ctx.displacement,
                                 vectors=ctx.vectors, training=training, compute_force=True, compute_virials=False,
                                 compute_stress=False)
        return {"energy": e0 + inter_e + e_lr, "forces": forces, "les_energy": e_lr, "latent_charges": q.detach()}
