"""A small MACE as the in-situ student.

The body-order-3 student in ``frost.student`` is cheap but capacity-limited on
silica (week-1 sweep, plan section 1.9). This wraps mace-torch's own model,
trained from scratch at a short cutoff with the same loss, so the replay and
online harnesses can compare the two families on equal terms.

It keeps the ``FrostStudent`` calling convention:
``forward(x, types, edge_i, edge_j, shifts, batch, n_graphs, edge_mask)``
returns total energies per graph, and forces come from autograd outside, so
``InSituTrainer`` and ``GraphedStudent`` work unchanged.

Masked or padded edges are pushed beyond the cutoff rather than zeroed. MACE's
radial basis divides by the edge length, and its polynomial envelope is exactly
zero beyond r_max, so far edges contribute nothing.

Kernel backends. ``backend`` selects mace-torch's own implementations of the
same model: "e3nn" (the reference), "cueq" (NVIDIA cuEquivariance) or "oeq"
(OpenEquivariance). They are configured through mace's constructor arguments,
never by patching modules. ``converted`` moves trained weights from the e3nn
student to another backend with mace's converter functions. A backend that is
not importable raises instead of silently falling back to e3nn.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn

_FAR = 1.0e3  # A; any length beyond r_max works
BACKENDS = ("e3nn", "cueq", "oeq")


def kernel_config(backend: str) -> dict:
    """mace-torch constructor arguments that select a kernel backend."""
    from mace.modules import wrapper_ops

    if backend == "e3nn":
        return {}
    if backend == "cueq":
        if not wrapper_ops.CUET_AVAILABLE:
            raise RuntimeError("cuEquivariance is not importable; refusing to fall back to e3nn")
        return {"cueq_config": wrapper_ops.CuEquivarianceConfig(
            enabled=True, layout="ir_mul", group="O3_e3nn", optimize_all=True, conv_fusion=True)}
    if backend == "oeq":
        if not wrapper_ops.OEQ_AVAILABLE:
            raise RuntimeError("OpenEquivariance is not importable; refusing to fall back to e3nn")
        return {"oeq_config": wrapper_ops.OEQConfig(
            enabled=True, optimize_all=True, conv_fusion="atomic")}
    raise ValueError(f"unknown kernel backend {backend!r}; choose from {BACKENDS}")


class MaceStudent(nn.Module):
    def __init__(self, species: Sequence[int], cutoff: float = 5.0, channels: int = 32,
                 max_ell: int = 2, layers: int = 2, correlation: int = 3,
                 mean_neighbours: float = 20.0, vector_channels: int | None = None,
                 backend: str = "e3nn", num_bessel: int = 8):
        super().__init__()
        from e3nn import o3
        from mace import modules

        kernels = kernel_config(backend)
        z = sorted({int(s) for s in species})
        table = torch.full((119,), -1, dtype=torch.long)
        for k, s in enumerate(z):
            table[s] = k
        self.species = z
        self.n_species = len(z)
        vec = channels if vector_channels is None else vector_channels
        hidden = f"{channels}x0e + {vec}x1o" if vec > 0 else f"{channels}x0e"
        self.config = {
            "kind": "mace", "species": z, "cutoff": float(cutoff), "channels": channels,
            "vector_channels": vec, "max_ell": max_ell, "layers": layers,
            "correlation": correlation, "mean_neighbours": float(mean_neighbours),
            "backend": backend, "num_bessel": num_bessel,
        }
        self.register_buffer("species_table", table)
        self.model = modules.ScaleShiftMACE(
            atomic_inter_scale=1.0,
            atomic_inter_shift=0.0,
            r_max=float(cutoff),
            num_bessel=num_bessel,
            num_polynomial_cutoff=5,
            max_ell=max_ell,
            interaction_cls=modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
            interaction_cls_first=modules.interaction_classes["RealAgnosticInteractionBlock"],
            num_interactions=layers,
            num_elements=self.n_species,
            hidden_irreps=o3.Irreps(hidden),
            MLP_irreps=o3.Irreps("16x0e"),
            atomic_energies=np.zeros(self.n_species),
            avg_num_neighbors=float(mean_neighbours),
            atomic_numbers=z,
            correlation=correlation,
            gate=torch.nn.functional.silu,
            radial_MLP=[32, 32, 32],
            **kernels,
        )

    def converted(self, backend: str) -> MaceStudent:
        """The same student, with its trained weights, on another kernel backend."""
        cfg = self.config
        if backend == cfg["backend"]:
            return self
        if cfg["backend"] != "e3nn" and backend != "e3nn":
            return self.converted("e3nn").converted(backend)  # through e3nn, mace's hub format
        ref = next(self.parameters())
        other = MaceStudent(cfg["species"], cutoff=cfg["cutoff"], channels=cfg["channels"],
                            max_ell=cfg["max_ell"], layers=cfg["layers"],
                            correlation=cfg["correlation"], mean_neighbours=cfg["mean_neighbours"],
                            vector_channels=cfg["vector_channels"], backend=backend,
                            num_bessel=cfg.get("num_bessel", 8))
        other = other.to(device=ref.device, dtype=ref.dtype)
        if cfg["backend"] == "cueq":  # cueq -> e3nn, with mace's own converter
            from mace.cli.convert_cueq_e3nn import transfer_weights as to_e3nn
            from mace.tools.scripts_utils import extract_config_mace_model

            source = extract_config_mace_model(self.model)
            to_e3nn(self.model, other.model, len(source["hidden_irreps"].slices()) - 1,
                    source["correlation"], source["num_interactions"],
                    source.get("use_reduced_cg", True), source.get("keep_last_layer_irreps", False))
        elif backend == "cueq":
            from mace.cli.convert_e3nn_cueq import transfer_weights
            from mace.tools.scripts_utils import extract_config_mace_model

            source = extract_config_mace_model(self.model)
            transfer_weights(self.model, other.model, len(source["hidden_irreps"].slices()) - 1,
                             source["correlation"], source["num_interactions"],
                             source.get("use_reduced_cg", True),
                             source.get("keep_last_layer_irreps", False))
        else:
            # as mace.cli.convert_e3nn_oeq: OpenEquivariance keeps e3nn's parameter
            # layout, and its convolution tensor product holds no weights
            src = self.model.state_dict()
            dst = other.model.state_dict()
            for key in dst:
                if ".conv_tp." not in key and key in src:
                    dst[key] = src[key]
            other.model.load_state_dict(dst)
        return other

    def types(self, numbers: torch.Tensor) -> torch.Tensor:
        """Atomic numbers to the student's species indices."""
        t = self.species_table[numbers.long()]
        if bool((t < 0).any()):
            raise ValueError("an atomic number is outside the student's species list")
        return t

    def forward(self, x, types, edge_i, edge_j, shifts, batch=None, n_graphs: int = 1,
                edge_mask=None):
        """Total energy per graph, shape (n_graphs,). Graphs must be the same size."""
        return self._run(x, types, edge_i, edge_j, shifts, batch, n_graphs, edge_mask)["energy"]

    def energy_and_edge_forces(self, x, types, edge_i, edge_j, shifts, edge_mask=None):
        """Energy and the conservative forces -dE/dx, through mace's public edge-force output.

        mace returns dE/dr per edge (r = x[j] - x[i] + shift); the atomic forces are its sums with index_add_.
        Differentiating to x instead runs PyTorch's sort-based backward of mace's edge gather, about 30% of a
        MACE force call (experiments/e13_runtime/teacher_profile.py). Same forces to float32 round-off.
        """
        out = self._run(x, types, edge_i, edge_j, shifts, edge_mask=edge_mask, edge_forces=True)
        g = out["edge_forces"]
        forces = torch.zeros_like(x).index_add_(0, edge_i, g).index_add_(0, edge_j, -g)
        return out["energy"], forces

    def node_features(self, x, types, edge_i, edge_j, shifts):
        """Per-atom invariant (0e) features of every layer, shape (n_atoms, channels).

        MACE concatenates each layer's node features; every layer but the last
        carries ``channels`` scalars then ``vector_channels`` 1o vectors, and the last
        carries scalars only. Differentiable in ``x``, so a linear head on these is a
        conservative energy term (new_directions.md, C5).
        """
        feats = self._run(x, types, edge_i, edge_j, shifts)["node_feats"]
        c, v, layers = (self.config["channels"], self.config["vector_channels"],
                        self.config["layers"])
        width = c + 3 * v
        blocks = [feats[:, k * width:k * width + c] for k in range(layers - 1)]
        blocks.append(feats[:, (layers - 1) * width:(layers - 1) * width + c])
        return torch.cat(blocks, dim=-1)

    def _run(self, x, types, edge_i, edge_j, shifts, batch=None, n_graphs: int = 1,
             edge_mask=None, edge_forces: bool = False):
        n = x.shape[0]
        if batch is None:
            batch = torch.zeros(n, dtype=torch.long, device=x.device)
            n_graphs = 1
        ptr = torch.arange(n_graphs + 1, device=x.device) * (n // n_graphs)
        if edge_mask is not None:
            shifts = shifts + (1.0 - edge_mask).unsqueeze(-1) * _FAR
        data = {
            "positions": x,
            "node_attrs": torch.nn.functional.one_hot(types, self.n_species).to(x.dtype),
            "edge_index": torch.stack((edge_i, edge_j)),
            "shifts": shifts,
            "unit_shifts": torch.zeros_like(shifts),
            "cell": torch.zeros(n_graphs, 3, 3, dtype=x.dtype, device=x.device),
            "batch": batch,
            "ptr": ptr,
        }
        if edge_forces:
            return self.model(data, training=False, compute_force=False, compute_virials=False,
                              compute_stress=False, compute_edge_forces=True)
        return self.model(data, training=True, compute_force=False)
