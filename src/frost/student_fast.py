"""Faster student force calls: compiled kernels captured in a CUDA graph.

G2 is measured against the cuEquivariance teacher, which is about 4x faster than
e3nn on silica. So the student needs fused kernels too, not just fewer Python
launches. ``CompiledGraphedStudent`` compiles the student's energy with
TorchInductor, so forward and backward become a few fused Triton kernels, and
captures the compiled force call in a CUDA graph exactly as ``GraphedStudent``
does. The eager ``GraphedStudent`` remains the reference: always compare the two
on a frame before trusting the compiled one (``experiments/e5_student/student_cost.py``).
"""

from __future__ import annotations

import torch

from frost.student import GraphedStudent


class AutocastStudent(torch.nn.Module):
    """A student whose energy is evaluated under ``torch.autocast``.

    Autocast picks a dtype per operator. Matrix multiplies and linear layers run
    in ``dtype``, and ops it does not list take whatever their inputs are.
    Benoit (2025, arXiv:2510.23621) reports that half precision confined to MACE's
    linear layers keeps MD observables. Precision elsewhere broke our all-bfloat16
    student, so always compare forces with the plain student before trusting
    this one (``experiments/e5_student/student_cost.py``). The energy comes back in
    float32. This is composition through a public API; nothing is patched.
    """

    def __init__(self, student, dtype=torch.bfloat16):
        super().__init__()
        self.student = student
        self.dtype = dtype
        self.config = {**student.config, "autocast": str(dtype).removeprefix("torch.")}

    def types(self, numbers: torch.Tensor) -> torch.Tensor:
        return self.student.types(numbers)

    def forward(self, x, types, edge_i, edge_j, shifts, batch=None, n_graphs: int = 1,
                edge_mask=None):
        with torch.autocast(device_type=x.device.type, dtype=self.dtype):
            energy = self.student(x, types, edge_i, edge_j, shifts, batch, n_graphs, edge_mask)
        return energy.float()


class DetachedStudent:
    """A student called with its parameters detached: forces by autograd differentiate to the positions only.

    The tensors share the parameters' storage, so in-place optimiser updates are seen (and by a CUDA graph that
    captured this call). Wrap it in ``torch.compile`` for the inner loop; see ``CompiledGraphedStudent``.
    """

    def __init__(self, model):
        self.model = model
        self.params = {k: v.detach() for k, v in model.named_parameters()}
        self.buffers = dict(model.named_buffers())

    def __call__(self, x, types, edge_i, edge_j, shifts, batch=None, n_graphs: int = 1, edge_mask=None):
        return torch.func.functional_call(self.model, (self.params, self.buffers),
                                          (x, types, edge_i, edge_j, shifts, batch, n_graphs),
                                          {"edge_mask": edge_mask})


class CompiledGraphedStudent(GraphedStudent):
    """``GraphedStudent`` with the energy compiled by TorchInductor before capture.

    The compiled energy is called with the parameters *detached* (``torch.func.functional_call`` on
    ``p.detach()``: the same storage, so in-place optimiser updates still reach the captured graph). Inductor
    compiles one backward for every input that requires grad, and ``autograd.grad`` to the positions then
    discards the rest: with live parameters the force call computed every weight gradient as well, 21% of its
    GPU time in the pair-gate embedding's backward alone (Cu, 21k edges). Detached, the backward is dE/dx only.
    """

    def __init__(self, model, types: torch.Tensor, n_atoms: int, capacity: int,
                 dtype=torch.float32, mode: str = "default"):
        def energy(params, buffers, x, types, i, j, shifts, mask):
            return torch.func.functional_call(model, (params, buffers), (x, types, i, j, shifts),
                                              {"edge_mask": mask})

        self.compiled_energy = torch.compile(energy, mode=mode, dynamic=False)
        self.params = {k: v.detach() for k, v in model.named_parameters()}
        self.buffers = dict(model.named_buffers())
        super().__init__(model, types, n_atoms, capacity, dtype)

    def _capture(self) -> None:
        def call():
            if getattr(self.model, "returns_forces", False):  # a force-only student: its output is the force
                with torch.no_grad():
                    f = self.compiled_energy(self.params, self.buffers, self.x, self.types, self.i, self.j,
                                             self.shifts, self.mask)
                return f.new_zeros(1), f
            e = self.compiled_energy(self.params, self.buffers, self.x, self.types, self.i, self.j, self.shifts,
                                     self.mask)
            (g,) = torch.autograd.grad(e.sum(), self.x)
            return e, -g

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())  # the static inputs were just written there
        with torch.cuda.stream(side):
            for _ in range(3):
                call()
        torch.cuda.current_stream().wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.energy, self.forces = call()
