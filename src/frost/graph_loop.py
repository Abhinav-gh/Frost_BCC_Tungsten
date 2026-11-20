"""k inner MD steps as one CUDA graph: the multirate inner loop without host overhead.

Profiling the liquid-Cu async run (k = 12) showed 2.78 ms per inner step against
about 1.0 ms of student device time. The rest is host work, step by step:
- the integrator's small kernels;
- a neighbour-list displacement check that syncs the host every step;
- a device-to-host copy for sampling.

The host and the GPU take turns instead of overlapping. This module captures the
whole interval instead:
- BAOAB (or velocity Verlet) for k steps;
- the force model;
- the kinetic energy and samples, written to device buffers;
- the running maximum displacement since the neighbour list was built.

That leaves one replay and one sync per outer step.

**Neighbour lists are reused speculatively.** The graph's edge list is fixed for
the interval. The caller checks once, at the boundary, that no atom moved more
than half the skin. If one did, the interval is redone on the eager path: the
same rollback machinery as ``validate.py --rollback``, restoring x, v, f and the
generator state.

**The noise is identical to the eager integrator's.** Before each replay the k
Langevin draws are made one step at a time, with the same generator and shape as
``Integrator.inner_step``, into a static buffer. A graph run and an eager run of
the same arm see the same noise sequence, so the teacher and frost arms stay
noise-matched.

Nothing here changes a library: the force models are called through their public
forward, inside our own capture.
"""

from __future__ import annotations

import torch

from frost.padding import spread_padding


class StudentForce:
    """A student's force on static edge buffers, callable inside a capture."""

    def __init__(self, model, types: torch.Tensor, n_atoms: int, capacity: int, dtype=torch.float32):
        self.model, self.types, self.n_atoms, self.dtype = model, types, n_atoms, dtype
        self._allocate(capacity)

    def _allocate(self, capacity: int) -> None:
        dev = self.types.device
        self.capacity = capacity
        self.x = torch.zeros(self.n_atoms, 3, device=dev, dtype=self.dtype, requires_grad=True)
        self.i = torch.zeros(capacity, dtype=torch.long, device=dev)
        self.j = torch.zeros(capacity, dtype=torch.long, device=dev)
        self.shifts = torch.zeros(capacity, 3, device=dev, dtype=self.dtype)
        self.mask = torch.zeros(capacity, device=dev, dtype=self.dtype)

    def set_edges(self, i, j, shifts) -> bool:
        """Write a new list into the static buffers. True if they had to grow (recapture)."""
        n = int(i.numel())
        grew = n > self.capacity
        if grew:
            self._allocate(int(1.25 * n) + 1)
        with torch.no_grad():
            self.mask.zero_()
            self.shifts.zero_()
            self.i[:n], self.j[:n] = i, j
            pad = spread_padding(n, self.capacity, self.n_atoms, self.i.device)  # not all on atom 0: frost.padding
            self.i[n:], self.j[n:] = pad, pad
            self.shifts[:n] = shifts.to(self.dtype)
            self.mask[:n] = 1.0
        return grew

    def __call__(self, x64):
        with torch.no_grad():
            self.x.copy_(x64.to(self.dtype))
        e = self.model(self.x, self.types, self.i, self.j, self.shifts, edge_mask=self.mask)
        (g,) = torch.autograd.grad(e.sum(), self.x)
        return e.sum().detach(), (-g).detach().double()


class TeacherForce:
    """``GraphedTeacher``'s model on its static buffers, callable inside a capture."""

    def __init__(self, graphed_teacher):
        self.t = graphed_teacher

    def set_edges(self, i, j, shifts, unit_shifts) -> bool:
        before = self.t.capacity
        self.t.set_edges(i, j, shifts, unit_shifts)
        return self.t.capacity != before

    def __call__(self, x64):
        with torch.no_grad():
            self.t.data["positions"].copy_(x64.to(self.t.dtype))
        out = self.t.model(self.t.data, compute_force=True, training=False)
        return out["energy"].sum().detach(), out["forces"].detach().double()


class GraphedLoop:
    """k steps of ``Integrator.inner_step`` (BAOAB, or VV without a thermostat) in one graph.

    ``run(x, v, f, x_ref)`` advances k steps from (x, v) with f the force at x, and
    returns static buffers:
    - x, v, f after the interval;
    - per step: energy, kinetic energy, float32 samples of x and v;
    - the maximum displacement from ``x_ref`` (the list's reference positions) over
      the interval.
    """

    def __init__(self, force, masses: torch.Tensor, dt: float, k: int, thermostat=None):
        self.force, self.masses, self.h, self.k, self.th = force, masses, dt, k, thermostat
        n, dev = masses.shape[0], masses.device
        f64 = dict(device=dev, dtype=torch.float64)
        self.x, self.v, self.f = (torch.zeros(n, 3, **f64) for _ in range(3))
        self.x_ref = torch.zeros(n, 3, **f64)
        self.noise = torch.zeros(k, n, 3, **f64)
        self.energy = torch.zeros(k, **f64)
        self.kinetic = torch.zeros(k, **f64)
        self.xs = torch.zeros(k, n, 3, device=dev, dtype=torch.float32)
        self.vs = torch.zeros(k, n, 3, device=dev, dtype=torch.float32)
        self.maxdisp = torch.zeros((), **f64)
        self.graph = None
        self.captures = 0

    def _body(self):
        h, m = self.h, self.masses
        x, v, f = self.x.clone(), self.v.clone(), self.f.clone()
        disp = torch.zeros((), device=x.device, dtype=x.dtype)
        for j in range(self.k):
            v = v + 0.5 * h * f / m
            if self.th is None:
                x = x + h * v
            else:
                x = x + 0.5 * h * v
                v = self.th.c1 * v + self.th.c2 * self.th.sigma * self.noise[j]
                x = x + 0.5 * h * v
            e, f = self.force(x)
            v = v + 0.5 * h * f / m
            self.energy[j].copy_(e)
            self.kinetic[j].copy_(0.5 * (m * v * v).sum())
            self.xs[j].copy_(x)
            self.vs[j].copy_(v)
            disp = torch.maximum(disp, (x - self.x_ref).norm(dim=1).max())
        self.x.copy_(x)
        self.v.copy_(v)
        self.f.copy_(f)
        self.maxdisp.copy_(disp)

    def _capture(self) -> None:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())  # trap 12: wait for the static inputs
        snapshot = (self.x.clone(), self.v.clone(), self.f.clone())
        with torch.cuda.stream(side):
            for _ in range(2):
                self._body()
                self.x.copy_(snapshot[0])
                self.v.copy_(snapshot[1])
                self.f.copy_(snapshot[2])
        torch.cuda.current_stream().wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._body()
        self.captures += 1

    def invalidate(self) -> None:
        """The force's static buffers were reallocated: recapture before the next run."""
        self.graph = None

    def run(self, x, v, f, x_ref):
        with torch.no_grad():
            self.x.copy_(x)
            self.v.copy_(v)
            self.f.copy_(f)
            self.x_ref.copy_(x_ref)
            if self.th is not None:
                # the eager integrator's draws, in its order and shape: noise-identical runs
                for j in range(self.k):
                    self.noise[j].copy_(torch.randn(v.shape, generator=self.th.gen, device=v.device,
                                                    dtype=v.dtype))
        if self.graph is None:
            self._capture()
        self.graph.replay()
        return self.x, self.v, self.f
