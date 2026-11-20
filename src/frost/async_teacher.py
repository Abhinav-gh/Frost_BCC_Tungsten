"""The teacher one outer step behind, on its own GPU (new_directions.md, C2).

Synchronous Frost pays, per outer step, k student steps and then one teacher call,
in series. The teacher does not have to be on the critical path. While the student
integrates interval n on one GPU, the teacher can evaluate the configuration that
closed interval n - 1 on the other. Wall time per outer step falls from
k c_S + c_T to max(k c_S, c_T).

The price is a lag. The kick at boundary n can only use the newest *finished* call,
R(x_{n-1}) = F_T(x_{n-1}) - F_S(x_{n-1}). A force computed from a past state is
delayed feedback. On one harmonic mode with curvature mismatch mu and
theta = omega D, a one-step lag multiplies the amplitude by about
1 + mu theta sin(theta) / 2 per outer step: negative friction. Langevin friction
with gamma D >= mu theta sin(theta) removes it. Extrapolating the lagged residual
from the two newest calls makes it *worse*: 1.022 against 1.013 at mu = 0.02,
0.22 T (`experiments/e9_direction/async_theory.py`). So the lagged residual is held,
never extrapolated. The scheme is not time reversible, so it belongs to the
Langevin (NVT) contract.

Pieces:
- ``ProcessTeacher`` serves the teacher from a child process that sees only the
  second GPU. It is the production path, for a measured reason. cuEquivariance's
  kernel state is process-wide and assumes one device. With a cuda:0 teacher
  already running:
  - a worker *thread* capturing a cuda:1 graph fails with ``cudaFree(nullptr)``
    during capture (cudaErrorStreamCaptureUnsupported);
  - a cuda:1 graph captured from the main thread and replayed from a worker faults
    with an illegal address.

  One device per process sidesteps both without touching the library. Positions go
  in and forces come out over a pipe, about 27 kB per call on 1152 atoms, against a
  37 ms teacher call.
- ``AsyncTeacher`` is the same contract on a worker thread. It is used for CPU
  teachers and the unit tests. Calls run in submission order either way, so the
  lag is exactly one call.
- ``LaggedResidual`` is the bookkeeping at each boundary: submit x_n, collect the
  call on x_{n-1}, and return the residual to kick with.
- ``mace_teacher_builder`` builds the production teacher (graphed cuEquivariance
  MACE on a Verlet list) inside the worker.
"""

from __future__ import annotations

import os
import traceback
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np
import torch

TeacherCall = Callable[[torch.Tensor], tuple[float, torch.Tensor]]


class AsyncTeacher:
    """A teacher served by one worker thread, pinned to ``device``.

    ``build`` runs inside the worker, after the thread's current CUDA device is set,
    and returns ``call(x_cpu) -> (energy, forces)``. Calls run in submission order.
    """

    def __init__(self, build: Callable[[], TeacherCall], device: str | None = None):
        self.device = device
        self._call: TeacherCall | None = None
        self._pool = ThreadPoolExecutor(max_workers=1, initializer=self._init,
                                        initargs=(build, device))
        # building inside the initializer happens on first submit; force it now so
        # construction never lands inside a timed region
        self._pool.submit(lambda: None).result()

    def _init(self, build: Callable[[], TeacherCall], device: str | None) -> None:
        if device is not None and device.startswith("cuda"):
            dev = torch.device(device)
            torch.cuda.set_device(dev)
            # bind the device's context to this thread before anything is captured:
            # cuEquivariance binds it lazily with cudaFree(nullptr) on first use,
            # which is illegal inside a CUDA-graph capture
            torch.zeros(1, device=dev)
            torch.cuda.synchronize(dev)
        self._call = build()

    def _run(self, x_cpu: torch.Tensor) -> tuple[float, torch.Tensor]:
        energy, forces = self._call(x_cpu)
        return float(energy), forces.detach().to("cpu", torch.float64)

    def submit(self, x: torch.Tensor) -> Future:
        """Evaluate the teacher at ``x``; the copy to host is taken now."""
        return self._pool.submit(self._run, x.detach().to("cpu", torch.float64, copy=True))

    def __call__(self, x: torch.Tensor) -> tuple[float, torch.Tensor]:
        """A synchronous call, for the starting residual and for checks."""
        return self.submit(x).result()

    def close(self) -> None:
        self._pool.shutdown(wait=True)


class LaggedResidual:
    """Which residual to kick with at each outer boundary, ``depth`` calls behind.

    At boundary n: submit x_n, then collect the call on x_{n-1} (submitted one
    boundary earlier), and kick with R(x_{n-depth}). With one teacher device the
    newest finished call is always x_{n-1}; ``depth`` > 1 holds finished residuals
    longer. That reproduces the staleness an L-device round-robin pipeline would
    have, so its fidelity can be measured on two GPUs (its speed cannot). The
    residual is held, never extrapolated. ``boundary`` also returns what finished,
    so the caller can record the teacher energy against the kinetic energy of the
    boundary it belongs to.
    """

    def __init__(self, teacher, residual: torch.Tensor, depth: int = 1, compensate: bool = False,
                 types: torch.Tensor | None = None, f_student0: torch.Tensor | None = None, memory: float = 0.98):
        """``compensate`` (depth 1): kick with R(x_{n-1}) + mu (F_S(x_n) - F_S(x_{n-1})), mu the
        measured curvature mismatch, per species (``types``): the least-squares slope of the
        residual's change against the student force's change between finished calls, with
        exponential ``memory``. In the linear model of the two-GPU law (new_directions.md C2) the
        residual is -mu w^2 x and this is exact, so the lag's negative friction cancels; in general
        it removes the part of the lag error that is linear in the student's force change."""
        if depth < 1:
            raise ValueError("depth must be at least 1")
        if compensate and depth != 1:
            raise ValueError("lag compensation is for depth 1")
        self.teacher = teacher
        self.depth = depth
        self.residual = residual  # the newest finished residual
        self.history = [residual]  # finished residuals, oldest first, newest last
        self.pending = None
        self.waited_s = 0.0
        self.compensate, self.memory = compensate, memory
        self.types = types
        self.f_at_residual = f_student0  # the student force where the newest residual was measured
        n_types = int(types.max()) + 1 if types is not None else 1
        self.num = torch.zeros(n_types, dtype=torch.float64, device=residual.device)
        self.den = torch.zeros(n_types, dtype=torch.float64, device=residual.device)
        self.mu_log = []

    def _per_atom(self, per_type: torch.Tensor) -> torch.Tensor:
        return (per_type[self.types] if self.types is not None else per_type.expand(self.residual.shape[0])).unsqueeze(1)

    def mu(self) -> torch.Tensor:
        return torch.where(self.den > 0, self.num / self.den.clamp(min=1e-300), torch.zeros_like(self.num))

    def kick_residual(self, f_student: torch.Tensor | None = None) -> torch.Tensor:
        """R(x_{n-depth}): the finished residual depth - 1 places before the newest (compensated if asked)."""
        index = max(len(self.history) - self.depth, 0)
        r = self.history[index]
        if self.compensate and f_student is not None and self.f_at_residual is not None:
            r = r + self._per_atom(self.mu()).to(r.dtype) * (f_student - self.f_at_residual)
        return r

    def boundary(self, x: torch.Tensor, f_student: torch.Tensor, kinetic: float, step: int):
        import time

        future = self.teacher.submit(x)
        finished = None
        if self.pending is not None:
            finished = self._collect(time)
        self.pending = (future, f_student.detach().clone(), kinetic, step)
        return self.kick_residual(f_student), finished

    def _collect(self, time):
        previous, f_student, kinetic, step = self.pending
        t0 = time.perf_counter()
        energy, f_teacher = previous.result()
        self.waited_s += time.perf_counter() - t0
        new = f_teacher.to(f_student.device, f_student.dtype) - f_student
        if self.compensate and self.f_at_residual is not None:
            d_r, d_f = (new - self.residual).double(), (f_student - self.f_at_residual).double()
            num, den = (d_r * d_f).sum(1), (d_f * d_f).sum(1)
            if self.types is not None:
                num = torch.zeros_like(self.num).index_add_(0, self.types, num)
                den = torch.zeros_like(self.den).index_add_(0, self.types, den)
            else:
                num, den = num.sum().reshape(1), den.sum().reshape(1)
            self.num = self.memory * self.num + num
            self.den = self.memory * self.den + den
            self.mu_log.append(self.mu().tolist())
        finished = {"energy": energy, "kinetic": kinetic, "step": step,
                    "residual": new, "previous": self.residual}
        self.residual = new
        self.f_at_residual = f_student
        self.history.append(new)
        if len(self.history) > self.depth:
            del self.history[0]
        return finished

    def drain(self):
        """Collect the last call once the run has ended."""
        import time

        if self.pending is None:
            return None
        finished = self._collect(time)
        self.pending = None
        return finished


def mace_teacher_builder(atoms, cell, device: str, skin: float = 1.0, builder: str = "alchemi",
                         teacher: str = "mp-medium", edge_forces: bool = False,
                         tight: bool = False) -> Callable[[], TeacherCall]:
    """The production teacher (a ``frost.teachers`` name), built inside the worker thread on ``device``; with
    ``edge_forces``, mace's edge-force output summed onto atoms, as the runtime's own teacher runs it."""

    def build() -> TeacherCall:
        from frost.neighbours import VerletList
        from frost.teacher_fast import GraphedTeacher
        from frost.teachers import load_calc

        calc = load_calc(teacher, device, "float32", enable_cueq=True)
        model = GraphedTeacher(calc, atoms, device, edge_forces=edge_forces)
        t_list = VerletList(cell, model.r_max, skin, device, wrap_positions=False,
                            builder=builder)

        r_cut2 = float(model.r_max) ** 2

        def call(x_cpu: torch.Tensor):
            x = x_cpu.to(device, non_blocking=False)
            rebuilt = t_list.update(x, force=t_list.i is None)[1]
            if tight:  # the skin list filtered to r_max before every call, as md.build(tight_teacher=True)
                from frost.neighbours import tight_edges

                model.set_edges(*tight_edges(t_list, x, r_cut2))
            elif rebuilt:
                model.set_edges(t_list.i, t_list.j, t_list.shifts, t_list.unit_shifts)
            energy, forces = model(x)
            return energy, forces

        return call

    return build


class _Pending:
    """A call submitted to a ``ProcessTeacher``; ``result()`` blocks until it is back."""

    def __init__(self, owner: ProcessTeacher, call_id: int):
        self.owner = owner
        self.call_id = call_id

    def result(self) -> tuple[float, torch.Tensor]:
        return self.owner._wait(self.call_id)


def _serve(conn, spec) -> None:
    """Child process: build the teacher, then answer calls in order until told to stop."""
    try:
        call = spec()
        conn.send(("ready", None))
    except Exception:
        conn.send(("error", traceback.format_exc()))
        return
    while True:
        message = conn.recv()
        if message is None:
            break
        call_id, x = message
        try:
            energy, forces = call(torch.from_numpy(x))
            conn.send((call_id, (float(energy), forces.detach().to("cpu", torch.float64).numpy())))
        except Exception:
            conn.send((call_id, ("error", traceback.format_exc())))


class ProcessTeacher:
    """``AsyncTeacher``'s contract, served by a child process that sees one GPU.

    ``spec`` must be picklable (a top-level callable object): the child is started
    with the ``spawn`` method and calls ``spec()`` to build its teacher. The child
    sees only ``visible_device`` (a CUDA_VISIBLE_DEVICES value), so inside it the
    teacher's device is simply "cuda".
    """

    def __init__(self, spec, visible_device: str | None = None):
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        self._conn, child = ctx.Pipe()
        before = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible_device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = visible_device
        try:
            self._proc = ctx.Process(target=_serve, args=(child, spec), daemon=True)
            self._proc.start()
        finally:
            if visible_device is not None:
                if before is None:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = before
        status, detail = self._recv()
        if status != "ready":
            raise RuntimeError(f"teacher process failed to start:\n{detail}")
        self._next = 0
        self._done: dict[int, tuple] = {}

    def submit(self, x: torch.Tensor) -> _Pending:
        call_id = self._next
        self._next += 1
        self._conn.send((call_id, x.detach().to("cpu", torch.float64).numpy()))
        return _Pending(self, call_id)

    def _wait(self, call_id: int) -> tuple[float, torch.Tensor]:
        while call_id not in self._done:
            got_id, payload = self._recv()
            self._done[got_id] = payload
        payload = self._done.pop(call_id)
        if isinstance(payload[0], str) and payload[0] == "error":
            raise RuntimeError(f"teacher call failed in the child process:\n{payload[1]}")
        energy, forces = payload
        return energy, torch.from_numpy(np.asarray(forces))

    def _recv(self):
        """Receive, but never wait on a child that has died (a crash, or a spawn import error)."""
        while not self._conn.poll(1.0):
            if not self._proc.is_alive():
                raise RuntimeError(f"teacher process exited with code {self._proc.exitcode}")
        return self._conn.recv()

    def __call__(self, x: torch.Tensor) -> tuple[float, torch.Tensor]:
        return self.submit(x).result()

    def close(self) -> None:
        if not self._proc.is_alive():
            return
        try:
            self._conn.send(None)
        finally:
            self._proc.join(timeout=30)


class MaceTeacherSpec:
    """Picklable recipe for the production teacher inside a ``ProcessTeacher``."""

    def __init__(self, atoms, cell, skin: float = 1.0, builder: str = "alchemi",
                 device: str = "cuda", teacher: str = "mp-medium", edge_forces: bool = False, tight: bool = False):
        self.atoms, self.cell, self.skin, self.builder, self.device, self.teacher = (
            atoms, np.asarray(cell), skin, builder, device, teacher)
        self.edge_forces, self.tight = edge_forces, tight

    def __call__(self) -> TeacherCall:
        torch.set_num_threads(1)
        return mace_teacher_builder(self.atoms, self.cell, self.device, self.skin,
                                    self.builder, getattr(self, "teacher", "mp-medium"),
                                    getattr(self, "edge_forces", False), getattr(self, "tight", False))()

