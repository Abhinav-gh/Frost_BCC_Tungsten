"""MD backend for the runtime: a student every step, the teacher's residual refreshed every k steps.

The operator is the teacher's force, gradient-consumed, so the runtime holds only its output in
residual form, R = F_T - F_S (``reuse.check_hold(GRADIENT, "output")``). Modes:

- "sync": boundary r-RESPA. Half kick with R_n, k student steps, teacher at the new boundary,
  half kick with R_{n+1} (Tuckerman, Berne and Martyna 1992).
- "lag": the teacher on a second GPU, one interval behind (``frost.async_teacher.LaggedResidual``, C2): at each
  boundary the positions go to the teacher process and the newest finished residual, R(x_{n-1}), closes the kick.
  Wall time per interval falls from k c_S + c_T to max(k c_S, c_T). A lag is negative friction, so the mode belongs
  to the thermostatted contract; a rollback cannot recall a call in flight, so the driver latches refusals instead.
- "spec": speculative refresh. The student looks ahead over the interval with R held (same
  thermostat noise), the teacher is evaluated at that predicted boundary, its residual is
  transported to the true one, and the interval is integrated with the residual ramped from
  R_n to the transported R_{n+1} (``Integrator.respa_interp``): no impulse.

``build`` sets up a run from an E1 trajectory frame exactly as ``validate.py`` does (the same
teacher, CUDA graphs, lists and student), so the two harnesses can be checked against each other.

Under a thermostat whose integrator tracks heat (``Integrator.track_heat``), every energy the backend reports is the
effective energy E_total - Q, conserved by exact dynamics, so the drift clause certifies NVT runs as it does NVE ones.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from ase import units

from frost.integrators import Integrator, Langevin, MDState, NoseHooverChain
from frost.locality import staleness_stats
from frost.runtime.executor import Interval
from frost.runtime.reuse import ReuseClass, check_hold

ROOT = Path(__file__).resolve().parents[3]
TRAJ = ROOT / "data" / "trajectories"


class MDBackend:
    reuse = ReuseClass.GRADIENT

    def __init__(self, state: MDState, integ: Integrator, student_force, teacher_force, dt_fs: float,
                 n_atoms: int, on_step=None, lagged=None, track_temperature: bool = False, structure=None,
                 barostat=None, local=None):
        # local: {"regions": n, "z": z, "cell": 3 x 3} measures the staleness region by region as well
        # (frost.locality), for the local magnitude clause; the controller then grows on the binding of the two
        self.local, self._local_last = local, None
        check_hold(self.reuse, "output")
        # the pair-distance histogram of each interval (frost.runtime.StructureClause): ``structure`` is a dict with
        # the student's Verlet list ("list"), the range ("r_max", A), "bins" and the sampling period "every" (steps);
        # summed on the device and read once per interval into extra["pair_hist"]
        self.structure = structure
        self._pair_sum, self._pair_n = None, 0
        # the kinetic temperature averaged over every inner step of an interval (all phases of the kicks, as the
        # fidelity gate samples it), summed on the device and read once per interval: extra["T_mean"]
        self.track_temperature = track_temperature
        self._ke_sum, self._ke_n = None, 0
        self.state, self.integ = state, integ
        self.student_force, self.teacher_force = student_force, teacher_force
        self.dt_fs, self.n = dt_fs, n_atoms
        self.on_step = on_step
        self.step = 0
        self.cursor = 0
        e_t, f_t, self.state.x = teacher_force(self.state.x)
        self.e_s, self.f_s, self.state.x = student_force(self.state.x)
        self.residual = f_t - self.f_s
        self.e0 = float(e_t) + state.kinetic_energy()
        self.lagged = lagged  # a LaggedResidual for mode "lag", started on this residual
        if lagged is not None:
            lagged.residual, lagged.history, lagged.f_at_residual = self.residual, [self.residual], self.f_s.clone()
        self.teacher_calls = 1
        self.student_steps = 0
        self.transport = []
        gen = integ.thermostat.gen if integ.thermostat is not None else None
        self.gen = gen
        # NPT (further_expansions_beyond.md 6.1): ``barostat`` is a dict with "baro" (StochasticCellRescaling), "virial"
        # (the last teacher call's virial), "set_cell" (lists and graphs follow a new cell) and "cell". The cell moves at
        # the start of each interval from the pressure of the previous boundary, where the teacher's virial is exact, so
        # nothing is held; the student force is recomputed at the scaled positions (one student call per interval).
        self.baro = barostat
        if barostat is not None:
            self.cell = np.array(barostat["cell"], dtype=np.float64)
            self.stress_trace = []  # (t fs, L_x, L_y, L_z, P_xx, P_yy, P_zz) at each per-axis move
            self.pressure = self._pressure()
            self.volume_trace = [(0.0, self.volume())]

    # ---- constant pressure
    def volume(self) -> float:
        return float(abs(np.linalg.det(self.cell)))

    def _uniaxial(self) -> bool:
        return getattr(self.baro["baro"], "axes", "xyz") != "xyz"

    def _kinetic_axes(self) -> torch.Tensor:
        s = self.state
        return 0.5 * (s.masses * s.v ** 2).sum(0)  # (K_xx, K_yy, K_zz)

    def _pressure(self):
        if self._uniaxial():  # per-axis barostats (NP_zAT, uniaxial stress): normal stresses from the virial tensor
            k_ax = self._kinetic_axes().double().cpu().numpy()
            w = self.baro["virial_tensor"]().detach().double().cpu().numpy()
            # the full diagonal at this boundary, for a stress-strain record (mechanics: tension along a is -P_aa)
            self.stress_diag = self.baro["baro"].stress_diagonal(k_ax, w, self.volume())
            return self.baro["baro"].pressure_axes(k_ax, w, self.volume())
        return self.baro["baro"].pressure(self.state.kinetic_energy(), float(self.baro["virial"]()), self.volume())

    def _barostat_move(self, steps: int) -> None:
        from ase import units

        if self._uniaxial():
            self._barostat_move_axes(steps * self.dt_fs * units.fs)
            return
        mu = self.baro["baro"].scale(self.volume(), self.pressure, steps * self.dt_fs * units.fs)
        s = self.state
        k0 = s.kinetic_energy()
        s.x = s.x * mu
        s.v = s.v / mu
        self.cell = self.cell * mu
        self.baro["set_cell"](self.cell)
        e_after, self.f_s, s.x = self.student_force(s.x)
        if self.integ.heat is not None:
            # the move's energy exchange, booked as heat so E - Q measures integration error only (version 2): kinetic
            # K (mu^-2 - 1) exactly; potential from the student's own energies before and after the move, exact to all
            # orders for the student. The first-order form -W ln mu misses a positive second-order term ~ 9 B V (ln mu)^2
            # / 2 that never cancels (8% of a move at mu = 1.001 on Lennard-Jones; ~0.02 meV/atom per move on Cu 500),
            # and the energy clauses would read it as shadow work. The teacher differs only by its residual's share,
            # -(W_T - W_S) ln mu, first order and of random sign.
            self.integ.heat += k0 * (mu ** -2 - 1.0) + float(e_after) - float(self.e_s)
        self.e_s = e_after
        self.volume_trace.append((self.time(), self.volume()))

    def _barostat_move_axes(self, duration: float) -> None:
        """The move with per-axis factors (NP_zAT, uniaxial stress, a deformed axis), booked as heat as the isotropic
        move is."""
        if getattr(self, "stress_diag", None) is not None:
            self.stress_trace.append((self.time(), *np.diag(self.cell).tolist(), *self.stress_diag.tolist()))
        mu = self.baro["baro"].scale_vector(self.volume(), self.pressure, duration)
        s = self.state
        mu_t = torch.as_tensor(mu, dtype=s.x.dtype, device=s.x.device)
        k_axes = self._kinetic_axes()
        s.x = s.x * mu_t
        s.v = s.v / mu_t
        self.cell = self.cell * mu[None, :]  # rows are lattice vectors: each component scales with its axis
        self.baro["set_cell"](self.cell)
        e_after, self.f_s, s.x = self.student_force(s.x)
        if self.integ.heat is not None:
            self.integ.heat += float((k_axes * (mu_t ** -2 - 1.0)).sum()) + float(e_after) - float(self.e_s)
        self.e_s = e_after
        self.volume_trace.append((self.time(), self.volume()))

    # ---- runtime interface
    def time(self) -> float:
        return self.step * self.dt_fs

    def steps_left(self, horizon_fs: float) -> int:
        return max(0, round(horizon_fs / self.dt_fs) - self.step)

    def checkpoint(self):
        heat = self.integ.heat.clone() if self.integ.heat is not None else None
        thermo = self.integ.thermostat
        chain = thermo.state() if thermo is not None and hasattr(thermo, "state") else None  # a deterministic chain
        baro = ((self.cell.copy(), self.baro["baro"].gen.get_state(), self.pressure, self.e_s)
                if self.baro is not None else None)
        return (self.state.x.clone(), self.state.v.clone(), self.f_s.clone(), self.residual.clone(),
                self.step, self.gen.get_state() if self.gen is not None else None, heat, chain, baro)

    def restore(self, ck) -> None:
        x, v, f_s, r, step, g, heat, chain, baro = ck
        if baro is not None:
            self.cell, self.pressure, self.e_s = baro[0].copy(), baro[2], baro[3]
            self.baro["baro"].gen.set_state(baro[1])
            self.baro["set_cell"](self.cell)
        self.state.x, self.state.v, self.f_s, self.residual, self.step = x.clone(), v.clone(), f_s.clone(), r.clone(), step
        if baro is not None and getattr(self, "stress_trace", None) is not None:
            # an undone interval's stress records go with it, and the cached stress is re-read at the next boundary
            self.stress_trace = [row for row in self.stress_trace if row[0] <= self.time()]
            self.stress_diag = None
        if g is not None:
            self.gen.set_state(g)
        if heat is not None:
            self.integ.heat = heat.clone()
        if chain is not None:
            self.integ.thermostat.restore(chain)

    def _effective(self, total: float) -> float:
        """Total energy, less the heat the thermostat has added when the integrator tracks it."""
        return total - float(self.integ.heat) if self.integ.heat is not None else total

    def finish(self) -> None:
        torch.cuda.synchronize()

    def _staleness(self, r_new, r_old) -> tuple[float, float]:
        m = self.state.masses
        d = ((r_new - r_old) / m).norm(dim=1)
        if self.local is not None:
            x = self.state.x
            cell = torch.as_tensor(self.cell if self.baro is not None else self.local["cell"], dtype=x.dtype,
                                   device=x.device)
            self._local_last = staleness_stats((r_new - r_old) / m * 1e3, x, cell, int(self.local["regions"]),
                                               float(self.local.get("z", 4.0)))["local"]
        return float(d.pow(2).mean().sqrt()) * 1e3 / np.sqrt(3.0), float(d.max()) * 1e3

    def interval(self, k: int, mode: str) -> Interval:
        s, integ = self.state, self.integ
        self.cursor = self.step  # the simulated step inside this interval, for callbacks
        self._ke_sum, self._ke_n = None, 0  # a rolled-back attempt's samples are discarded with it
        self._pair_sum, self._pair_n = None, 0
        self._local_last = None
        iv = self._interval(k, mode)
        if self.local is not None and self._local_last is not None:
            iv.extra["local"] = self._local_last
            iv.extra["change"] = max(iv.staleness, self._local_last) * float(np.sqrt(3.0))
        if self.track_temperature and self._ke_n:
            iv.extra["T_mean"] = 2.0 * float(self._ke_sum) / (self._ke_n * 3 * self.n * units.kB)
        if self.structure is not None and self._pair_n:
            iv.extra["pair_hist"] = (self._pair_sum / self._pair_n).cpu().numpy()
        if self.baro is not None:  # the energy clauses regress the volume out of E - Q under a barostat (clauses.py)
            iv.extra["volume"] = self.volume()
        return iv

    def _interval(self, k: int, mode: str) -> Interval:
        s, integ = self.state, self.integ
        if mode == "sync":
            if self.baro is not None:
                self._barostat_move(k)
            integ.kick(self.residual, 0.5 * k * integ.dt)
            for _ in range(k):
                self.e_s, self.f_s = integ.inner_step(self.student_force, self.f_s)
                self._sampled()
            e_t, f_t, s.x = self.teacher_force(s.x)
            r_new = f_t - self.f_s
            integ.kick(r_new, 0.5 * k * integ.dt)
            energy = float(e_t)
            if self.baro is not None:
                self.pressure = self._pressure()
        elif mode == "lag":
            integ.kick(self.residual, 0.5 * k * integ.dt)
            for _ in range(k):
                _, self.f_s = integ.inner_step(self.student_force, self.f_s)
                self._sampled()
            heat = float(integ.heat) if integ.heat is not None else 0.0
            r_kick, finished = self.lagged.boundary(s.x, self.f_s, s.kinetic_energy() - heat, self.step + k)
            integ.kick(r_kick, 0.5 * k * integ.dt)
            self.student_steps += k
            self.teacher_calls += 1
            self.step += k
            if finished is not None:
                stale, worst = self._staleness(finished["residual"], finished["previous"])
                total = float(finished["energy"]) + finished["kinetic"]  # that boundary's effective energy
            else:
                stale, worst, total = 0.0, 0.0, self.e0
            self.residual = r_kick
            finite = bool(np.isfinite(total)) and bool(torch.isfinite(s.x).all())
            return Interval(t=self.time(), k=k, staleness=stale, finite=finite,
                            energy=(total - self.e0) / self.n * 1e3,
                            extra={"worst": worst, "change": stale * float(np.sqrt(3.0)), "lagged": True})
        elif mode == "spec":
            saved = (s.x.clone(), s.v.clone(), self.f_s.clone(), self.gen.get_state() if self.gen is not None else None)
            e_s_hat, f_s_hat = integ.respa_interp(self.student_force, self.f_s, self.residual, self.residual, k)
            x_hat = s.x.clone()
            e_hat, f_t_hat, _ = self.teacher_force(x_hat)
            r_new = f_t_hat - f_s_hat
            s.x, s.v, self.f_s = saved[0], saved[1], saved[2]
            if saved[3] is not None:
                self.gen.set_state(saved[3])
            e_s, self.f_s = integ.respa_interp(self.student_force, self.f_s, self.residual, r_new, k,
                                               on_step=self._sampled)
            self.student_steps += k  # the look-ahead's
            energy = float(e_hat) + float(e_s) - float(e_s_hat)  # transported, as the force is
        elif mode == "mid":
            # the residual once, at the interval's midpoint (Integrator.respa_midpoint): half of every boundary
            # kick sat k dt / 2 from where its residual was measured. The teacher energy is known at the midpoint,
            # so the energy sample is taken there, against the kinetic energy at those positions (validate.py does
            # the same); the held residual and its staleness are midpoint to midpoint.
            box = {}

            def slow_at(x, f_student):
                e_t, f_t, _ = self.teacher_force(x)
                box["total"] = float(e_t) + s.kinetic_energy()
                r = f_t - f_student
                return r, r

            _, self.f_s, r_new = integ.respa_midpoint(self.student_force, self.f_s, slow_at, k, on_step=self._sampled)
            energy = None
        else:
            raise ValueError(f"mode {mode!r}")
        self.student_steps += k
        self.teacher_calls += 1
        stale, worst = self._staleness(r_new, self.residual)
        self.residual = r_new
        self.step += k
        total = box["total"] if mode == "mid" else self._effective(energy + s.kinetic_energy())
        finite = bool(np.isfinite(total)) and bool(torch.isfinite(s.x).all())
        return Interval(t=self.time(), k=k, staleness=stale, finite=finite,
                        energy=(total - self.e0) / self.n * 1e3,
                        extra={"worst": worst, "change": stale * float(np.sqrt(3.0))})

    def baseline_interval(self, n: int, probe_k: int, mode: str = "sync") -> Interval:
        """Never lose: n steps of the reference dynamics (the teacher every step, the student idle).

        With ``probe_k``, the student is evaluated at two points probe_k steps apart at the end, which gives the
        residual change a sync interval of probe_k would have held, for the clauses to judge without running it.
        The chunk ends with the student force and residual at the final positions, so a sync interval can follow.
        """
        s, integ = self.state, self.integ
        self.cursor = self.step
        if self.baro is not None and mode == "sync":
            self._barostat_move(n)
            _, f_t0, s.x = self.teacher_force(s.x)  # the residual at the scaled cell: one extra call per chunk
            self.residual = f_t0 - self.f_s
            self.teacher_calls += 1
        if mode == "sync":
            f_t = self.f_s + self.residual  # the teacher force at x, split by one call (trap 8)
        else:
            _, f_t, s.x = self.teacher_force(s.x)
            self.teacher_calls += 1
        r0 = None
        e_t = None
        for i in range(n):
            e_t, f_t = integ.inner_step(self.teacher_force, f_t)
            self._sampled()
            if probe_k and i == n - 1 - probe_k:
                _, f_s0, s.x = self.student_force(s.x)
                r0 = f_t - f_s0
        self.e_s, self.f_s, s.x = self.student_force(s.x)
        r_new = f_t - self.f_s
        stale, worst = self._staleness(r_new, r0) if r0 is not None else (0.0, 0.0)
        self.residual = r_new
        self.teacher_calls += n
        self.student_steps += 2 if r0 is not None else 1
        self.step += n
        if self.baro is not None:
            self.pressure = self._pressure()
        total = self._effective(float(e_t) + s.kinetic_energy())
        finite = bool(np.isfinite(total)) and bool(torch.isfinite(s.x).all())
        return Interval(t=self.time(), k=1, staleness=stale, finite=finite, energy=(total - self.e0) / self.n * 1e3,
                        extra={"worst": worst, "change": stale * float(np.sqrt(3.0)), "probed": r0 is not None,
                               "baseline": True, "steps": n})

    def _sampled(self):
        self.cursor += 1
        if self.structure is not None and self.cursor % self.structure["every"] == 0:
            lst, x = self.structure["list"], self.state.x
            d = (x.index_select(0, lst.j) - x.index_select(0, lst.i) + lst.shifts).norm(dim=-1)
            h = torch.histc(d, bins=self.structure["bins"], min=0.0, max=self.structure["r_max"])  # outside: ignored
            h = h / h.sum().clamp(min=1.0)
            self._pair_sum = h if self._pair_sum is None else self._pair_sum + h
            self._pair_n += 1
        if self.track_temperature:
            ke = 0.5 * (self.state.masses * self.state.v * self.state.v).sum()  # on the device: no sync per step
            self._ke_sum = ke if self._ke_sum is None else self._ke_sum + ke
            self._ke_n += 1
        if self.on_step is not None:
            self.on_step(self)


from frost.neighbours import tight_edges  # noqa: E402  (re-exported: the function moved to frost.neighbours)


def build(run: str, student_path: str, device: str = "cuda", start: int = 9000, dt_fs: float = 1.0,
          ensemble: str = "nve", friction: float = 0.01, seed: int = 0, skin: float = 1.0,
          thermostat: str = "langevin", nhc_tau_fs: float = 100.0, npt: bool = False,
          student_skin: float = 0.3, neighbours: str = "alchemi", derived_lists: bool = True,
          lagged_lists: bool = False, compile_student: str = "", tile: int = 1, hmr: float = 0.0,
          edge_forces: bool = False, teacher_ext: str = "", teacher_path: str = "graph", tight_teacher: bool = False):
    """The teacher, student and lists of a run, as ``validate.py`` builds them.

    Returns a dict with the initial state, masses, both force callables and a factory for an
    ``Integrator`` (so a baseline teacher arm and a Frost arm start from the same state).

    ``teacher_path``: "graph" (``GraphedTeacher``, with ``edge_forces`` if asked), "eager" (``MaceTeacher``), or
    "auto", which times both on the start frame and keeps the faster. The baseline and every Frost arm share the
    chosen teacher, so the baseline is always the strongest stack measured (trap 17): the graphed call wins at 500
    atoms (15 against 38 ms on Cu) and was measured slower than eager at 4,000 (118-371 against 92 ms).
    """
    import h5py

    from frost.neighbours import VerletList
    from frost.student import GraphedStudent
    from frost.students import load_student
    from frost.teacher import frame_atoms
    from frost.teacher_fast import GraphedTeacher
    from frost.teachers import load_calc, repartition_hydrogen, teacher_of

    temperature = float(run.split("_T")[1])
    with h5py.File(TRAJ / f"{run}.h5", "r") as fh:
        numbers = np.asarray(fh.attrs["numbers"], dtype=np.int64)
        cell = np.asarray(fh.attrs["cell"], dtype=np.float64)
        positions = np.asarray(fh["positions"][start], dtype=np.float64)
        velocities = np.asarray(fh["velocities"][start], dtype=np.float64)
        teacher_name = teacher_of(fh.attrs)
    if tile > 1:
        # a supercell of the equilibrated frame, for timing at scale (tiled velocities are correlated, so this is
        # not a fidelity test)
        shifts = np.array([[a, b, c] for a in range(tile) for b in range(tile) for c in range(tile)], dtype=np.float64)
        offsets = shifts @ cell
        positions = np.concatenate([positions + o for o in offsets])
        velocities = np.concatenate([velocities] * len(offsets))
        numbers = np.concatenate([numbers] * len(offsets))
        cell = cell * tile
    atoms = frame_atoms(numbers, positions, cell)
    n = len(atoms)
    mass_np = repartition_hydrogen(numbers, atoms.get_masses(), hmr)
    if hmr > 0:
        velocities = velocities * np.sqrt(atoms.get_masses() / mass_np)[:, None]
    masses = torch.as_tensor(mass_np, dtype=torch.float64, device=device).unsqueeze(1)
    x0 = torch.as_tensor(atoms.get_positions(), dtype=torch.float64, device=device)
    v0 = torch.as_tensor(velocities, dtype=torch.float64, device=device)

    if teacher_ext:
        # a teacher served from its own environment (frost.external_teacher): it keeps its own neighbour list, so the
        # student's list is built on its own (no derived list)
        from frost.external_teacher import ExternalTeacher

        ext_teacher = ExternalTeacher(teacher_ext, numbers, cell, device)
        teacher, t_list, derived_lists = None, None, False
    else:
        from frost.teacher import MaceTeacher

        calc = load_calc(teacher_name, device, "float32", enable_cueq=True)
        candidates = {"graph": lambda: GraphedTeacher(calc, atoms, device, edge_forces=edge_forces, virial=npt),
                      "eager": lambda: MaceTeacher(calc, atoms, device)}
        if teacher_path not in (*candidates, "auto"):
            raise ValueError(f"teacher_path {teacher_path!r}")
        timings = {}
        for name in (candidates if teacher_path == "auto" else (teacher_path,)):
            teacher = candidates[name]()
            t_list = VerletList(cell, teacher.r_max, skin, device, wrap_positions=False, builder=neighbours)
            t_list.update(x0, force=True)
            # with tight_teacher the graph's edge buffers are sized for the filtered list (they only ever grow, so a first
            # call on the skin list would leave every later tight call padded back to the skin's size)
            teacher.set_edges(*(tight_edges(t_list, x0, float(teacher.r_max) ** 2) if tight_teacher
                                else (t_list.i, t_list.j, t_list.shifts, t_list.unit_shifts)))
            if teacher_path == "auto":
                import time

                for _ in range(3):
                    teacher(x0)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(10):
                    teacher(x0)
                torch.cuda.synchronize()
                timings[name] = (1e3 * (time.perf_counter() - t0) / 10, teacher, t_list)
        if timings:
            teacher_path = min(timings, key=lambda n: timings[n][0])
            _, teacher, t_list = timings[teacher_path]
            timings = {k: v[0] for k, v in timings.items()}  # the slower candidate is released here
            torch.cuda.empty_cache()
    student = load_student(student_path, device)
    if student.config.get("kind", "frost") == "mace" and student.config.get("backend", "e3nn") != "cueq":
        student = student.converted("cueq")  # mace's own converter (public API)
    types_t = student.types(torch.as_tensor(numbers, device=device))
    s_list = VerletList(cell, float(student.config["cutoff"]), student_skin, device, wrap_positions=False,
                        builder=neighbours, outer=t_list if derived_lists else None,
                        lag_margin=0.05 if lagged_lists else None)
    s_list.update(x0, force=True)
    cap = int(1.3 * s_list.i.numel()) + 64
    graphed = GraphedStudent(student, types_t, n, capacity=cap)
    graphed.set_edges(s_list.i, s_list.j, s_list.shifts)
    if compile_student:
        from frost.student_fast import CompiledGraphedStudent

        f_eager = graphed(x0)[1].detach().double().clone()
        graphed = CompiledGraphedStudent(student, types_t, n, capacity=cap, mode=compile_student)
        graphed.set_edges(s_list.i, s_list.j, s_list.shifts)
        diff = float((graphed(x0)[1].detach().double() - f_eager).abs().max())
        if diff > 1e-3:
            raise RuntimeError(f"compiled student differs by {diff:.2e} eV/A")
    version = {"teacher": t_list.rebuilds if t_list is not None else 0, "student": s_list.rebuilds}
    r_cut2 = float(teacher.r_max) ** 2 if teacher is not None else 0.0

    def teacher_force(x):
        if teacher_ext:
            return ext_teacher(x)
        t_list.update(x)
        if tight_teacher:
            # the skin list filtered to r_max before every call (2026-11-16): MACE evaluates every edge it is given and
            # zeroes the skin's pairs only through its envelope, so they cost compute for nothing (teacher_list_cost.py:
            # a call on the tight list was 31-35% cheaper on five systems, with identical forces). The list itself is still
            # rebuilt only when an atom has moved half the skin; the filter is a distance check and a compaction.
            teacher.set_edges(*tight_edges(t_list, x, r_cut2))
        elif t_list.rebuilds != version["teacher"]:
            version["teacher"] = t_list.rebuilds
            teacher.set_edges(t_list.i, t_list.j, t_list.shifts, t_list.unit_shifts)
        e, f = teacher(x)
        return e, f.double(), x

    def student_force(x):
        s_list.update(x)
        if s_list.rebuilds != version["student"]:
            version["student"] = s_list.rebuilds
            graphed.set_edges(s_list.i, s_list.j, s_list.shifts)
        e, f = graphed(x)
        return e.detach().sum(), f.detach().double(), x

    def teacher_virial():
        v = getattr(teacher, "static_virial", None)
        if v is None:
            raise RuntimeError("NPT needs the teacher's virial: the in-process graphed teacher with --edge-forces")
        return v

    def teacher_virial_tensor():
        v = getattr(teacher, "static_virial_tensor", None)
        if v is None:
            raise RuntimeError("NP_zAT needs the teacher's virial tensor: the in-process graphed teacher with --edge-forces")
        return v

    def set_cell(new_cell) -> None:
        for lst in (t_list, s_list):
            if lst is not None:
                lst.set_cell(new_cell)
        if t_list is not None and not tight_teacher:
            # (with tight_teacher the next teacher call filters the list at the new cell; writing the whole skin list here
            # grew the graph's buffers back to the skin's size, so under a barostat every later tight call was padded to
            # it: Cu NPT's tight baseline read x1.104 of the skin one, 2026-11-16)
            teacher.set_edges(t_list.i, t_list.j, t_list.shifts, t_list.unit_shifts)
        graphed.set_edges(s_list.i, s_list.j, s_list.shifts)

    def integrator(state):
        thermo = None
        if ensemble in ("nvt", "npt"):
            thermo = (NoseHooverChain(dt_fs, nhc_tau_fs, temperature, masses) if thermostat == "nhc"
                      else Langevin(dt_fs, friction, temperature, masses, seed))
        return Integrator(state, dt_fs, thermo)

    return {"n": n, "x0": x0, "v0": v0, "masses": masses, "teacher_force": teacher_force,
            "student_force": student_force, "integrator": integrator, "t_list": t_list, "s_list": s_list,
            "student_config": student.config, "temperature": temperature,
            "teacher_virial": teacher_virial, "teacher_virial_tensor": teacher_virial_tensor, "set_cell": set_cell, "cell0": np.array(cell, dtype=np.float64),
            # for a teacher served from a second GPU (mode "lag"): what its process rebuilds the same teacher from
            "atoms": atoms, "cell": cell, "teacher_name": teacher_name, "edge_forces": edge_forces,
            "teacher_path": teacher_path if not teacher_ext else "external",
            "teacher_path_ms": timings if not teacher_ext else {}}
