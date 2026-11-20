"""E6 (gate G3): does the multirate loop change the physics?

Speed means nothing if the trajectory is wrong. This runs two trajectories from
the same E1 frame with the same kernels, seeds and time step:

  teacher   plain teacher MD, the reference;
  frost     the same dynamics with a frozen student every step and the teacher
            residual refreshed every k steps, exactly as E5b runs it.

and compares four observables:

- **Energy drift** (NVE only): total energy per atom, fitted in meV/atom/ps. The
  teacher energy of the Frost trajectory is read at the outer steps, where it is
  computed anyway.
- **Vibrational density of states**, from the velocity autocorrelation of the
  whole cell. Reported as a spectrum and as its first moment.
- **Radial distribution functions**, per species pair.
- **Diffusion**, from the slope of the mean squared displacement (meaningful for
  water and Li3PO4; the solids should stay flat).

Nothing here is thermostatted unless --ensemble nvt, because a thermostat hides
integration error. The Frost side reports its realised staleness, so a fidelity
result is always paired with the certificate that produced it.

Output: data/results/e6/fidelity_<run>_<ensemble>[_<tag>].json

Usage (guarded launcher; GPU)
  scripts/safe_run.sh e6_silica .venvs/frost-mace/bin/python \\
      experiments/e6_dynamics/validate.py --run silica1152_T300 \\
      --student data/models/students/silica1152_T300_m45c16.pt --k 8 --fs 10000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from frost.observables import as_list, msd, rdf, vdos
from frost.teachers import load_calc, repartition_hydrogen, teacher_of

ROOT = Path(__file__).resolve().parents[2]
TRAJ = ROOT / "data" / "trajectories"
OUT_DIR = ROOT / "data" / "results" / "e6"
def main() -> int:
    import h5py
    from frost.controller import IntervalController
    from frost.integrators import BandGLE, Integrator, Langevin, MDState
    from frost.neighbours import VerletList
    from frost.student import GraphedStudent
    from frost.student_mace import BACKENDS
    from frost.students import load_student
    from frost.teacher import frame_atoms, mace_forces
    from frost.teacher_fast import GraphedTeacher

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="silica1152_T300")
    ap.add_argument("--student", required=True, help="saved student weights")
    ap.add_argument("--student-plugin", default="",
                    help="a .py file whose make_student(path, device) builds the student (see the code)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--start", type=int, default=9000)
    ap.add_argument("--fs", type=int, default=10000)
    ap.add_argument("--dt", type=float, default=1.0, help="inner step, fs")
    ap.add_argument("--k", type=int, default=0, help="fixed outer interval; 0 lets the controller pick")
    ap.add_argument("--k-max", type=int, default=16)
    ap.add_argument("--derived-lists", action="store_true",
                    help="the student's list is refreshed by filtering the teacher's (a Verlet list "
                    "of a Verlet list) instead of a cell-list build: 3.9 ms per rebuild on liquid "
                    "Cu, every ~5 steps of the student's skin. Exact while cutoff + skin <= r_max")
    ap.add_argument("--lagged-lists", action="store_true",
                    help="lists consulted every step decide from the previous step's displacement "
                    "(non-blocking copy) with --lag-margin, so the host never stalls the GPU: the "
                    "student list in the frost arm, the teacher list in the teacher arm")
    ap.add_argument("--lag-margin", type=float, default=0.05, help="A")
    ap.add_argument("--device-samples", action="store_true",
                    help="keep velocity and position samples on the GPU until the end, instead of a "
                    "blocking device-to-host copy every step. Both arms")
    ap.add_argument("--graph-loop", action="store_true",
                    help="run each arm's inner steps as one CUDA graph per interval "
                    "(frost.graph_loop): no host sync or Python per step; neighbour lists "
                    "refresh at boundaries and an interval whose atoms moved past half the skin "
                    "is redone eagerly. Both arms, so the baseline stays the strongest stack")
    ap.add_argument("--predictive-lists", action="store_true",
                    help="with --graph-loop: rebuild the student list before a chunk when an atom could leave the "
                         "skin within it (v_max k dt), instead of redoing the chunk after it did")
    ap.add_argument("--student-chunk", type=int, default=0,
                    help="with --graph-loop: student steps per graph replay (0 = the whole "
                    "interval). Lists refresh between chunks, so the production skin survives "
                    "a liquid's displacement: one host sync per chunk instead of per step")
    ap.add_argument("--teacher-chunk", type=int, default=8,
                    help="with --graph-loop: teacher-arm steps per graph replay")
    ap.add_argument("--rollback", action="store_true",
                    help="optimistic outer intervals (new_directions.md S5): checkpoint each "
                    "boundary; an interval whose residual change exceeds --rollback-epsilon, or "
                    "whose state is non-finite, is undone and redone as two halves, recursively "
                    "down to k = 1, with the same thermostat noise. Async runs pause the "
                    "pipeline, redo the last two intervals synchronously, and restart it")
    ap.add_argument("--rollback-epsilon", type=float, default=None,
                    help="staleness, in the report's units, that triggers a rollback "
                    "(default: --epsilon, the certificate's clause)")
    ap.add_argument("--rollback-atom", type=float, default=None,
                    help="also roll back when one atom's residual change exceeds this, "
                    "meV/(A amu); off by default, its distribution is reported either way")
    ap.add_argument("--teacher-ref", default=None,
                    help="with --arms frost: a fidelity JSON whose teacher arm this run reuses. "
                    "Its run, start, seed, dt, ensemble, friction and fs must match, so the "
                    "teacher trajectory is the one this run would have integrated")
    ap.add_argument("--arms", choices=("both", "teacher", "frost"), default="both",
                    help="'teacher' runs the reference arm alone. Every speedup in this project "
                         "compares against a teacher at --dt 1.0, which was never justified: the "
                         "fair baseline is the teacher at the largest step that holds its own "
                         "energy, and on a system whose fastest period is 113.7 fs that may be "
                         "several femtoseconds (plan section 1.15)")
    ap.add_argument("--respa", choices=("boundary", "midpoint", "spread", "spec", "student", "switch"), default="boundary",
                    help="where the residual is evaluated: at the interval's boundaries (the "
                         "r-RESPA of weeks 1-3), once at its midpoint, spread over every "
                         "inner step instead of applied as an impulse, or 'spec': speculative "
                         "refresh, the teacher evaluated at a predicted boundary state (a student "
                         "look-ahead with the residual held), its residual transported to the true "
                         "state, and the interval integrated with the residual ramped linearly "
                         "from its start value to that end value (Integrator.respa_interp)")
    ap.add_argument("--spec-predict", choices=("hold", "extrap"), default="hold",
                    help="with --respa spec: the look-ahead holds the residual, or extrapolates it "
                         "linearly from the last two refreshes")
    ap.add_argument("--spec-diagnose", action="store_true",
                    help="with --respa spec: also call the teacher at the true boundary, for the "
                         "energy record and the transport error |R(x_hat) - R(x)|. Production "
                         "transports the energy as it does the force, so this call is a diagnostic "
                         "and makes the run's timing meaningless (accuracy runs only)")
    ap.add_argument("--k-jitter", type=float, default=0.0,
                    help="draw each outer step from k * [1 - jitter, 1 + jitter]. r-RESPA's "
                         "resonance is coherent, so breaking the fixed ratio to the fastest "
                         "period should cut drift at the same mean interval (plan section 1.14)")
    ap.add_argument("--units", choices=("acceleration", "force"), default="acceleration",
                    help="certificate units: meV/(A amu), or meV/A as in weeks 1-2")
    ap.add_argument("--epsilon", type=float, default=None,
                    help="staleness budget; default 2.0 for acceleration, 20.0 for force")
    ap.add_argument("--fastest-period-fs", type=float, default=0.0,
                    help="fastest vibrational period; caps the outer step at 0.4 of it")
    ap.add_argument("--ensemble", choices=("nve", "nvt"), default="nve")
    ap.add_argument("--friction", type=float, default=0.01, help="per fs, nvt only")
    ap.add_argument("--thermostat", choices=("langevin", "band-gle"), default="langevin",
                    help="nvt only: uniform Langevin friction, or frost.integrators.BandGLE, a "
                    "canonical friction peaked on one band (new_directions.md R3); both arms "
                    "use the same one")
    ap.add_argument("--gle-center-thz", type=float, default=31.0)
    ap.add_argument("--gle-width-thz", type=float, default=4.0)
    ap.add_argument("--gle-peak", type=float, default=0.01, help="per fs; keep <= Gamma / 2")
    ap.add_argument("--gle-base", type=float, default=0.001, help="per fs, everywhere")
    ap.add_argument("--sample-every", type=int, default=2, help="velocity and position sampling, fs")
    ap.add_argument("--rdf-every", type=int, default=200, help="frames kept for the RDF, fs")
    ap.add_argument("--skin", type=float, default=1.0)
    ap.add_argument("--student-skin", type=float, default=0.3)
    ap.add_argument("--neighbours", default="alchemi")
    ap.add_argument("--student-backend", choices=BACKENDS, default="cueq")
    ap.add_argument("--student-compile", default="", choices=("", "default", "max-autotune-no-cudagraphs"),
                    help="compile the descriptor student with TorchInductor inside its CUDA graph "
                         "(frost.student_fast.CompiledGraphedStudent); checked against the eager call")
    ap.add_argument("--async-teacher", action="store_true",
                    help="run the teacher on --teacher-device one outer step behind, holding "
                    "the lagged residual (new_directions.md C2); needs a fixed --k, boundary "
                    "RESPA, and belongs to the NVT contract")
    ap.add_argument("--teacher-device", default="cuda:1")
    ap.add_argument("--lag-compensate", action="store_true",
                    help="with --async-teacher: kick with R(x_{n-1}) + mu (F_S(x_n) - F_S(x_{n-1})), mu the "
                         "curvature mismatch measured online per species (frost.async_teacher.LaggedResidual)")
    ap.add_argument("--lag", type=int, default=1,
                    help="with --async-teacher: kick with the residual this many outer steps "
                    "old. One device finishes each call one step behind; --lag 2 holds results "
                    "a further step, reproducing a two-device pipeline's staleness (its fidelity, "
                    "not its speed)")
    ap.add_argument("--bond-stiffness", default="",
                    help="online harmonic stiffness correction along bonds Za-Zb (e.g. 1-8 for O-H), fitted from the "
                         "residual changes at each refresh (frost.bond_correction); sync boundary mode only")
    ap.add_argument("--bond-memory", type=float, default=0.7)
    ap.add_argument("--edge-forces", action="store_true",
                    help="the teacher's forces through mace's public edge-force output (GraphedTeacher(edge_forces=True)): "
                         "the same forces, 1.41-1.47x faster on both arms; a --teacher-ref must match, so baseline "
                         "timings with and without it never mix")
    ap.add_argument("--hmr", type=float, default=0.0,
                    help="hydrogen mass repartitioning, Da added to each H (DMTS uses 3.0); applies to the arms "
                         "this invocation runs, so run an un-repartitioned --arms teacher reference separately")
    ap.add_argument("--tag", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--perturb-velocities", type=float, default=0.0,
                    help="relative Gaussian perturbation of the start velocities (kinetic energy kept), seeded by "
                         "--seed: a paired teacher replica for the gate's noise floor, e.g. 1e-6")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    dev = args.device
    temperature = float(args.run.split("_T")[1])

    with h5py.File(TRAJ / f"{args.run}.h5", "r") as fh:
        numbers = np.asarray(fh.attrs["numbers"], dtype=np.int64)
        cell = np.asarray(fh.attrs["cell"], dtype=np.float64)
        positions = np.asarray(fh["positions"][args.start], dtype=np.float64)
        velocities = np.asarray(fh["velocities"][args.start], dtype=np.float64)
        teacher_name = teacher_of(fh.attrs)
    atoms = frame_atoms(numbers, positions, cell)
    n = len(atoms)
    mass_np = repartition_hydrogen(numbers, atoms.get_masses(), args.hmr)
    if args.hmr > 0:
        # the stored velocities are thermal for the true masses; rescale each atom's to the new mass
        velocities = velocities * np.sqrt(atoms.get_masses() / mass_np)[:, None]
    if args.perturb_velocities > 0:
        # a paired replica: the same start, velocities perturbed at this relative size and the kinetic energy restored.
        # Chaos separates it from the unperturbed run within picoseconds at the same total energy, so two teacher
        # replicas give the noise floor of a paired comparison (independent frames differ in temperature too)
        rng = np.random.default_rng(args.seed)
        ke = float((mass_np[:, None] * velocities**2).sum())
        velocities = velocities * (1.0 + args.perturb_velocities * rng.standard_normal(velocities.shape))
        velocities = velocities * np.sqrt(ke / float((mass_np[:, None] * velocities**2).sum()))
    masses = torch.as_tensor(mass_np, dtype=torch.float64, device=dev).unsqueeze(1)
    x0 = torch.as_tensor(atoms.get_positions(), dtype=torch.float64, device=dev)
    v0 = torch.as_tensor(velocities, dtype=torch.float64, device=dev)

    # both sides run the strongest teacher we have: cuEquivariance in a CUDA graph
    calc = load_calc(teacher_name, dev, "float32", enable_cueq=True)
    teacher = GraphedTeacher(calc, atoms, dev, edge_forces=args.edge_forces)
    t_list = VerletList(cell, teacher.r_max, args.skin, dev, wrap_positions=False,
                        builder=args.neighbours)
    t_list.update(x0, force=True)
    teacher.set_edges(t_list.i, t_list.j, t_list.shifts, t_list.unit_shifts)

    plugin = None
    if args.student_plugin:
        # a student from outside frost (e.g. MACE-NEP's own NEP through TorchNEP, GPL, in its own environment):
        # the file defines make_student(path, device) -> an object with .config, .types() and
        # .graphed(types, n_atoms, capacity) giving set_edges(i, j, shifts) and __call__(x) -> (energy, forces)
        import importlib.util

        spec = importlib.util.spec_from_file_location("frost_student_plugin", args.student_plugin)
        plugin = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(plugin)
        student = plugin.make_student(args.student, dev)
    else:
        student = load_student(args.student, dev)
    if student.config.get("kind", "frost") == "mace" and args.student_backend != student.config.get("backend", "e3nn"):
        student = student.converted(args.student_backend)
    types_t = student.types(torch.as_tensor(numbers, device=dev))
    s_list = VerletList(cell, float(student.config["cutoff"]), args.student_skin, dev,
                        wrap_positions=False, builder=args.neighbours,
                        outer=t_list if args.derived_lists else None,
                        lag_margin=args.lag_margin if args.lagged_lists else None)
    s_list.update(x0, force=True)
    graphed = (student.graphed(types_t, n, int(1.3 * s_list.i.numel()) + 64) if plugin is not None
               else GraphedStudent(student, types_t, n, capacity=int(1.3 * s_list.i.numel()) + 64))
    graphed.set_edges(s_list.i, s_list.j, s_list.shifts)
    compile_error = None
    if args.student_compile:
        # the descriptor student (frost.student, no message passing) through TorchInductor inside
        # the same CUDA graph; MACE students do not trace (Dynamo), so this is for that family only
        from frost.student_fast import CompiledGraphedStudent

        eager_f = graphed(x0)[1].detach().double().clone()
        graphed = CompiledGraphedStudent(student, types_t, n, capacity=int(1.3 * s_list.i.numel()) + 64,
                                         mode=args.student_compile)
        graphed.set_edges(s_list.i, s_list.j, s_list.shifts)
        compile_error = float((graphed(x0)[1].detach().double() - eager_f).abs().max())
        if compile_error > 1e-3:
            raise RuntimeError(f"compiled student differs from eager by {compile_error:.2e} eV/A")

    remote, remote_error = None, None
    if args.async_teacher:
        if not args.k or args.respa != "boundary":
            raise SystemExit("--async-teacher needs a fixed --k and --respa boundary")
        from frost.async_teacher import MaceTeacherSpec, ProcessTeacher

        # built before any timed region, in a child process that sees only the second GPU
        # (cuEquivariance's kernel state is per process and assumes one device)
        remote = ProcessTeacher(MaceTeacherSpec(atoms, cell, args.skin, args.neighbours, teacher=teacher_name),
                                visible_device=args.teacher_device.split(":")[-1])
        remote_error = float((remote(x0)[1].to(dev) - teacher(x0)[1].double()).abs().max())
        if remote_error > 1e-3:
            raise RuntimeError(f"teacher on {args.teacher_device} differs by {remote_error:.2e} eV/A")

    f_ref = mace_forces(calc, atoms, dev)[0]
    skin_error = float(np.abs(teacher(x0)[1].double().cpu().numpy() - f_ref).max())
    if skin_error > 1e-3:
        raise RuntimeError(f"teacher on the Verlet list differs by {skin_error:.2e} eV/A")

    # edges are synced by each list's rebuild counter, not by whether this call rebuilt it:
    # a derived student list can rebuild the teacher's list between teacher calls
    edge_version = {"teacher": -1, "student": -1}

    def teacher_force(x):
        t_list.update(x)
        if t_list.rebuilds != edge_version["teacher"]:
            edge_version["teacher"] = t_list.rebuilds
            teacher.set_edges(t_list.i, t_list.j, t_list.shifts, t_list.unit_shifts)
        e, f = teacher(x)
        return e, f.double(), x

    bond = None
    if args.bond_stiffness:
        # an online stiffness correction along bonds (frost.bond_correction): the student's fast-mode error,
        # fitted from the residual changes the refreshes measure, added to the student as a harmonic potential
        from frost.bond_correction import BondStiffness

        pairs = [tuple(int(z) for z in p.split("-")) for p in args.bond_stiffness.split(",")]
        b_idx, b_kind = BondStiffness.detect(numbers, x0.cpu().numpy(), cell, pairs)
        b_idx, b_kind = b_idx.to(dev), b_kind.to(dev)
        bond = BondStiffness(b_idx, b_kind, x0, cell, len(pairs), memory=args.bond_memory)

    def student_force(x):
        s_list.update(x)
        if s_list.rebuilds != edge_version["student"]:
            edge_version["student"] = s_list.rebuilds
            graphed.set_edges(s_list.i, s_list.j, s_list.shifts)
        e, f = graphed(x)
        if bond is not None:
            e_b, f_b = bond.energy_forces(x)
            return e.detach().sum() + e_b, f.detach().double() + f_b, x
        return e.detach().sum(), f.detach().double(), x

    gl = {"student_invalid": 0, "teacher_invalid": 0, "student_runs": 0, "teacher_runs": 0}
    if args.graph_loop:
        from frost.graph_loop import GraphedLoop, StudentForce, TeacherForce

        # a compiled descriptor student goes into the interval's graph as compiled kernels (warmed up
        # by GraphedLoop on its side stream before capture)
        if args.student_compile:
            from frost.student_fast import DetachedStudent

            # detached parameters: the compiled backward then differentiates to the positions only
            loop_model = torch.compile(DetachedStudent(student), mode=args.student_compile, dynamic=False)
        else:
            loop_model = student
        sforce = StudentForce(loop_model, types_t, n, capacity=int(1.3 * s_list.i.numel()) + 64)
        sforce.set_edges(s_list.i, s_list.j, s_list.shifts)
        tforce = TeacherForce(teacher)

    synced = {"student": -1, "teacher_capacity": -1}

    def refresh_lists_for_graph(x, which, margin=0.0):
        """At a boundary: rebuild the list if needed and make the graph's copy current.

        The eager fallback rebuilds lists on its own, so the graph's copy is synced by the
        list's rebuild counter, not by whether *this* call rebuilt it. A stale copy here
        once fed the graph an old edge list and heated liquid Cu by 29 K (2026-10-02).
        Returns True when the graph must be recaptured (its buffers were reallocated).
        """
        if which == "student":
            if s_list.update(x, margin=margin)[1]:
                graphed.set_edges(s_list.i, s_list.j, s_list.shifts)
            if s_list.rebuilds != synced["student"]:
                synced["student"] = s_list.rebuilds
                return sforce.set_edges(s_list.i, s_list.j, s_list.shifts)
            return False
        if t_list.update(x)[1]:
            teacher.set_edges(t_list.i, t_list.j, t_list.shifts, t_list.unit_shifts)
        if teacher.capacity != synced["teacher_capacity"]:  # GraphedTeacher reallocated its buffers
            synced["teacher_capacity"] = teacher.capacity
            return True
        return False

    def thermostat(seed):
        if args.ensemble == "nvt" and args.thermostat == "band-gle":
            return BandGLE(args.dt, temperature, masses, args.gle_center_thz, args.gle_width_thz,
                           args.gle_peak, args.gle_base, seed)
        if args.ensemble == "nvt":
            return Langevin(args.dt, args.friction, temperature, masses, seed)
        return None

    sample_steps = max(round(args.sample_every / args.dt), 1)
    # what the sampling interval actually is, which stops being --sample-every as soon
    # as dt exceeds it: at dt = 4 with --sample-every 2 the stride floors at one step,
    # so velocities arrive every 4 fs. Passing 2 fs to vdos() then scales every
    # frequency by two -- Cu's first moment read 9.498 THz at dt = 4 against 4.750 at
    # dt = 1, a factor of 1.9996, which is a time axis and not physics (frost-ec)
    interval_fs = sample_steps * args.dt
    rdf_steps = max(round(args.rdf_every / args.dt), 1)

    def sample(store, state, step):
        """Velocities and positions at the inner rate, identical on both sides."""
        keep = (lambda t: t.to(torch.float32)) if args.device_samples else \
            (lambda t: t.to(torch.float32).cpu())
        if step % sample_steps == 0:
            store["v"].append(keep(state.v))
            store["x"].append(keep(state.x))
        if step % rdf_steps == 0:
            store["rdf_x"].append(keep(state.x))

    def energy_point(store, state, step, energy, kinetic=None):
        """Total energy wherever the teacher energy is known: every step, or every outer step.

        The running peak excursion is kept because a drift fit cannot distinguish
        instability from a larger bounded offset, and a symplectic integrator gives
        the second: the student alone at 1, 2 and 4 fs drifts by nothing while its
        peak grows 0.018 -> 0.063 -> 0.281, i.e. as dt^2 (frost-ec). The 1 ps trace
        cannot show this -- it is binned and monotonic, so its maximum is its last
        point -- so the peak is accumulated here, at the cadence the teacher energy
        actually arrives.
        """
        kinetic = state.kinetic_energy() if kinetic is None else kinetic
        total = float(energy) + kinetic
        if not np.isfinite(total) and store["blowup_fs"] is None:
            # stop here: a non-finite state otherwise runs to the end of the arm and crashes
            # the observables, taking the other arm's results with it (liq50, 2026-10-01)
            store["blowup_fs"] = step * args.dt
            return
        store["energy"].append(float(energy))
        store["kinetic"].append(kinetic)
        store["t_fs"].append(step * args.dt)
        if store["reference_total"] is None:
            store["reference_total"] = total
        excursion = abs(total - store["reference_total"]) / n * 1e3  # meV/atom
        store["peak_excursion"] = max(store["peak_excursion"], excursion)

    def empty_store():
        return {"v": [], "x": [], "rdf_x": [], "energy": [], "kinetic": [], "t_fs": [],
                "reference_total": None, "peak_excursion": 0.0, "blowup_fs": None}

    # ------------------------------------------------------------- teacher run
    steps = round(args.fs / args.dt)
    reused = None
    if args.arms == "frost":
        # the teacher arm is deterministic given (run, start, seed, dt, ensemble, friction,
        # fs), so a run that already integrated it can lend it; re-running it would cost
        # 30 minutes per 50 ps for the same trajectory
        ref, ref_seconds = None, None
        if args.teacher_ref:
            reused = json.loads(Path(args.teacher_ref).read_text())
            theirs = reused["config"]
            for key in ("run", "start", "seed", "dt", "ensemble", "friction", "fs",
                        "thermostat", "gle_center_thz", "gle_width_thz", "gle_peak", "gle_base",
                        "edge_forces"):
                if key == "seed" and args.perturb_velocities > 0:
                    continue  # a perturbed replica is a different realisation by design; its seed draws the perturbation
                # a key a reference predates takes the flag's default, which it then ran with
                theirs_value = theirs.get(key, ap.get_default(key))
                if theirs_value != getattr(args, key):
                    raise SystemExit(f"--teacher-ref {args.teacher_ref} differs in {key}: "
                                     f"{theirs_value!r} against {getattr(args, key)!r}")
            if reused.get("teacher_model", "mp-medium") != teacher_name:
                raise SystemExit(f"--teacher-ref {args.teacher_ref} was made with {reused.get('teacher_model')!r}, "
                                 f"this run's trajectory with {teacher_name!r}")
            # a teacher-only file (--arms teacher) records ms_per_fs and the self spread under "teacher"
            ref_seconds = ((reused.get("wall_seconds") or {}).get("teacher")
                           or reused["ms_per_fs"] * args.fs / 1e3)
            reused.setdefault("teacher_self", reused["teacher"].get("self"))
    else:
        state = MDState(x0.clone(), v0.clone(), masses)
        integ = Integrator(state, args.dt, thermostat(args.seed))
        energy, forces, state.x = teacher_force(state.x)
        ref = empty_store()
        sample(ref, state, 0)
        energy_point(ref, state, 0, energy)
        t_loop = (GraphedLoop(tforce, masses, integ.dt, args.teacher_chunk, integ.thermostat)
                  if args.graph_loop else None)
        if args.lagged_lists:
            t_list.lag_margin = args.lag_margin  # consulted every step in this arm only
        t0 = time.perf_counter()
        step = 0
        while step < steps:
            if t_loop is not None and steps - step >= t_loop.k:
                if refresh_lists_for_graph(state.x, "teacher"):
                    t_loop.invalidate()
                saved = (state.x.clone(), state.v.clone(), forces.clone(),
                         integ.thermostat.gen.get_state() if integ.thermostat is not None else None)
                xg, vg, fg = t_loop.run(state.x, state.v, forces, t_list.x_ref)
                gl["teacher_runs"] += 1
                if float(t_loop.maxdisp) <= 0.5 * t_list.skin:
                    state.x, state.v, forces = xg.clone(), vg.clone(), fg.clone()
                    e_c, k_c = t_loop.energy.cpu(), t_loop.kinetic.cpu()
                    # samples live where --device-samples keeps the eager path's, or the arm's stack mixes devices
                    xs_c, vs_c = ((t_loop.xs.clone(), t_loop.vs.clone()) if args.device_samples
                                  else (t_loop.xs.cpu(), t_loop.vs.cpu()))
                    for j in range(t_loop.k):
                        step += 1
                        if step % sample_steps == 0:
                            ref["v"].append(vs_c[j])
                            ref["x"].append(xs_c[j])
                        if step % rdf_steps == 0:
                            ref["rdf_x"].append(xs_c[j])
                        energy_point(ref, state, step, float(e_c[j]), kinetic=float(k_c[j]))
                    continue
                gl["teacher_invalid"] += 1  # an atom left the skin: redo these steps eagerly
                state.x, state.v, forces = saved[0], saved[1], saved[2]
                if saved[3] is not None:
                    integ.thermostat.gen.set_state(saved[3])
                for _ in range(t_loop.k):
                    step += 1
                    energy, forces = integ.inner_step(teacher_force, forces)
                    sample(ref, state, step)
                    energy_point(ref, state, step, energy)
                continue
            step += 1
            energy, forces = integ.inner_step(teacher_force, forces)
            sample(ref, state, step)
            energy_point(ref, state, step, energy)
        torch.cuda.synchronize()
        ref_seconds = time.perf_counter() - t0
        t_list.lag_margin = None  # the frost arm consults it only at boundaries: check synchronously

    # ------------------------------------------------------------- frost run
    state = MDState(x0.clone(), v0.clone(), masses)
    integ = Integrator(state, args.dt, thermostat(args.seed))
    if args.epsilon is None:
        args.epsilon = 2.0 if args.units == "acceleration" else 20.0
    unit = "meV/(A amu)" if args.units == "acceleration" else "meV/A"
    controller = IntervalController(k_jitter=args.k_jitter, epsilon=args.epsilon,
                                    k=max(args.k, 2), k_max=args.k_max,
                                    period_fs=args.fastest_period_fs or None, dt_fs=args.dt)
    energy, f_t, state.x = teacher_force(state.x)
    _, f_s, state.x = student_force(state.x)
    residual = f_t - f_s
    x_boundary = {"x": state.x.clone()}  # the last boundary's positions (the bond-stiffness fit's dr)
    # ------------------------------------------------------------- observables
    def subset(store, lo_frac: float, hi_frac: float):
        """One part of a run. Comparing the teacher's halves gives the error bar."""
        return {key: (values[int(len(values) * lo_frac):int(len(values) * hi_frac)]
                      if isinstance(values, list) else values)
                for key, values in store.items()}

    def observables(store):
        v = torch.stack(store["v"]).to(dev).double()
        x = torch.stack(store["x"]).to(dev).double()
        total = np.asarray(store["energy"]) + np.asarray(store["kinetic"])
        t_ps = np.asarray(store["t_fs"]) / 1e3
        finite = np.isfinite(total)
        drift = (np.polyfit(t_ps[finite], total[finite] / n, 1)[0] * 1e3
                 if finite.sum() > 4 else float("nan"))
        # the trace, in 1 ps bins relative to the first bin, says whether a drift is
        # linear (a shadow-energy offset) or growing (an instability); the two halves'
        # slopes say the same in two numbers
        bins = np.floor(t_ps[finite]).astype(int)
        e_bin = np.array([total[finite][bins == b].mean() / n for b in np.unique(bins)])
        trace = (e_bin - e_bin[0]) * 1e3 if len(e_bin) else e_bin
        half = len(trace) // 2
        slopes = [float(np.polyfit(np.arange(len(part)), part, 1)[0]) if len(part) > 2 else float("nan")
                  for part in (trace[:half], trace[half:])]
        freq, spectrum, correlation = vdos(v, interval_fs, max_lag=1024)
        curve, diffusion = msd(x, interval_fs)
        # 150 THz covers every real mode; the ::4 downsampling below it was chosen
        # for file size and costs the resolution a timestep question needs -- a 0.86%
        # dispersion shift of silica's 15.8 THz band is 0.14 THz, a seventh of the
        # stored 0.98 THz bin. Keep every bin up to 60 THz, which is above the
        # highest real mode on every system here, and thin only the empty tail.
        keep = freq <= 150.0
        fine = freq <= 60.0
        moment = float((freq[keep] * spectrum[keep]).sum() / spectrum[keep].sum())
        return {
            "wall_seconds": None,
            "drift_meV_per_atom_per_ps": drift,
            "drift_halves_meV_per_atom_per_ps": slopes,
            "peak_excursion_meV_per_atom": store.get("peak_excursion", float("nan")),
            "energy_trace_meV_per_atom_1ps": as_list(trace),
            "mean_total_energy_eV_per_atom": float(np.nanmean(total) / n),
            "mean_temperature_K": float(np.mean([2.0 * k_e / (3 * n * 8.617333262e-5)
                                                 for k_e in store["kinetic"]])),
            "vdos_freq_THz": as_list(freq[keep][::4]),
            "vdos": as_list(spectrum[keep][::4] / spectrum[keep].max()),
            "vacf": as_list(correlation),
            "vacf_interval_fs": interval_fs,
            "vdos_freq_fine_THz": as_list(freq[fine]),
            "vdos_fine": as_list(spectrum[fine] / spectrum[keep].max()),
            "vdos_first_moment_THz": moment,
            "msd_A2": as_list(curve[::4]),
            "diffusion_A2_per_ps": diffusion,
            "rdf": rdf(torch.stack(store["rdf_x"]), cell,
                       types_t.cpu().numpy(), int(types_t.max()) + 1),
        }

    def compare(a: dict, b: dict) -> dict:
        def l1(key, field=None):
            x = np.asarray(a[key][field] if field else a[key], dtype=float)
            y = np.asarray(b[key][field] if field else b[key], dtype=float)
            if key == "vdos" and x.shape != y.shape:
                # a reused reference written before the spectrum's grid changed: compare on its grid
                y = np.interp(np.asarray(a["vdos_freq_THz"]), np.asarray(b["vdos_freq_THz"]), y)
            return float(np.abs(y - x).sum() / np.abs(x).sum())

        return {
            "drift_ratio": (b["drift_meV_per_atom_per_ps"] / a["drift_meV_per_atom_per_ps"]
                            if a["drift_meV_per_atom_per_ps"] else None),
            "vdos_l1_relative": l1("vdos"),
            "vdos_first_moment_shift_THz": b["vdos_first_moment_THz"] - a["vdos_first_moment_THz"],
            "rdf_l1_relative": l1("rdf", "g_total"),
            "diffusion_ratio": (b["diffusion_A2_per_ps"] / a["diffusion_A2_per_ps"]
                                if a["diffusion_A2_per_ps"] else None),
            "temperature_shift_K": b["mean_temperature_K"] - a["mean_temperature_K"],
        }

    if args.arms == "teacher":
        a = observables(ref)
        # the teacher's own half-to-half spread: Cu at k = 32 passed the energy line
        # and failed on structure, so a step is admissible only if the RDF agrees too
        a["self"] = compare(observables(subset(ref, 0.0, 0.5)),
                            observables(subset(ref, 0.5, 1.0)))
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUT_DIR / f"fidelity_{args.run}_{args.ensemble}{'_' + args.tag if args.tag else ''}.json"
        out.write_text(json.dumps({"run": args.run, "n_atoms": n, "arms": "teacher",
                                   "ensemble": args.ensemble, "dt_fs": args.dt, "fs": args.fs,
                                   "teacher_model": teacher_name,
                                   "config": vars(args), "ms_per_fs": 1e3 * ref_seconds / args.fs,
                                   "wall_seconds": {"teacher": ref_seconds, "frost": None},
                                   "teacher": a, "teacher_self": a["self"]}, indent=2))
        trace = a["energy_trace_meV_per_atom_1ps"]
        print(f"{args.run} {args.ensemble} teacher only: dt {args.dt} fs | "
              f"{1e3 * ref_seconds / args.fs:.2f} ms/fs | drift "
              f"{a['drift_meV_per_atom_per_ps']:.4f} meV/atom/ps | total "
              f"{trace[-1] if trace else float('nan'):+.3f} over {args.fs / 1e3:.0f} ps | halves "
              f"{[round(x, 4) for x in a['drift_halves_meV_per_atom_per_ps']]} | T "
              f"{a['mean_temperature_K']:.1f} K | own halves: RDF "
              f"{a['self']['rdf_l1_relative']:.4f} VDOS {a['self']['vdos_l1_relative']:.4f}"
              f" | -> {out}", flush=True)
        return 0

    frost = empty_store()
    sample(frost, state, 0)
    energy_point(frost, state, 0, energy)
    from collections import deque

    recent = deque(maxlen=8)  # (step, x, v) at outer boundaries, for a blow-up's post-mortem

    def remember(at):
        recent.append((at, state.x.detach().clone(), state.v.detach().clone()))
    staleness_log: list[tuple[float, int, float]] = []
    step = 0
    jitter_gen = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    t0 = time.perf_counter()
    inner = {"step": 0}

    def on_inner_step():
        inner["step"] += 1
        sample(frost, state, inner["step"])

    def residual_staleness(r_new, previous):
        """Both staleness units for one refreshed residual, against the one it replaces."""
        change = r_new - previous
        force = float(change.norm(dim=1).pow(2).mean().sqrt()) * 1e3
        accel = float((change / masses).norm(dim=1).pow(2).mean().sqrt()) * 1e3
        return force, accel, (accel if args.units == "acceleration" else force)

    if args.rollback and args.ensemble == "nvt" and args.thermostat != "langevin":
        raise SystemExit("--rollback restores the Langevin generator; the band-GLE state is not "
                         "checkpointed")
    if args.rollback and args.respa != "boundary":
        raise SystemExit("--rollback needs --respa boundary")
    eps_rb = args.rollback_epsilon if args.rollback_epsilon is not None else args.epsilon
    rb = {"events": [], "teacher_calls": 0, "atom_max": []}
    gen = integ.thermostat.gen if integ.thermostat is not None else None

    def checkpoint(closing=None):
        """Everything an outer interval changes. ``closing``: the half kick a restore owes."""
        return {"x": state.x.clone(), "v": state.v.clone(), "f_s": f_s.clone(),
                "residual": residual.clone(), "closing": closing, "step": step,
                "gen": gen.get_state() if gen is not None else None, "inner": inner["step"],
                "n": {key: len(frost[key]) for key in ("v", "x", "rdf_x")}}

    def restore(c):
        # the Verlet lists need no restore: each rebuilds from its own reference positions
        # whenever the displacement says so, in either direction
        nonlocal f_s, residual, step
        state.x, state.v = c["x"].clone(), c["v"].clone()
        f_s, residual, step = c["f_s"].clone(), c["residual"].clone(), c["step"]
        if gen is not None:
            gen.set_state(c["gen"])  # the redo replays the same noise: refinement, not selection
        inner["step"] = c["inner"]
        for key, length in c["n"].items():
            del frost[key][length:]

    s_loop = (GraphedLoop(sforce, masses, integ.dt, args.student_chunk or args.k, integ.thermostat)
              if args.graph_loop and args.k else None)

    def student_steps(k_use):
        """k inner student steps: graph replays of ``s_loop.k`` steps, eager for any remainder."""
        if s_loop is not None and k_use % s_loop.k == 0:
            for _ in range(k_use // s_loop.k):
                student_chunk(s_loop.k)
            return
        for _ in range(k_use):
            eager_inner()

    def eager_inner():
        nonlocal f_s
        _, f_s = integ.inner_step(student_force, f_s)
        on_inner_step()

    def student_chunk(k_use):
        """One graph replay of k_use steps; redone eagerly if an atom left the skin."""
        nonlocal f_s
        # rebuild ahead of need: the chunk runs k_use steps unchecked, and an atom at the fastest current speed
        # covers v_max k_use dt of the half-skin in them (1.5x for acceleration). Rebuilding only after a chunk
        # failed redid 41% of Cu's chunks at a 0.8 A skin (fidelity_cu500_T300_nve_f45_k24_graphloop_skin08.json)
        reach = (1.5 * float(state.v.norm(dim=1).max()) * k_use * integ.dt) if args.predictive_lists else 0.0
        if refresh_lists_for_graph(state.x, "student", margin=reach):
            s_loop.invalidate()
        saved = (state.x.clone(), state.v.clone(), f_s.clone(),
                 gen.get_state() if gen is not None else None)
        xg, vg, fg = s_loop.run(state.x, state.v, f_s, s_list.x_ref)
        gl["student_runs"] += 1
        if float(s_loop.maxdisp) <= 0.5 * s_list.skin:
            state.x, state.v, f_s = xg.clone(), vg.clone(), fg.clone()
            xs_c, vs_c = ((s_loop.xs.clone(), s_loop.vs.clone()) if args.device_samples
                          else (s_loop.xs.cpu(), s_loop.vs.cpu()))
            for j in range(k_use):
                inner["step"] += 1
                if inner["step"] % sample_steps == 0:
                    frost["v"].append(vs_c[j])
                    frost["x"].append(xs_c[j])
                if inner["step"] % rdf_steps == 0:
                    frost["rdf_x"].append(xs_c[j])
            return
        gl["student_invalid"] += 1  # an atom left the student skin: redo eagerly
        state.x, state.v, f_s = saved[0], saved[1], saved[2]
        if saved[3] is not None:
            gen.set_state(saved[3])
        for _ in range(k_use):
            eager_inner()

    def atom_change(r_new, r_old):
        return float(((r_new - r_old) / masses).norm(dim=1).max()) * 1e3

    def violates(st_report, atom, finite):
        return (not finite or not np.isfinite(st_report) or st_report > eps_rb
                or (args.rollback_atom is not None and atom > args.rollback_atom))

    def interval(k_use):
        """One boundary-RESPA outer step. With --rollback, undone and halved on a violation."""
        nonlocal f_s, residual, step
        ck = checkpoint() if args.rollback else None
        integ.kick(residual, 0.5 * k_use * integ.dt)  # respa_outer's opening half kick
        student_steps(k_use)
        finite = bool(torch.isfinite(state.x).all()) and bool(torch.isfinite(state.v).all())
        if finite:
            energy, f_t, state.x = teacher_force(state.x)
            rb["teacher_calls"] += 1
            r_new = f_t - f_s
            f_st, a_st, st = residual_staleness(r_new, residual)
            atom = atom_change(r_new, residual)
            finite = bool(np.isfinite(float(energy)))
        else:
            energy, st, atom = float("nan"), float("nan"), float("nan")
        if args.rollback and k_use > 1 and violates(st / np.sqrt(3.0), atom, finite):
            rb["events"].append({"t_fs": step * args.dt, "k": k_use, "async": False,
                                 "staleness": st / np.sqrt(3.0), "atom": atom})
            restore(ck)
            k1 = k_use // 2
            interval(k1)
            interval(k_use - k1)
            return
        if not finite:  # nothing to roll back to below k = 1: record the blow-up and stop
            energy_point(frost, state, step + k_use, float("nan"))
            step += k_use
            return
        if bond is not None:
            # refit dK from this interval's residual change, then recompute the student force and the residual at
            # x from the same teacher call, so F_S + R = F_T at the boundary (CLAUDE.md trap 8)
            _, f_b_old = bond.energy_forces(state.x)
            bond.update(r_new - residual, x_boundary["x"], state.x)
            _, f_b_new = bond.energy_forces(state.x)
            f_s = f_s - f_b_old + f_b_new
            r_new = f_t - f_s
        x_boundary["x"] = state.x.clone()
        integ.kick(r_new, 0.5 * k_use * integ.dt)
        residual = r_new
        step += k_use
        energy_point(frost, state, step, energy)
        staleness_log.append((step * args.dt, k_use, st / np.sqrt(3.0), f_st / np.sqrt(3.0),
                              a_st / np.sqrt(3.0)))
        rb["atom_max"].append(atom)
        remember(step)
        if not args.k:
            controller.observe(st, k_use)

    async_stats = None
    async_ran = remote is not None
    if remote is not None:
        # C2: the teacher evaluates x_n on the second GPU while the student integrates the
        # next interval; the kick at boundary n uses R(x_{n-1}), held, never extrapolated
        from frost.async_teacher import LaggedResidual

        from collections import deque as _deque

        lag = LaggedResidual(remote, residual, depth=args.lag, compensate=args.lag_compensate,
                             types=types_t, f_student0=f_s)
        waited = 0.0
        cks = _deque(maxlen=2)  # pre-kick checkpoints at the last two boundaries
        if args.rollback:
            cks.append(checkpoint(closing=None))
        integ.kick(residual, 0.5 * args.k * integ.dt)

        def record(finished):
            energy_point(frost, state, finished["step"], finished["energy"],
                         kinetic=finished["kinetic"])
            f_st, a_st, st = residual_staleness(finished["residual"], finished["previous"])
            staleness_log.append((finished["step"] * args.dt, args.k, st / np.sqrt(3.0),
                                  f_st / np.sqrt(3.0), a_st / np.sqrt(3.0)))

        phase = {"student_s": [], "boundary_s": []}  # per outer step, to see transients
        while step < steps:
            k_now = min(args.k, steps - step)
            t_a = time.perf_counter()
            student_steps(k_now)
            step += k_now
            t_b = time.perf_counter()
            kick, finished = lag.boundary(state.x, f_s, state.kinetic_energy(), step)
            if finished is not None:
                rb["atom_max"].append(atom_change(finished["residual"], finished["previous"]))
            if args.rollback:
                finite = bool(torch.isfinite(state.x).all()) and bool(torch.isfinite(state.v).all())
                bad = not finite
                if finished is not None and not bad:
                    _, _, st = residual_staleness(finished["residual"], finished["previous"])
                    bad = violates(st / np.sqrt(3.0), rb["atom_max"][-1],
                                   bool(np.isfinite(float(finished["energy"]))))
                if bad:
                    # the change R(x_n) - R(x_{n-1}) arrives at boundary n + 1: undo the last two
                    # intervals, redo them synchronously (halving as needed), restart the pipeline
                    lag.drain()
                    waited += lag.waited_s
                    detect = step
                    ck = cks[0]
                    rb["events"].append({"t_fs": ck["step"] * args.dt, "k": args.k, "async": True,
                                         "staleness": (st / np.sqrt(3.0)) if finite and finished
                                         is not None else float("nan")})
                    restore(ck)
                    if ck["closing"] is not None:
                        integ.kick(ck["closing"], 0.5 * args.k * integ.dt)  # the past, as it was
                    _, f_t, state.x = teacher_force(state.x)
                    rb["teacher_calls"] += 1
                    residual = f_t - f_s
                    while step < detect and frost["blowup_fs"] is None:
                        interval(min(args.k, detect - step))
                    if frost["blowup_fs"] is not None:
                        break
                    lag = LaggedResidual(remote, residual, depth=args.lag, compensate=args.lag_compensate,
                                         types=types_t, f_student0=f_s)
                    cks.clear()
                    cks.append(checkpoint(closing=None))
                    integ.kick(residual, 0.5 * args.k * integ.dt)
                    phase["student_s"].append(t_b - t_a)
                    phase["boundary_s"].append(time.perf_counter() - t_b)
                    continue
                cks.append(checkpoint(closing=kick))
            if finished is not None:
                record(finished)
            integ.kick(kick, k_now * integ.dt)
            phase["student_s"].append(t_b - t_a)
            phase["boundary_s"].append(time.perf_counter() - t_b)
            remember(step)
            if frost["blowup_fs"] is not None:
                break
        finished = lag.drain() if frost["blowup_fs"] is None else None
        if finished is not None:
            record(finished)
        async_stats = {"teacher_device": args.teacher_device, "k": args.k, "lag": args.lag,
                       "lag_compensate": args.lag_compensate,
                       "mu_final": (lag.mu().tolist() if args.lag_compensate else None),
                       "main_thread_waited_s": waited + lag.waited_s,
                       "student_ms_per_outer_step_by_quarter": [
                           1e3 * float(np.mean(q)) for q in np.array_split(phase["student_s"], 4)],
                       "boundary_ms_per_outer_step_by_quarter": [
                           1e3 * float(np.mean(q)) for q in np.array_split(phase["boundary_s"], 4)],
                       "remote_max_abs_force_error_eV_per_A": remote_error}
        # the second GPU's process is shut down after the clock stops: its start-up is outside
        # the timed region too, as the teacher arm's model load and graph capture are. Until
        # 2026-10-02 teardown sat inside it: 1.3-3.6 s per 10 ps run, up to 24% of the frost time
        torch.cuda.synchronize()
        loop_end = time.perf_counter()
        t_close = time.perf_counter()
        remote.close()
        async_stats["teardown_s"] = time.perf_counter() - t_close

    spec = {"transport": [], "transport_rel": [], "residual_prev": residual.clone()}

    def spec_interval(k_use):
        """Speculative refresh: predict the boundary, refresh the teacher there, transport.

        1. Look-ahead: the student integrates the interval with the residual held (or
           extrapolated), replaying the thermostat noise the true interval will use, to a
           predicted boundary state x_hat. Nothing is sampled.
        2. The teacher at x_hat gives R_hat = F_T(x_hat) - F_S(x_hat). On two GPUs this call
           runs while the student integrates the previous interval; here it is serial.
        3. The true interval ramps the residual from R_n to R_hat (respa_interp).
        4. R_hat becomes R_{n+1}: the residual is transported from x_hat to x_{n+1}, an error
           of grad R . (x_{n+1} - x_hat), second order in the residual's change over one
           interval, where the one-step-behind pipeline makes it first order.
        """
        nonlocal f_s, residual, step
        saved = (state.x.clone(), state.v.clone(), f_s.clone(),
                 gen.get_state() if gen is not None else None)
        r_end_guess = (residual if args.spec_predict == "hold"
                       else residual + (residual - spec["residual_prev"]))
        e_s_hat, f_s_hat = integ.respa_interp(student_force, f_s, residual, r_end_guess, k_use)
        x_hat = state.x.clone()
        e_hat, f_t_hat, _ = teacher_force(x_hat)
        r_hat = f_t_hat - f_s_hat
        state.x, state.v, f_s = saved[0], saved[1], saved[2]
        if saved[3] is not None:
            gen.set_state(saved[3])  # the true interval replays the look-ahead's noise
        e_s_true, f_s = integ.respa_interp(student_force, f_s, residual, r_hat, k_use,
                                           on_step=on_inner_step)
        spec.setdefault("dx", []).append(float((state.x - x_hat).norm(dim=1).max()))
        if args.spec_diagnose:
            energy, f_t, state.x = teacher_force(state.x)
            r_true = f_t - f_s
            err = float(((r_hat - r_true) / masses).norm(dim=1).pow(2).mean().sqrt()) * 1e3
            spec["transport"].append(err / np.sqrt(3.0))
            change = float(((r_true - residual) / masses).norm(dim=1).pow(2).mean().sqrt()) * 1e3
            spec["transport_rel"].append(err / max(change, 1e-12))
        else:
            # the energy is transported as the force is: E_T(x) ~ E_T(x_hat) + E_S(x) - E_S(x_hat)
            energy = float(e_hat) + float(e_s_true) - float(e_s_hat)
        f_st, a_st, st = residual_staleness(r_hat, residual)
        spec["residual_prev"] = residual
        residual = r_hat
        step += k_use
        energy_point(frost, state, step, energy)
        staleness_log.append((step * args.dt, k_use, st / np.sqrt(3.0), f_st / np.sqrt(3.0),
                              a_st / np.sqrt(3.0)))
        remember(step)
        if not args.k:
            controller.observe(st, k_use)

    while step < steps and not async_ran:
        k_use = min(controller.draw(jitter_gen, args.k or controller.k), steps - step)
        if args.respa == "boundary":
            interval(k_use)  # logs, remembers and observes for itself
            if frost["blowup_fs"] is not None:
                break
            continue
        if args.respa == "spec":
            spec_interval(k_use)
            if frost["blowup_fs"] is not None:
                break
            continue
        if args.respa == "switch":
            # Kanda et al.'s multi-timestep switching (Adv. Intell. Discov. 2026), as LAMMPS runs it: the
            # cheap model on k - 1 steps, the expensive one on the k-th, each force carried into the next
            # step's first half kick; no residual, so the trajectory feels the student's error on (k-1)/k
            # of the time. Same teacher-call rate as Frost at the same k
            f_cur = f_s
            for _ in range(k_use - 1):
                _, f_cur = integ.inner_step(student_force, f_cur)
                on_inner_step()
            energy, f_cur = integ.inner_step(teacher_force, f_cur)
            on_inner_step()
            f_s = f_cur  # LAMMPS semantics: the teacher's force carries into the next student step's first half kick
            step += k_use
            energy_point(frost, state, step, energy)
            staleness_log.append((step * args.dt, k_use, 0.0, 0.0, 0.0))
            if frost["blowup_fs"] is not None:
                break
            continue
        if args.respa == "student":
            # the distilled model alone (MACE-NEP's protocol, our positive control): no residual is
            # ever applied; the teacher is called every k steps only to record its energy along the
            # student's trajectory, so the gates compare the student's physics with the teacher's
            for _ in range(k_use):
                eager_inner()
            energy, f_t, state.x = teacher_force(state.x)
            r_now = f_t - f_s
            force_staleness, accel_staleness, staleness = residual_staleness(r_now, torch.zeros_like(r_now))
            step += k_use
            energy_point(frost, state, step, energy)
            staleness_log.append((step * args.dt, k_use, staleness / np.sqrt(3.0), force_staleness / np.sqrt(3.0),
                                  accel_staleness / np.sqrt(3.0)))
            if frost["blowup_fs"] is not None:
                break
            continue
        if args.respa == "midpoint":
            def slow_at(x, f_student, k=k_use, at=step):
                # the residual's two halves must come from the same positions: the
                # student force here is the one the last inner step already produced
                e_t, f_teacher, _ = teacher_force(x)
                r = f_teacher - f_student
                # the teacher energy is known *here*, so the energy sample belongs
                # here too, against the kinetic energy at these same positions
                energy_point(frost, state, at + k // 2, e_t)
                return r, r

            _, f_s, r_new = integ.respa_midpoint(student_force, f_s, slow_at, k_use,
                                                 on_step=on_inner_step)
            force_staleness, accel_staleness, staleness = residual_staleness(r_new, residual)
            residual = r_new
            step += k_use
        else:  # spread
            _, f_s = integ.respa_spread(student_force, f_s, residual, k_use, on_step=on_inner_step)
            energy, f_t, state.x = teacher_force(state.x)
            r_new = f_t - f_s
            force_staleness, accel_staleness, staleness = residual_staleness(r_new, residual)
            residual = r_new  # no closing kick: the residual was never applied impulsively
            step += k_use
            energy_point(frost, state, step, energy)
        staleness_log.append((step * args.dt, k_use, staleness / np.sqrt(3.0),
                              force_staleness / np.sqrt(3.0), accel_staleness / np.sqrt(3.0)))
        remember(step)
        if frost["blowup_fs"] is not None:
            break
        if not args.k:
            controller.observe(staleness, k_use)
    torch.cuda.synchronize()
    frost_seconds = (loop_end if async_ran else time.perf_counter()) - t0
    suffix = f"_{args.tag}" if args.tag else ""
    if frost["blowup_fs"] is not None:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        frames = OUT_DIR / f"blowup_{args.run}_{args.ensemble}{suffix}.pt"
        torch.save({"steps": [r[0] for r in recent], "x": [r[1].cpu() for r in recent],
                    "v": [r[2].cpu() for r in recent], "dt_fs": args.dt, "k": args.k,
                    "blowup_fs": frost["blowup_fs"]}, frames)
        out = OUT_DIR / f"fidelity_{args.run}_{args.ensemble}{suffix}.json"
        report = {"run": args.run, "n_atoms": n, "ensemble": args.ensemble, "dt_fs": args.dt,
                  "teacher_model": teacher_name,
                  "fs": args.fs, "config": {k: v for k, v in vars(args).items() if k != "device"},
                  "student": student.config,
                  "teacher": (observables(ref) if ref is not None
                              else reused["teacher"] if reused is not None else None),
                  "teacher_self": (compare(observables(subset(ref, 0.0, 0.5)),
                                           observables(subset(ref, 0.5, 1.0)))
                                   if ref is not None
                                   else reused["teacher_self"] if reused is not None else None),
                  "teacher_ref": args.teacher_ref,
                  "frost": None,
                  "frost_blowup": {"t_fs": frost["blowup_fs"], "frames": str(frames),
                                   "wall_seconds": frost_seconds},
                  "wall_seconds": {"teacher": ref_seconds, "frost": None}}
        out.write_text(json.dumps(report, indent=2))
        print(f"{args.run} {args.ensemble}: FROST BLEW UP at {frost['blowup_fs']:.0f} fs "
              f"(k {args.k}, dt {args.dt}); teacher arm kept; frames -> {frames}; -> {out}",
              flush=True)
        return 3

    report = {
        "run": args.run, "n_atoms": n, "ensemble": args.ensemble, "dt_fs": args.dt,
        "teacher_model": teacher_name,
        "fs": args.fs, "config": {k: v for k, v in vars(args).items() if k != "device"},
        "student": student.config,
        "skin_list_max_abs_force_error_eV_per_A": skin_error,
        "student_compile": {"mode": args.student_compile, "max_abs_force_diff_eV_per_A": compile_error},
        "teacher": (observables(ref) if ref is not None
                    else reused["teacher"] if reused is not None else None),
        "teacher_ref": args.teacher_ref,
        "frost": observables(frost),
        "staleness": {
            "units": unit,
            "mean": float(np.mean([s for _, _, s, _, _ in staleness_log])),
            "max": float(np.max([s for _, _, s, _, _ in staleness_log])),
            "over_budget_fraction": float(np.mean([s > args.epsilon
                                                   for _, _, s, _, _ in staleness_log])),
            "mean_k": float(np.mean([k for _, k, _, _, _ in staleness_log])),
            "mean_meV_per_A": float(np.mean([f for _, _, _, f, _ in staleness_log])),
            "mean_meV_per_A_amu": float(np.mean([a for _, _, _, _, a in staleness_log])),
            "max_meV_per_A_amu": float(np.max([a for _, _, _, _, a in staleness_log])),
        },
        "wall_seconds": {"teacher": ref_seconds, "frost": frost_seconds},
        "async_teacher": async_stats,
        "spec": ({"predict": args.spec_predict, "diagnose": args.spec_diagnose,
                  "max_boundary_dx_A": {q: float(np.quantile(spec["dx"], float(q)))
                                        for q in ("0.5", "0.99", "1.0")} if spec.get("dx") else None,
                  "transport_error_meV_per_A_amu": ({"mean": float(np.mean(spec["transport"])),
                                                     "max": float(np.max(spec["transport"]))}
                                                    if spec["transport"] else None),
                  "transport_error_over_residual_change": ({
                      "mean": float(np.mean(spec["transport_rel"])),
                      "max": float(np.max(spec["transport_rel"]))} if spec["transport_rel"] else None)}
                 if args.respa == "spec" else None),
        "list_rebuilds": {"teacher": t_list.rebuilds, "student": s_list.rebuilds},
        "graph_loop": {"enabled": args.graph_loop, "teacher_chunk": args.teacher_chunk,
                       "student_chunk": args.student_chunk or args.k, **gl},
        "rollback": {"enabled": args.rollback, "epsilon": eps_rb, "atom": args.rollback_atom,
                     "n_events": len(rb["events"]), "events": rb["events"][:100],
                     "teacher_calls": rb["teacher_calls"],
                     "atom_change_quantiles_meV_per_A_amu": (
                         {q: float(np.nanquantile(rb["atom_max"], float(q)))
                          for q in ("0.5", "0.99", "0.999", "1.0")} if rb["atom_max"] else None)},
        "speedup": ref_seconds / frost_seconds if ref_seconds else None,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"fidelity_{args.run}_{args.ensemble}{suffix}.json"
    # the structure frames of both arms, so a later estimator fix re-scores without a rerun (the RDF of unwrapped
    # frames was wrong until 2026-10-19, and every verdict before then had to be re-run to be re-checked)
    frames = {"frost": torch.stack(frost["rdf_x"]).cpu().numpy().astype(np.float32)}
    if ref is not None:
        frames["teacher"] = torch.stack(ref["rdf_x"]).cpu().numpy().astype(np.float32)
    np.savez_compressed(OUT_DIR / f"frames_{args.run}_{args.ensemble}{suffix}.npz", cell=cell, **frames)
    b = report["frost"]
    if report["teacher"] is None:
        # frost arm alone: the verdict is its own energy record (G3's 50 ps criterion)
        out.write_text(json.dumps(report, indent=2))
        print(f"{args.run} {args.ensemble} frost only: k {args.k} dt {args.dt} | "
              f"{1e3 * frost_seconds / args.fs:.2f} ms/fs | drift "
              f"{b['drift_meV_per_atom_per_ps']:.4f} meV/atom/ps | halves "
              f"{[round(x, 4) for x in b['drift_halves_meV_per_atom_per_ps']]} | peak "
              f"{b.get('peak_excursion_meV_per_atom', float('nan')):.3f} | T "
              f"{b['mean_temperature_K']:.1f} K | -> {out}", flush=True)
        return 0
    a = report["teacher"]
    report["comparison"] = compare(a, b)
    # the same statistics between the teacher's own halves: the bar any difference
    # must clear before it counts as a change in the physics
    report["teacher_self"] = (compare(observables(subset(ref, 0.0, 0.5)),
                                      observables(subset(ref, 0.5, 1.0)))
                              if ref is not None else reused["teacher_self"])
    out.write_text(json.dumps(report, indent=2))
    c, s = report["comparison"], report["teacher_self"]
    print(f"{args.run} {args.ensemble}: speedup {report['speedup']:.2f}x | drift teacher "
          f"{a['drift_meV_per_atom_per_ps']:.3f} frost {b['drift_meV_per_atom_per_ps']:.3f} "
          f"meV/atom/ps | VDOS L1 {c['vdos_l1_relative']:.3f} (teacher halves "
          f"{s['vdos_l1_relative']:.3f}), moment shift "
          f"{c['vdos_first_moment_shift_THz']:+.2f} THz | RDF L1 {c['rdf_l1_relative']:.3f} "
          f"(halves {s['rdf_l1_relative']:.3f}) | "
          f"T shift {c['temperature_shift_K']:+.1f} K | staleness "
          f"{report['staleness']['mean']:.2f} mean, {report['staleness']['max']:.2f} max {unit} "
          f"({report['staleness']['mean_meV_per_A']:.1f} meV/A) | -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
