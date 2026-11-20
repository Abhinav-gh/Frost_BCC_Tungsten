"""MD through Frost's runtime (frost.runtime): the baseline arm and the Frost arm in one process.

The Frost arm is ``runtime.Executor`` driving ``runtime.md.MDBackend`` with the certificate as
clauses (magnitude and non-finite roll back; resonance bounds the interval; drift latches a
refusal, NVE only). Optional arms compare monitors and schedules on the same start:
- ``--fixed-k K``: Frost at a fixed interval, no adaptation (rollback still on unless
  ``--no-rollback``);
- ``--monitor green:T``: the clauses replaced by a Green-style QoS threshold T.

``--ramp T1:PS`` heats the thermostat linearly from the run's temperature to T1 over PS ps
(NVT only), so a schedule certified at the start meets conditions it was never certified for
(gap N10: Cu melting at 1358 K).

``--never-lose`` lets the adaptive arm run the teacher itself when even the smallest multirate interval fails, probing
the student cheaply until an interval certifies again (``Executor(never_lose=True)``). ``--frames`` keeps every arm's
velocities (2 fs) and positions (20 fs) for fidelity scoring by ``score_frames.py``, so a headline run through the
runtime carries its own paired teacher reference.

Output: data/results/e13/md_<run>_<tag>.json (and frames_<run>_<tag>.npz)

Usage (guarded launcher; GPU)
  scripts/safe_run.sh md_rt .venvs/frost-mace/bin/python experiments/e13_runtime/md_runtime.py \\
      --run cu500_T300 --student data/models/students/cu500_T300_f45c16.pt --compile default --fs 10000 --tag t
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from frost.controller import DriftMonitor, IntervalController
from frost.integrators import MDState, StochasticCellRescaling
from frost.runtime import (DriftPosterior, Executor, KineticBias, LocalMagnitude, Magnitude, NonFinite, QoSThreshold,
                           ResonancePrior, StructureClause, ThermostatBias)
from frost.runtime.md import MDBackend, build
from frost.gpu_guard import ContentionWatch
from frost.runtime.store import config_key, load as store_load, save as store_save, warm_start

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "results" / "e13"


def gpu_energy_j() -> float | None:
    """Cumulative energy of every GPU, joules (NVML's counter; None if unavailable)."""
    try:
        import pynvml

        pynvml.nvmlInit()
        # NVML numbers GPUs physically and ignores CUDA_VISIBLE_DEVICES: count only this job's GPUs, or the other lane's
        # job is billed to this one (the 2026-10-20 runtime runs read 5,123 and 8,779 J/ps for the same Cu teacher)
        import os

        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        ids = [int(i) for i in visible.split(",") if i.strip().isdigit()] or list(range(pynvml.nvmlDeviceGetCount()))
        return sum(pynvml.nvmlDeviceGetTotalEnergyConsumption(pynvml.nvmlDeviceGetHandleByIndex(i)) for i in ids) / 1e3
    except Exception:  # noqa: BLE001  (no NVML: report no energy rather than fail the run)
        return None
PERIOD = {"cu500": 113.7, "si512": 78.7, "silica1152": 27.7, "silica_glass1152": 28.4, "li3po4": 33.0}


def period_of(run: str) -> float:
    for key in sorted(PERIOD, key=len, reverse=True):
        if run.startswith(key):
            return PERIOD[key]
    return 0.0


class Frames:
    """Velocities every ``v_fs`` and positions every ``x_fs``, kept on the GPU (no sync in the timed loop) and keyed
    by the simulated step, so a rollback's redo overwrites what the undone attempt recorded."""

    def __init__(self, dt_fs: float, v_fs: float = 2.0, x_fs: float = 20.0, cell_of=None):
        self.v_every, self.x_every = max(1, round(v_fs / dt_fs)), max(1, round(x_fs / dt_fs))
        self.v_fs, self.x_fs = self.v_every * dt_fs, self.x_every * dt_fs
        self.v, self.x = {}, {}
        # NPT: the cell of every position frame (unwrapped positions need the cell they were written in: a distance
        # taken with another cell is wrong across the boundary, as in trap 28)
        self.cell_of, self.cells = cell_of, {}

    def __call__(self, step: int, state) -> None:
        if step % self.v_every == 0:
            self.v[step] = state.v.detach().float().clone()
        if step % self.x_every == 0:
            self.x[step] = state.x.detach().float().clone()
            if self.cell_of is not None:
                self.cells[step] = np.array(self.cell_of(), dtype=np.float64)

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        return (torch.stack([self.v[k] for k in sorted(self.v)]).cpu().numpy(),
                torch.stack([self.x[k] for k in sorted(self.x)]).cpu().numpy())

    def cells_array(self):
        return np.stack([self.cells[k] for k in sorted(self.cells)]) if self.cells else None


class QTrace:
    """Interface pinning's order parameter Q every ``every_fs`` (frost.pinning), on the GPU, keyed by the simulated step
    so a rollback's redo overwrites the undone attempt's samples, as ``Frames`` does."""

    def __init__(self, pin, dt_fs: float, every_fs: float = 10.0):
        self.pin, self.dt = pin, dt_fs
        self.every = max(1, round(every_fs / dt_fs))
        self.q = {}

    def __call__(self, step: int, state) -> None:
        if step % self.every == 0:
            self.q[step] = self.pin.q(state.x)

    def summary(self, skip_fs: float) -> dict:
        from frost.pinning import block_se

        steps = sorted(self.q)
        t = np.array(steps, dtype=np.float64) * self.dt
        v = torch.stack([self.q[s] for s in steps]).double().cpu().numpy()
        keep = v[t >= skip_fs]
        if len(keep) < 20:
            return {"q_trace": [(float(a), float(c)) for a, c in zip(t, v, strict=True)]}
        m = float(keep.mean())
        every = max(1, len(v) // 500)
        return {"q_mean": m, "q_sd": float(keep.std()), "q_se_block": block_se(keep), "gap": self.pin.a - m,
                "kappa": self.pin.kappa, "a": self.pin.a, "skip_fs": skip_fs,
                "q_trace": [(float(a), float(c)) for a, c in zip(t[::every], v[::every], strict=True)]}


def energy_blocks(trace, width_fs: float = 1000.0) -> list:
    """[(t_ps, mean energy in meV/atom)] in blocks of ``width_fs``: 1 ps is the statistic DriftMonitor.sequential decides
    on; 0.1 ps is saved too, so a finer-block detector can be replayed on real runs before it is adopted (gaps.md E.8)."""
    out = {}
    for t, e in trace:
        out.setdefault(int(t // width_fs), []).append(e)
    return [((b + 0.5) * width_fs / 1e3, float(np.mean(v))) for b, v in sorted(out.items())]


units_fs = 0.09822694788464063  # ase.units.fs


def volume_summary(trace, fs: float) -> dict:
    """Mean volume and its relative fluctuation over the second half (the first half equilibrates the barostat)."""
    vols = np.array([v for t, v in trace if t >= 0.5 * fs]) if trace else np.array([])
    if len(vols) < 4:
        return {}
    return {"volume_mean_A3": float(vols.mean()), "volume_rel_sd": float(vols.std() / vols.mean()),
            "volume_trace": [(float(t), float(v)) for t, v in trace[:: max(1, len(trace) // 400)]]}


def drift_summary(trace) -> dict:
    """Slope of an energy trace [(t fs, meV/atom)] over the whole run and over each half, meV/atom/ps (G3's shape)."""
    t = np.array([a for a, _ in trace]) / 1e3
    e = np.array([b for _, b in trace])
    if len(t) < 4:
        return {}
    h = len(t) // 2
    slope = lambda a, b: float(np.polyfit(a, b, 1)[0])  # noqa: E731
    return {"eff_drift_meV_per_atom_per_ps": slope(t, e), "eff_drift_halves": [slope(t[:h], e[:h]), slope(t[h:], e[h:])],
            "eff_change_meV_per_atom": float(e[-1])}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--student", required=True)
    ap.add_argument("--compile", default="", help="TorchInductor mode for a descriptor student")
    ap.add_argument("--start", type=int, default=9000)
    ap.add_argument("--fs", type=int, default=10000)
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--ensemble", choices=("nve", "nvt", "npt"), default="nve")
    ap.add_argument("--pressure-bar", type=float, default=1.0, help="npt: target pressure")
    ap.add_argument("--tau-p", type=float, default=1000.0, help="npt: barostat time constant, fs")
    ap.add_argument("--compressibility", type=float, default=1e-5,
                    help="npt: the barostat's compressibility estimate, 1/bar (sets the relaxation rate only)")
    ap.add_argument("--baro-every", type=int, default=10, help="npt: steps between barostat moves on the teacher arm")
    ap.add_argument("--npt-equil-fs", type=float, default=0.0,
                    help="npt: equilibrate the start under NPT with the teacher (untimed) and start every arm from there, so "
                         "no arm carries the barostat's start-up (a one-way volume change the bias clause reads as heat)")
    ap.add_argument("--npt-axes", default="xyz",
                    help="npt: xyz isotropic, or the axes barostatted each on its own: z (NP_zAT, a slab), yz (with "
                         "--strain-rate: uniaxial stress)")
    ap.add_argument("--strain-rate", type=float, default=0.0,
                    help="npt mechanics: stretch x at this true strain rate (per ps, e.g. 0.002 = 2e9/s), the --npt-axes "
                         "axes free under the pressure; records the stress-strain curve (tension = -P_xx)")
    ap.add_argument("--pin", default="", help="interface pinning KAPPA:A (eV, dimensionless): U = KAPPA/2 (Q - A)^2 on the "
                    "crystal's in-plane long-range order Q (frost.pinning), on the teacher's and the student's force alike")
    ap.add_argument("--pin-cells", default="", help="pinning: conventional cells across x and y, NX,NY")
    ap.add_argument("--pin-hkl", default="2,0,0;0,2,0", help="pinning: in-plane reflections, h,k,0;...")
    ap.add_argument("--pin-skip-fs", type=float, default=5000.0, help="pinning: equilibration left out of <Q>")
    ap.add_argument("--friction", type=float, default=0.01)
    ap.add_argument("--predictive-arm", action="store_true",
                    help="also run the certified arm with the predictive controller (IntervalController.predictive)")
    ap.add_argument("--nvt-drift", action="store_true",
                    help="nvt: every arm tracks the thermostat's heat and reports effective energies (E - Q) and their "
                         "drift (Integrator.track_heat)")
    ap.add_argument("--nvt-bias-clause", type=float, default=0.0, metavar="TOL",
                    help="with --nvt-drift: certify the temperature bias the shadow work implies, dT = p / (3 gamma k_B), "
                         "under TOL x T (frost.runtime.ThermostatBias); pair with a wider --resonance-fraction, which "
                         "then bounds the search as it does under NVE's drift clause")
    ap.add_argument("--epsilon-rel", type=float, default=0.0, metavar="FRAC",
                    help="the magnitude clause in the system's own units: epsilon = FRAC x the teacher's thermal "
                         "acceleration scale (rms per component of F/m at the start frame, meV/(A amu)). 0.043 is the "
                         "oxides' calibrated 2.0 over their 46.4 (silica, glass, Li3PO4), so it leaves them unchanged; "
                         "water's 154 gives 6.6, Cu's 6.5 gives 0.28")
    ap.add_argument("--certified-store", default="", metavar="PATH",
                    help="record the certified arm's final interval, probation rung and ceiling under a hash of the "
                         "configuration (frost.runtime.store); with --warm-start, begin from the matching record")
    ap.add_argument("--warm-start", action="store_true",
                    help="with --certified-store: start the certified arm from the last record of this exact "
                         "configuration (the clauses still judge from an empty record; the arm's JSON names the source)")
    ap.add_argument("--small-n-allowance", action="store_true",
                    help="the magnitude clause's per-interval threshold carries the scatter allowance of an RMS over 3N "
                         "components, normalised to 500 atoms (Magnitude.n_atoms): for small molecules, whose single "
                         "intervals scatter far more than a 500-atom cell's")
    ap.add_argument("--predictive-admission", action="store_true",
                    help="probation: the bias clause's measured margin at an admitted rung sets the next rung "
                         "(k / rho^(1/3), within x1.25-x2) instead of a fixed x1.25 (ThermostatBias.predictive)")
    ap.add_argument("--local-clause", type=int, default=0, metavar="N",
                    help="also hold every one of N x N x N regions to the magnitude bound (frost.locality: the worst "
                         "region's RMS over its allowance 1 + z / sqrt(6 m)); for local physics (defects, reactions, "
                         "cascades, interfaces) that the whole-cell RMS dilutes. 0: off")
    ap.add_argument("--local-z", type=float, default=4.0, help="the local clause's allowance, 1 + z / sqrt(6 m)")
    ap.add_argument("--structure-clause", action="store_true",
                    help="growth past the 0.22 prior only when the run's own pair-distance histogram at the new interval "
                         "matches the one recorded at the prior (frost.runtime.StructureClause); with the NVT clauses")
    ap.add_argument("--nvt-kinetic-clause", type=float, default=0.0, metavar="TOL",
                    help="nvt: certify the step-averaged kinetic temperature against the target, within TOL x T "
                         "(frost.runtime.KineticBias; noise-limited below thousands of atoms: see its docstring)")
    ap.add_argument("--nvt-drift-clause", action="store_true",
                    help="with --nvt-drift: the drift clause also certifies, on NVE's budget (too strict under a strong "
                         "bath: the 3500 K ramp at 10/ps refuses down to k = 1.4)")
    ap.add_argument("--thermal-epsilon", action="store_true",
                    help="the magnitude bound in thermal units: epsilon x sqrt(T_target / 300 K), 300 K being where it was "
                         "calibrated (ramps; runs at other temperatures)")
    ap.add_argument("--no-drift-clause", action="store_true",
                    help="ablation only: drop the NVE energy-drift posterior from the certificate")
    ap.add_argument("--drift-rewind", action="store_true",
                    help="with --drift-sequential: a drift refusal rewinds the trajectory to where the refused interval was "
                         "adopted (Executor.segment), so the energy it spent never stays in the run (gaps.md E.8)")
    ap.add_argument("--fine-lattice", type=float, default=0.0,
                    help="FS: every step count below FS femtoseconds is an allowed interval (the even lattice above)")
    ap.add_argument("--thermostat", choices=("langevin", "nhc"), default="langevin",
                    help="with --ensemble nvt: Langevin (BAOAB) or a Nose-Hoover chain (frost.integrators.NoseHooverChain)")
    ap.add_argument("--nhc-tau", type=float, default=100.0, help="Nose-Hoover chain period, fs")
    ap.add_argument("--drift-step", action="store_true",
                    help="with --drift-rewind: after a drift refusal step down by the measured excess, k x (allowed / "
                         "measured)^(1/1.5), between half and one rung, instead of halving (gaps.md E.8, AgI)")
    ap.add_argument("--drift-horizon-ps", type=float, default=50.0,
                    help="the length the NVE energy budget is spent over (G3: 50 ps). A nanosecond run judged per 50 ps "
                         "window may gain 20 meV/atom per ns (Li3PO4 1 ns gained 8.0, +24 K, 2026-11-14): give it its own length")
    ap.add_argument("--drift-budget", type=float, default=1.0, help="meV/atom allowed over --drift-horizon-ps")
    ap.add_argument("--drift-sequential", action="store_true",
                    help="the drift posterior's group-sequential looks from 7 ps and its 50 ps run ledger "
                         "(DriftMonitor.sequential, gaps.md E.9) instead of one decision per 20 ps of an interval")
    ap.add_argument("--ramp", default="", help="T1:PS, heat the thermostat linearly to T1 over PS ps (nvt)")
    ap.add_argument("--mode", choices=("sync", "spec", "mid", "lag"), default="sync",
                    help="sync: boundary r-RESPA; spec: speculative refresh; mid: the residual at the interval's midpoint; "
                         "lag: the teacher on a second GPU one interval behind (nvt; refusals latch, no rollback)")
    ap.add_argument("--teacher-device", default="1", help="--mode lag: the CUDA device the teacher process sees")
    ap.add_argument("--teacher-path", choices=("graph", "eager", "auto"), default="graph",
                    help="the teacher's call path for every arm; auto times both on the start frame and keeps the faster")
    ap.add_argument("--resonance-fraction", type=float, default=0.22,
                    help="the prior's cap, k dt <= fraction x fastest period (0.22: the boundary form's measured margin)")
    ap.add_argument("--k0", type=int, default=2, help="adaptive arm's starting interval")
    ap.add_argument("--k-max", type=int, default=64)
    ap.add_argument("--fixed-k", default="", help="comma list of fixed intervals to run as extra arms")
    ap.add_argument("--monitor", default="", help="green:T, a Green-style arm")
    ap.add_argument("--no-rollback", action="store_true")
    ap.add_argument("--fixed-uncertified", action="store_true",
                    help="fixed arms run with no clause at all (a fixed schedule, as DMTS and switching run)")
    ap.add_argument("--no-teacher", action="store_true", help="skip the baseline arm (no speedup)")
    ap.add_argument("--teacher-fs", type=int, default=0, help="baseline arm length (default --fs)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tile", type=int, default=1, help="supercell of the start frame, tile^3 copies (timing at scale)")
    ap.add_argument("--edge-forces", action="store_true",
                    help="both arms' teacher through mace's edge-force output (same forces, 1.41-1.47x faster)")
    ap.add_argument("--never-lose", action="store_true",
                    help="adaptive arm: the teacher itself when the smallest multirate interval fails, with cheap probes")
    ap.add_argument("--hmr", type=float, default=0.0, help="hydrogen mass repartitioning, Da added to each H")
    ap.add_argument("--period-fs", type=float, default=0.0,
                    help="fastest period from the teacher's spectrum, overriding the table (needed with --hmr)")
    ap.add_argument("--frames", action="store_true", help="keep every arm's frames for score_frames.py")
    ap.add_argument("--frames-x-fs", type=float, default=20.0,
                    help="fs between saved positions (2026-11-17: a Green-Kubo viscosity needs the stress every few fs, "
                         "which experiments/e13_runtime/viscosity.py recomputes from the positions with the teacher)")
    ap.add_argument("--frames-v-fs", type=float, default=2.0,
                    help="velocity frame stride, fs (default 2, what the VDOS needs; a structure-only run passes a "
                         "large value: velocities are 90%% of a frames file)")
    ap.add_argument("--teacher-ext", default="",
                    help="a teacher served from its own environment (frost.external_teacher), e.g. pet:pet-omatpes-v2 "
                         "or uma:uma-m-1p1:omat; the baseline arm calls it the same way")
    ap.add_argument("--tight-teacher", action="store_true",
                    help="filter the teacher's skin list to r_max before every call (the skin's pairs cost MACE compute "
                         "for nothing: teacher_list_cost.py); both the teacher arm and Frost's arms use it")
    ap.add_argument("--neighbours", choices=("vesin", "alchemi", "brute", "auto"), default="alchemi",
                    help="list builder for teacher and student; auto: the all-pairs search for small orthorhombic cells "
                         "(frost.neighbours.edges_brute), the GPU cell list otherwise")
    ap.add_argument("--lagged-lists", action="store_true",
                    help="the student's list decides each step from the previous step's displacement, copied to the host "
                         "without blocking (frost.neighbours lag_margin 0.05 A): no per-step host sync in the inner loop")
    ap.add_argument("--no-derived-lists", action="store_true",
                    help="the student's own Verlet list (needed when its cutoff plus skin exceeds the teacher's cutoff)")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)

    b = build(args.run, args.student, start=args.start, dt_fs=args.dt, ensemble=args.ensemble,
              friction=args.friction, seed=args.seed, compile_student=args.compile, lagged_lists=args.lagged_lists,
              tile=args.tile,
              thermostat=args.thermostat, nhc_tau_fs=args.nhc_tau, npt=args.ensemble == "npt",
              edge_forces=args.edge_forces, hmr=args.hmr, derived_lists=not args.no_derived_lists,
              teacher_ext=args.teacher_ext, teacher_path=args.teacher_path, neighbours=args.neighbours,
              tight_teacher=args.tight_teacher)
    n, masses = b["n"], b["masses"]
    pin = None
    if args.pin:
        from frost.pinning import Pinning, inplane_kvectors

        kappa, a_pin = (float(v) for v in args.pin.split(":"))
        hkl = [tuple(int(c) for c in t.split(",")) for t in args.pin_hkl.split(";")]
        cells_xy = tuple(int(c) for c in args.pin_cells.split(","))
        pin = Pinning(inplane_kvectors(b["cell0"], cells_xy, hkl), kappa, a_pin, n, device=b["x0"].device)
        t_force, s_force = b["teacher_force"], b["student_force"]

        def pinned(force):
            def call(x):
                e, f, x = force(x)
                e_b, f_b, _ = pin(x)
                return e + e_b, f + f_b, x
            return call

        # the bias on both: the held residual F_T - F_S is unchanged and the bias is exact on every inner step
        b["teacher_force"], b["student_force"] = pinned(t_force), pinned(s_force)
        print(f"pinning: kappa {kappa} eV, a {a_pin}, Q at the start {float(pin.q(b['x0'])):.4f}", flush=True)
    epsilon = 2.0
    if args.epsilon_rel:
        if args.ramp:
            raise SystemExit("--epsilon-rel takes the scale from the start frame; a ramp changes it")
        _, f0, _ = b["teacher_force"](b["x0"].clone())
        a_scale = float(1e3 * ((f0 / masses) ** 2).mean().sqrt())  # rms per component, meV/(A amu)
        epsilon = args.epsilon_rel * a_scale
        print(f"thermal acceleration scale {a_scale:.2f} meV/(A amu): epsilon {epsilon:.3f}", flush=True)
    t_start = b["temperature"]
    ramp = None
    if args.ramp:
        t1, ps = (float(x) for x in args.ramp.split(":"))
        ramp = (t1, ps)
        if args.ensemble != "nvt":
            raise SystemExit("--ramp needs --ensemble nvt")

    target = {"T": t_start}  # the thermostat's current target, read by --thermal-epsilon

    def set_temperature(integ, t_fs):
        if ramp is None or integ.thermostat is None:
            return
        temp = t_start + (ramp[0] - t_start) * min(1.0, t_fs / (1e3 * ramp[1]))
        target["T"] = temp
        if hasattr(integ.thermostat, "set_temperature"):  # a Nose-Hoover chain
            integ.thermostat.set_temperature(temp)
        else:
            integ.thermostat.sigma = torch.sqrt(8.617333262e-5 * temp / masses)

    if args.ensemble == "npt" and args.npt_equil_fs > 0:
        # one NPT-equilibrated start for every arm (untimed): the run's thermostat and barostat on the teacher alone
        st_e = MDState(b["x0"].clone(), b["v0"].clone(), masses)
        integ_e = b["integrator"](st_e)
        b["set_cell"](b["cell0"])
        baro_e = StochasticCellRescaling(args.pressure_bar, args.tau_p, args.compressibility, t_start, args.seed + 101,
                                         axes="xyz" if args.strain_rate else args.npt_axes)  # settle isotropically first
        cell_e = b["cell0"].copy()
        v_start = float(abs(np.linalg.det(cell_e)))
        _, f_e, st_e.x = b["teacher_force"](st_e.x)
        for i in range(round(args.npt_equil_fs / args.dt)):
            _, f_e = integ_e.inner_step(b["teacher_force"], f_e)
            if (i + 1) % args.baro_every == 0:
                vol = float(abs(np.linalg.det(cell_e)))
                k_ax = (0.5 * (masses * st_e.v ** 2).sum(0)).double().cpu().numpy()
                w_t = b["teacher_virial_tensor"]().detach().double().cpu().numpy()
                mu = baro_e.scale_vector(vol, baro_e.pressure_axes(k_ax, w_t, vol), args.baro_every * args.dt * units_fs)
                mu_t = torch.as_tensor(mu, dtype=st_e.x.dtype, device=st_e.x.device)
                st_e.x, st_e.v, cell_e = st_e.x * mu_t, st_e.v / mu_t, cell_e * mu[None, :]
                b["set_cell"](cell_e)
                _, f_e, st_e.x = b["teacher_force"](st_e.x)
        b["x0"], b["v0"], b["cell0"] = st_e.x.detach().clone(), st_e.v.detach().clone(), cell_e.copy()
        b["set_cell"](b["cell0"])
        print(f"NPT equilibration {args.npt_equil_fs:.0f} fs (teacher, untimed): volume {v_start:.1f} -> "
              f"{abs(np.linalg.det(cell_e)):.1f} A^3, T {st_e.temperature():.1f} K", flush=True)
    report = {"run": args.run, "n_atoms": n, "config": vars(args), "arms": {}, "epsilon": epsilon,
              "teacher_path": b["teacher_path"], "teacher_path_ms": b["teacher_path_ms"]}
    if b["teacher_path_ms"]:
        print(f"teacher path {b['teacher_path']} ({b['teacher_path_ms']})", flush=True)
    kept = {}

    # ------------------------------------------------------------- baseline: the teacher every step
    if not args.no_teacher:
        steps = round((args.teacher_fs or args.fs) / args.dt)
        st = MDState(b["x0"].clone(), b["v0"].clone(), masses)
        integ = b["integrator"](st)
        if args.nvt_drift:
            integ.track_heat()
        if args.ensemble == "npt":
            b["set_cell"](b["cell0"])
            baro_t = StochasticCellRescaling(args.pressure_bar, args.tau_p, args.compressibility, t_start, args.seed,
                                             axes=args.npt_axes, deform=("x", args.strain_rate / 1000.0) if args.strain_rate else None)
            cell_t = b["cell0"].copy()
            vol_t = []
            stress_t = []  # (t fs, L_x, L_y, L_z, P_xx, P_yy, P_zz) at each move
        e_now, f, st.x = b["teacher_force"](st.x)
        e_start = float(e_now) + st.kinetic_energy()
        eff = []  # (t fs, effective energy change meV/atom): the teacher's own shadow-work floor
        nve_e = []  # (t fs, energy change meV/atom), NVE
        temps = []
        cellbox = {"c": b["cell0"].copy()}
        frames = (Frames(args.dt, v_fs=args.frames_v_fs, x_fs=args.frames_x_fs, cell_of=(lambda: cellbox["c"]) if args.ensemble == "npt" else None)
                  if args.frames else None)
        qtr_t = QTrace(pin, args.dt) if pin is not None else None
        untimed = []  # (start, end) CUDA events of the barostat's booking calls, left out of the timing
        torch.cuda.synchronize()
        watch = ContentionWatch().start()  # was the arm alone on its GPU? (frost.gpu_guard; NVML, outside the stream)
        j0 = gpu_energy_j()
        t0 = time.perf_counter()
        for i in range(steps):
            if i % 100 == 0:
                set_temperature(integ, i * args.dt)
                temps.append(st.temperature())
                if args.nvt_drift and i:
                    eff.append((i * args.dt, 1e3 * (float(e_now) + st.kinetic_energy() - float(integ.heat) - e_start) / n))
                elif args.ensemble == "nve" and i:  # the teacher's own NVE energy record: the floor a Frost arm is read against
                    nve_e.append((i * args.dt, 1e3 * (float(e_now) + st.kinetic_energy() - e_start) / n))
            e_now, f = integ.inner_step(b["teacher_force"], f)
            if args.ensemble == "npt" and (i + 1) % args.baro_every == 0:
                # the move (isotropic, or z only for NP_zAT) from the teacher's own virial tensor, then a fresh teacher
                # call at the scaled positions: the next kick gets the force it should, and the move's energy is booked
                # exactly (a first-order -W ln mu misses (9 B V / 2)(ln mu)^2, ~0.67 meV/atom/ps on Cu 500 at this
                # barostat's noise, gaps.md NPT results). A production NPT code needs no such call, so its GPU time is
                # measured with CUDA events and left out of the teacher's timing.
                vol = float(abs(np.linalg.det(cell_t)))
                k_ax_t = 0.5 * (masses * st.v ** 2).sum(0)
                k_ax = k_ax_t.double().cpu().numpy()
                w_t = b["teacher_virial_tensor"]().detach().double().cpu().numpy()
                stress_t.append(((i + 1) * args.dt, *np.diag(cell_t).tolist(),
                                 *StochasticCellRescaling.stress_diagonal(k_ax, w_t, vol).tolist()))
                mu = baro_t.scale_vector(vol, baro_t.pressure_axes(k_ax, w_t, vol), args.baro_every * args.dt * units_fs)
                mu_t = torch.as_tensor(mu, dtype=st.x.dtype, device=st.x.device)
                e_before = e_now
                st.x, st.v, cell_t = st.x * mu_t, st.v / mu_t, cell_t * mu[None, :]
                b["set_cell"](cell_t)
                ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                ev0.record()
                e_now, f, st.x = b["teacher_force"](st.x)
                ev1.record()
                untimed.append((ev0, ev1))
                if integ.heat is not None:
                    integ.heat += (k_ax_t.double() * (mu_t ** -2 - 1.0)).sum() + (e_now - e_before)
                cellbox["c"] = cell_t
                vol_t.append(((i + 1) * args.dt, vol * float(np.prod(mu))))
            if frames is not None:
                frames(i + 1, st)
            if qtr_t is not None:
                qtr_t(i + 1, st)
        torch.cuda.synchronize()
        sec = time.perf_counter() - t0 - 1e-3 * sum(a.elapsed_time(z) for a, z in untimed)
        j1 = gpu_energy_j()
        report["arms"]["teacher"] = {"ms_per_fs": 1e3 * sec / (steps * args.dt), "mean_T": float(np.mean(temps)),
                                     "joules_per_ps": ((j1 - j0) / (steps * args.dt / 1e3)) if j0 is not None else None,
                                     "contention": watch.stop()}
        if report["arms"]["teacher"]["contention"].get("contended"):
            print(f"teacher arm SHARED its GPU (foreign {report['arms']['teacher']['contention']['foreign_mib_max']} MiB): "
                  "its timing is not quotable", flush=True)
        if eff:
            report["arms"]["teacher"].update(drift_summary(eff))
        if nve_e:
            t_e = np.array([a for a, _ in nve_e]) / 1e3
            e_e = np.array([b for _, b in nve_e])
            report["arms"]["teacher"].update({"energy_end_meV_per_atom": float(e_e[-1]),
                                              "energy_slope_meV_per_atom_per_ps": float(np.polyfit(t_e, e_e, 1)[0])})
        if args.ensemble == "npt":
            report["arms"]["teacher"].update(volume_summary(vol_t, args.fs))
            if stress_t and args.strain_rate:
                report["arms"]["teacher"]["stress_trace"] = stress_t[:: max(1, len(stress_t) // 800)]
        if qtr_t is not None:
            report["arms"]["teacher"]["pin"] = qtr_t.summary(args.pin_skip_fs)
            print(f"teacher: <Q> {report['arms']['teacher']['pin']['q_mean']:.4f} "
                  f"+- {report['arms']['teacher']['pin']['q_se_block']:.4f}", flush=True)
        print(f"teacher: {report['arms']['teacher']['ms_per_fs']:.2f} ms/fs", flush=True)
        if frames is not None:
            kept["teacher"] = frames

    # ------------------------------------------------------------- Frost arms through the runtime
    remote = None
    if args.mode == "lag":
        if args.ensemble != "nvt" or args.teacher_ext:
            raise SystemExit("--mode lag is the thermostatted contract on our MACE teacher (--ensemble nvt)")
        from frost.async_teacher import LaggedResidual, MaceTeacherSpec, ProcessTeacher

        remote = ProcessTeacher(MaceTeacherSpec(b["atoms"], b["cell"], 1.0, "alchemi", teacher=b["teacher_name"],
                                                edge_forces=b["edge_forces"], tight=args.tight_teacher),
                                visible_device=args.teacher_device)
        for _ in range(5):  # warm the remote teacher (graph capture, first kernels) outside every timed arm, as the
            remote(b["x0"])  # baseline's teacher is warmed before its loop: the first lag arm otherwise waited 14.8 s
    arms = [("adaptive", 0)] + ([("predictive", 0)] if args.predictive_arm else []) + \
        [(f"fixed{k}", int(k)) for k in args.fixed_k.split(",") if k]
    if args.monitor:
        arms.append((args.monitor, 0))
    for name, k_fixed in arms:
        st = MDState(b["x0"].clone(), b["v0"].clone(), masses)
        integ = b["integrator"](st)
        if args.nvt_drift:
            integ.track_heat()
        temps = {}  # keyed by the simulated step, so a rollback's redo overwrites the undone attempt's sample
        holder = {}
        frames = (Frames(args.dt, v_fs=args.frames_v_fs, x_fs=args.frames_x_fs, cell_of=(lambda: holder["backend"].cell) if args.ensemble == "npt" else None)
                  if args.frames else None)

        qtr = QTrace(pin, args.dt) if pin is not None else None

        def on_step(backend, integ=integ, temps=temps, frames=frames, qtr=qtr):
            if backend.cursor % 100 == 0:  # the step actually simulated (rollbacks restore it)
                set_temperature(integ, backend.cursor * args.dt)
                temps[backend.cursor] = backend.state.temperature()
            if frames is not None:
                frames(backend.cursor, backend.state)
            if qtr is not None:
                qtr(backend.cursor, backend.state)

        lagged = LaggedResidual(remote, torch.zeros_like(b["x0"])) if remote is not None else None
        structure = ({"list": b["s_list"], "r_max": float(b["student_config"]["cutoff"]), "bins": 90,
                      "every": max(1, round(20.0 / args.dt))} if args.structure_clause else None)
        barostat = None
        if args.ensemble == "npt":
            b["set_cell"](b["cell0"])
            barostat = {"baro": StochasticCellRescaling(args.pressure_bar, args.tau_p, args.compressibility, t_start, args.seed,
                                                        axes=args.npt_axes, deform=("x", args.strain_rate / 1000.0) if args.strain_rate else None),
                        "virial": b["teacher_virial"], "virial_tensor": b["teacher_virial_tensor"], "set_cell": b["set_cell"],
                        "cell": b["cell0"]}
        certified_arm = not name.startswith("green:") and not (k_fixed and args.fixed_uncertified)
        local = ({"regions": args.local_clause, "z": args.local_z, "cell": b["cell0"]}
                 if args.local_clause and certified_arm and not args.no_rollback else None)
        backend = MDBackend(st, integ, b["student_force"], b["teacher_force"], args.dt, n, on_step=on_step, barostat=barostat,
                            lagged=lagged, track_temperature=bool(args.nvt_kinetic_clause), structure=structure,
                            local=local)
        holder["backend"] = backend
        if name.startswith("green:"):
            clauses = [QoSThreshold(threshold=float(name.split(":")[1])), NonFinite()]
        elif k_fixed and args.fixed_uncertified:
            clauses = []  # a fixed interval and no certificate: the DMTS / switching baseline (B1)
        else:
            clauses = [ResonancePrior(period_fs=(args.period_fs or period_of(args.run)), dt_fs=args.dt,
                                      fraction=args.resonance_fraction), NonFinite()]
            if not args.no_rollback:
                # epsilon (2.0 meV/(A amu)) was calibrated at 300 K; thermal forces grow as sqrt(T), so the bound follows
                # the thermostat target from 300 K: a ramp from 300 K is unchanged, a run that starts hot is scaled from
                # its first step (Li3PO4 at 1000 K refused every interval under the 300 K bound, gaps.md E.7 L6)
                scale = (lambda: math.sqrt(target["T"] / 300.0)) if args.thermal_epsilon else None
                clauses.insert(0, Magnitude(epsilon=epsilon, scale=scale, n_atoms=n if args.small_n_allowance else None))
                if local is not None:
                    clauses.insert(1, LocalMagnitude(epsilon=epsilon, scale=scale))
            if remote is not None:  # a call in flight cannot be recalled: every clause latches instead of rewinding
                for c in clauses:
                    c.kind = "refuse" if c.kind == "rollback" else c.kind
            if (args.ensemble == "nve" or (args.nvt_drift and args.nvt_drift_clause)) and not args.no_drift_clause:
                drift_clause = DriftPosterior(monitor=DriftMonitor(sequential=args.drift_sequential,
                                                                   horizon_ps=args.drift_horizon_ps,
                                                                   gain_budget=args.drift_budget),
                                              rewind=args.drift_rewind and args.drift_sequential,
                                              measured_step=args.drift_step and args.drift_rewind)
                clauses.append(drift_clause)
            if args.nvt_bias_clause:
                if args.ensemble not in ("nvt", "npt") or not args.nvt_drift:
                    raise SystemExit("--nvt-bias-clause needs --ensemble nvt or npt and --nvt-drift (the heat must be tracked)")
                clauses.append(ThermostatBias(friction_per_ps=1e3 * args.friction,
                                              temperature_K=ramp[0] if ramp else t_start, tolerance=args.nvt_bias_clause,
                                              predictive=args.predictive_admission))
            if args.structure_clause:
                prior = IntervalController(k_max=args.k_max, period_fs=(args.period_fs or period_of(args.run)) or None,
                                           dt_fs=args.dt, resonance_fraction=0.22).resonance_cap()
                clauses.append(StructureClause(k_ref=prior))
            if args.nvt_kinetic_clause:
                if args.ensemble != "nvt" or ramp:
                    raise SystemExit("--nvt-kinetic-clause needs --ensemble nvt at a fixed target (no --ramp)")
                clauses.append(KineticBias(temperature_K=t_start, tolerance=args.nvt_kinetic_clause))
        # --small-n-allowance: the controller's single-interval shrink test carries the clause's allowance (one bound)
        allowance = Magnitude(epsilon=epsilon, n_atoms=n).allowance() if args.small_n_allowance else 1.0
        ctrl = IntervalController(k=args.k0, k_max=args.k_max, epsilon=epsilon, fine_fs=args.fine_lattice,
                                  interval_allowance=allowance,
                                  epsilon_scale=(lambda: math.sqrt(target["T"] / 300.0)) if args.thermal_epsilon else None,
                                  period_fs=(args.period_fs or period_of(args.run)) or None,
                                  dt_fs=args.dt, resonance_fraction=args.resonance_fraction,
                                  predictive=(name == "predictive"))
        nvt_measured = [c for c in clauses if isinstance(c, (ThermostatBias, KineticBias, StructureClause))]
        if nvt_measured and not k_fixed:
            # growth past the NVE-validated prior (0.22 of the fastest period) is probationary: one rung at a time,
            # each admitted by every measured NVT clause (IntervalController.probation)
            ctrl.start_probation(IntervalController(k_max=args.k_max, period_fs=(args.period_fs or period_of(args.run)) or None,
                                                    dt_fs=args.dt, resonance_fraction=0.22,
                                                    fine_fs=args.fine_lattice).resonance_cap())
            ctrl.admitters = tuple(c.name for c in nvt_measured)
            ctrl.linear_growth = True  # one-rung steps: the growth test scaled to the step (IntervalController)
            for c in nvt_measured:
                c.on_admit = ctrl.admit
        drift = [c for c in clauses if isinstance(c, DriftPosterior) and c.monitor.sequential]
        if drift and not nvt_measured and not k_fixed:
            # NVE: after a drift refusal, growth is probationary, each rung admitted by the drift clause (gaps.md E.9)
            ctrl.probation_on_refusal = True
            ctrl.admitters = (drift[0].name,)
            drift[0].on_admit = ctrl.admit
        warm, ckey, cfg = None, None, None
        if name == "adaptive" and args.certified_store:
            student_path = Path(args.student).resolve()
            cfg = {"run": args.run, "start_T": t_start, "student": str(student_path),
                   "student_mtime": student_path.stat().st_mtime, "teacher": b.get("teacher_name") or args.teacher_ext,
                   "ensemble": args.ensemble, "thermostat": args.thermostat, "friction": args.friction,
                   "nhc_tau": args.nhc_tau, "npt_axes": args.npt_axes, "dt": args.dt, "hmr": args.hmr,
                   "period": args.period_fs or period_of(args.run), "epsilon": epsilon,
                   "resonance_fraction": args.resonance_fraction, "fine_lattice": args.fine_lattice,
                   "k_max": args.k_max, "bias": args.nvt_bias_clause, "structure": args.structure_clause,
                   "local": args.local_clause, "predictive": args.predictive_admission, "mode": args.mode}
            ckey = config_key(cfg)
            if args.warm_start:
                rec = store_load(args.certified_store, ckey)
                if rec is not None:
                    warm = {"from": rec.get("tag"), "saved": rec.get("saved"), "applied": warm_start(ctrl, rec)}
                    print(f"adaptive: warm start from {rec.get('tag')} ({rec.get('saved')}): {warm['applied']}", flush=True)
                else:
                    print("adaptive: no certified record for this configuration; cold start", flush=True)
        ex = Executor(backend, clauses, controller=ctrl, mode=args.mode, k=k_fixed,
                      never_lose=args.never_lose and name in ("adaptive", "predictive") and remote is None)
        torch.cuda.synchronize()
        watch = ContentionWatch().start()
        j0 = gpu_energy_j()
        res = ex.run(args.fs)
        j1 = gpu_energy_j()
        contention = watch.stop()
        if contention.get("contended"):
            print(f"{name} arm SHARED its GPU (foreign {contention['foreign_mib_max']} MiB): its timing is not quotable",
                  flush=True)
        if lagged is not None:
            lagged.drain()  # after the timed loop: the last call's collection is teardown, not the loop (trap 23)
        log = ex.log
        trace = [e for _, _, _, e in log if e is not None]
        arm = {"contention": contention, "ms_per_fs": 1e3 * res["seconds"] / max(backend.time(), 1e-9), "reached_fs": backend.time(),
               "mean_k": res["mean_k"], "teacher_calls": backend.teacher_calls, "events": res["events"][:200],
               "baseline_steps": res["baseline_steps"], "student_steps": backend.student_steps,
               "cap_bound": res.get("cap_bound", 0),  # intervals where a prior capped the controller's k
               "warm_start": warm,
               "n_rollbacks": sum(e["action"] == "rollback" for e in res["events"]),
               "n_refusals": sum(e["action"] == "refuse" for e in res["events"]),
               # the first 200 events are mostly rollbacks; refusals, fallbacks and resumes are what a trace needs
               "key_events": [e for e in res["events"] if e["action"] != "rollback"][:2000],
               "event_counts": {f"{c}:{a}": n for (c, a), n in sorted(
                   Counter((e["clause"], e["action"]) for e in res["events"]).items())},
               "stopped": any(e["action"] == "stop" for e in res["events"]),
               "staleness_mean": float(np.mean([s for _, _, s, _ in log])) if log else None,
               "staleness_max": float(np.max([s for _, _, s, _ in log])) if log else None,
               "energy_end_meV_per_atom": trace[-1] if trace else None,
               # 1 ps block means of the monitored energy (validate.py's trace), so a new drift rule replays on this run
               "energy_trace_1ps": energy_blocks([(t, e) for t, _, _, e in log if e is not None]),
               "energy_trace_100fs": energy_blocks([(t, e) for t, _, _, e in log if e is not None], 100.0),
               "k_trace": [(t, k) for t, k, _, _ in log[:: max(1, len(log) // 400)]],
               "mean_T": float(np.mean(list(temps.values()))) if temps else None,
               "T_trace": [(c * args.dt, float(temps[c])) for c in sorted(temps)],
               "joules_per_ps": ((j1 - j0) / max(backend.time() / 1e3, 1e-9)) if j0 is not None else None}
        if args.nvt_drift:
            arm.update(drift_summary([(t, e) for t, _, _, e in log if e is not None]))
        if args.ensemble == "npt":
            arm.update(volume_summary(backend.volume_trace, args.fs))
            if args.strain_rate and getattr(backend, "stress_trace", None):
                arm["stress_trace"] = backend.stress_trace[:: max(1, len(backend.stress_trace) // 800)]
        if lagged is not None:
            arm["teacher_wait_s"] = lagged.waited_s  # the main loop blocked on the second GPU (0: fully hidden)
        for c in clauses:
            if isinstance(c, StructureClause):
                arm["structure"] = {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in c.last.items()}
        if ctrl.probation is not None:
            arm["probation_final"] = ctrl.probation
        if qtr is not None:  # interface pinning: this arm's own <Q> (the first two melting scans lacked it, 2026-11-14)
            arm["pin"] = qtr.summary(args.pin_skip_fs)
        if "teacher" in report["arms"]:
            arm["speedup"] = report["arms"]["teacher"]["ms_per_fs"] / arm["ms_per_fs"]
        report["arms"][name] = arm
        if frames is not None:
            kept[name] = frames
        if ckey is not None and not arm["stopped"] and backend.time() >= 0.99 * args.fs:
            store_save(args.certified_store, ckey, {"config": cfg, "tag": args.tag, "run": args.run, "k": int(ctrl.k),
                                                    "probation": ctrl.probation, "ceiling": ctrl.ceiling,
                                                    "fs": backend.time(), "speedup": arm.get("speedup")})
        if "q_mean" in arm.get("pin", {}):
            print(f"{name}: <Q> {arm['pin']['q_mean']:.4f} +- {arm['pin']['q_se_block']:.4f}", flush=True)
        print(f"{name}: {arm['ms_per_fs']:.2f} ms/fs" + (f" ({arm['speedup']:.2f}x)" if "speedup" in arm else "")
              + f" | reached {arm['reached_fs'] / 1e3:.1f} ps | mean k {arm['mean_k']:.1f} | rollbacks "
              f"{arm['n_rollbacks']} refusals {arm['n_refusals']} | staleness {arm['staleness_mean']:.3f} mean "
              f"{arm['staleness_max']:.3f} max | energy end {arm['energy_end_meV_per_atom']}"
              + (f" | effective drift {arm['eff_drift_meV_per_atom_per_ps']:.4f} meV/atom/ps"
                 if "eff_drift_meV_per_atom_per_ps" in arm else "")
              + (f" | capped by a prior {arm['cap_bound']} times" if arm.get("cap_bound") else ""), flush=True)
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"md_{args.run}_{args.tag}.json").write_text(json.dumps(report, indent=1))
    if remote is not None:
        remote.close()
    if kept:
        import h5py

        with h5py.File(ROOT / "data" / "trajectories" / f"{args.run}.h5", "r") as fh:
            numbers, cell = np.asarray(fh.attrs["numbers"], dtype=np.int64), np.asarray(fh.attrs["cell"], dtype=np.float64)
        if args.tile > 1:
            numbers, cell = np.tile(numbers, args.tile ** 3), cell * args.tile
        arrays = {}
        for name, fr in kept.items():
            arrays[f"{name}__v"], arrays[f"{name}__x"] = fr.arrays()
            if fr.cells_array() is not None:
                arrays[f"{name}__cells"] = fr.cells_array()
        first = next(iter(kept.values()))
        np.savez(OUT / f"frames_{args.run}_{args.tag}.npz", numbers=numbers, cell=cell,
                 masses=masses.squeeze(-1).cpu().numpy(), v_fs=first.v_fs, x_fs=first.x_fs, **arrays)
        print(f"-> {OUT / f'frames_{args.run}_{args.tag}.npz'}", flush=True)
    print(f"-> {OUT / f'md_{args.run}_{args.tag}.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
