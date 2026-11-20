"""Multirate integrators on GPU tensors, in ASE units (eV, A, amu).

``Integrator.inner_step`` is one velocity Verlet step, or one BAOAB Langevin
step when a thermostat is attached (Leimkuhler and Matthews 2013).
``respa_outer`` is r-RESPA (Tuckerman, Berne and Martyna 1992): a half kick with
the slow force, then k inner steps of the fast force. The caller evaluates the
slow force at the new positions, since that is the teacher call, and closes the
outer step with a second half kick. With a thermostat this is BAOAB-RESPA (note
section 5.2).

Force callables take positions and return (energy, forces, positions). The
returned positions are the ones to continue from; they differ from the input only
when a neighbour-list rebuild wrapped them into the cell.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from ase import units

ForceFn = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]


@dataclass
class MDState:
    x: torch.Tensor
    v: torch.Tensor
    masses: torch.Tensor  # (N, 1), amu

    def kinetic_energy(self) -> float:
        return float(0.5 * (self.masses * self.v * self.v).sum())

    def temperature(self) -> float:
        return 2.0 * self.kinetic_energy() / (3 * self.x.shape[0] * units.kB)


class Langevin:
    """Exact Ornstein-Uhlenbeck velocity update, the O in BAOAB."""

    def __init__(self, dt_fs: float, friction_per_fs: float, temperature_K: float,
                 masses: torch.Tensor, seed: int = 0):
        self.c1 = math.exp(-friction_per_fs * dt_fs)
        self.c2 = math.sqrt(1.0 - self.c1 * self.c1)
        self.sigma = torch.sqrt(units.kB * temperature_K / masses)
        self.gen = torch.Generator(device=masses.device).manual_seed(seed)

    def __call__(self, v: torch.Tensor) -> torch.Tensor:
        noise = torch.randn(v.shape, generator=self.gen, device=v.device, dtype=v.dtype)
        return self.c1 * v + self.c2 * self.sigma * noise


class NoseHooverChain:
    """A Nose-Hoover chain (Martyna, Klein and Tuckerman 1992) as the middle (O) step of the integrator.

    Deterministic, unlike ``Langevin``: the chain's first friction variable integrates the temperature error, so a steady
    energy injection by the multirate integrator (its shadow work) leaves no lasting temperature offset, as it does under
    Langevin friction (dT = p / (3 gamma k_B), ``frost.runtime.ThermostatBias``); it leaves a constant extra friction
    p / (2 K) on every atom instead (further_expansions_beyond.md 6.4). The chain is propagated over the whole step in two
    half steps of the standard update (Martyna, Tuckerman, Tobias and Klein 1996; Frenkel and Smit, algorithm 31), in the
    position BAOAB gives the O step, which keeps the step symmetric. Masses Q_1 = N_f k_B T tau^2, Q_j = k_B T tau^2.
    The chain's state is checkpointed with the trajectory (``state``, ``restore``), so a rollback rewinds it too.

    On a GPU the update is one kinetic-energy reduction, the chain's scalar arithmetic fused into one kernel by
    torch.compile, and one velocity scale, captured once as a CUDA graph and replayed (2026-11-13): written plainly it is
    ~70 kernel launches on 3-element tensors per step, 1.34 ms per O step on Cu 500 against Langevin's 0.046, which cost
    Frost half its speed (6.0x where Langevin gives ~12x at the same interval). The state (xi, vxi, Q, k_B T) lives in
    fixed buffers updated in place, so the graph, ``restore`` and ``set_temperature`` all act on the same memory; the first
    replay is checked against the eager update, and the test checks the fused path against the plain one.
    """

    gen = None  # no random stream (MDBackend checkpoints a generator only when there is one)

    def __init__(self, dt_fs: float, tau_fs: float, temperature_K: float, masses: torch.Tensor, chain: int = 3,
                 graphed: bool = True):
        self.dt = dt_fs * units.fs
        self.masses = masses
        self.nf = 3 * masses.shape[0] - 3
        self.chain = chain
        self.tau = tau_fs * units.fs
        dev = masses.device
        self.xi = torch.zeros(chain, dtype=torch.float64, device=dev)
        self.vxi = torch.zeros(chain, dtype=torch.float64, device=dev)
        self.Q = torch.zeros(chain, dtype=torch.float64, device=dev)
        self.kT_t = torch.zeros((), dtype=torch.float64, device=dev)
        self.graphed = graphed and dev.type == "cuda"
        self._graph = None
        self._math = torch.compile(self._chain_math, dynamic=False) if self.graphed else self._chain_math
        self.set_temperature(temperature_K)

    def set_temperature(self, temperature_K: float) -> None:
        self.kT = units.kB * temperature_K
        q = [self.kT * self.tau ** 2] * self.chain
        q[0] *= self.nf
        self.Q.copy_(torch.tensor(q, dtype=torch.float64))  # in place: a captured graph reads these buffers
        self.kT_t.fill_(self.kT)

    def state(self):
        return self.xi.clone(), self.vxi.clone()

    def restore(self, st) -> None:
        self.xi.copy_(st[0])
        self.vxi.copy_(st[1])

    def conserved_extra(self) -> float:
        """The chain's own energy: sum Q vxi^2 / 2 + N_f k_B T xi_1 + k_B T sum_{j>1} xi_j (eV)."""
        e = 0.5 * (self.Q * self.vxi * self.vxi).sum() + self.nf * self.kT * self.xi[0] + self.kT * self.xi[1:].sum()
        return float(e)

    def _force(self, j: int, k2):
        if j == 0:
            return (k2 - self.nf * self.kT_t) / self.Q[0]
        return (self.Q[j - 1] * self.vxi[j - 1] ** 2 - self.kT_t) / self.Q[j]

    def _half(self, v: torch.Tensor, h: float) -> torch.Tensor:
        m, last = self.chain - 1, self.chain - 1
        vxi = self.vxi
        k2 = (self.masses * v * v).sum().to(torch.float64)
        vxi[last] = vxi[last] + 0.5 * h * self._force(last, k2)
        for j in range(m - 1, -1, -1):
            s = torch.exp(-0.25 * h * vxi[j + 1])
            vxi[j] = (vxi[j] * s + 0.5 * h * self._force(j, k2)) * s
        scale = torch.exp(-h * vxi[0])
        v = v * scale.to(v.dtype)
        k2 = k2 * scale * scale
        self.xi.add_(h * vxi)
        for j in range(m):
            s = torch.exp(-0.25 * h * vxi[j + 1])
            vxi[j] = (vxi[j] * s + 0.5 * h * self._force(j, k2)) * s
        vxi[last] = vxi[last] + 0.5 * h * self._force(last, k2)
        return v

    def _chain_math(self, k2, vxi, xi, q, kt):
        """One whole O step (two half steps of dt / 2) of the chain alone, functional: the velocity scale it applies and
        the new (vxi, xi). The kinetic energy after the first half is k2 scale^2 exactly, so one reduction serves both
        halves; the order of the updates is ``_half``'s."""
        h = 0.5 * self.dt
        last = self.chain - 1
        vx, x = list(vxi.unbind()), list(xi.unbind())

        def g(j, k):
            return (k - self.nf * kt) / q[0] if j == 0 else (q[j - 1] * vx[j - 1] ** 2 - kt) / q[j]

        total = torch.ones_like(k2)
        for _ in range(2):
            vx[last] = vx[last] + 0.5 * h * g(last, k2)
            for j in range(last - 1, -1, -1):
                s = torch.exp(-0.25 * h * vx[j + 1])
                vx[j] = (vx[j] * s + 0.5 * h * g(j, k2)) * s
            scale = torch.exp(-h * vx[0])
            total = total * scale
            k2 = k2 * scale * scale
            x = [x[j] + h * vx[j] for j in range(self.chain)]
            for j in range(last):
                s = torch.exp(-0.25 * h * vx[j + 1])
                vx[j] = (vx[j] * s + 0.5 * h * g(j, k2)) * s
            vx[last] = vx[last] + 0.5 * h * g(last, k2)
        return total, torch.stack(vx), torch.stack(x)

    def _eager(self, v: torch.Tensor) -> torch.Tensor:
        if not self.graphed:  # CPU and the reference path: the two half steps as written
            v = self._half(v, 0.5 * self.dt)
            return self._half(v, 0.5 * self.dt)
        # GPU: one reduction, the chain's scalar arithmetic fused into one kernel by torch.compile (public API), one scale
        k2 = (self.masses * v * v).sum().to(torch.float64)
        total, vxi, xi = self._math(k2, self.vxi, self.xi, self.Q, self.kT_t)
        self.vxi.copy_(vxi)
        self.xi.copy_(xi)
        return v * total.to(v.dtype)

    def _capture(self, v: torch.Tensor) -> None:
        saved = self.state()
        self._v_in = v.detach().clone()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())  # the inputs were just written there (trap 12)
        with torch.cuda.stream(side):
            for _ in range(2):
                self._eager(self._v_in)
        torch.cuda.current_stream().wait_stream(side)
        self.restore(saved)  # the warm-up advanced the chain
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._v_out = self._eager(self._v_in)
        self.restore(saved)  # capture records the work but does not run it; keep the state exactly as it was
        # check the first replay against the eager update from the same state
        self._v_in.copy_(v)
        self._graph.replay()
        graphed_v, graphed_state = self._v_out.clone(), self.state()
        self.restore(saved)
        eager_v = self._eager(v.detach().clone())
        diff = max(float((graphed_v - eager_v).abs().max()), float((graphed_state[1] - self.vxi).abs().max()))
        self.restore(saved)
        if diff > 1e-10:
            raise RuntimeError(f"graphed Nose-Hoover chain differs from the eager update by {diff:.2e}")

    def __call__(self, v: torch.Tensor) -> torch.Tensor:
        if not self.graphed:
            return self._eager(v)
        if self._graph is None or self._v_in.shape != v.shape or self._v_in.dtype != v.dtype:
            self._capture(v)
        self._v_in.copy_(v)
        self._graph.replay()
        return self._v_out.clone()


class StochasticCellRescaling:
    """Isotropic stochastic cell rescaling (Bernetti and Bussi, JCP 153, 114107, 2020): a first-order barostat with the
    right volume fluctuations, needing only the instantaneous pressure. The log volume moves by
        d eps = -(beta_T / tau_p) (P0 - P - k_B T / V) dt + sqrt(2 k_B T beta_T dt / (V tau_p)) xi,
    and positions, cell and velocities scale by mu = exp(d eps / 3), mu, 1 / mu. In Frost it moves only at interval
    boundaries, where the teacher's exact virial is known (further_expansions_beyond.md 6.1): tau_p (~1 ps) is far longer
    than any interval, so nothing needs to be held. beta_T sets only how fast the volume relaxes, not the ensemble.
    Units: pressure in eV/A^3 (ASE), times in ASE units.
    """

    def __init__(self, pressure_bar: float, tau_fs: float, compressibility_per_bar: float, temperature_K: float,
                 seed: int = 0, axes: str = "xyz", deform: tuple[str, float] | None = None):
        if not axes or any(a not in "xyz" for a in axes) or len(set(axes)) != len(axes):
            raise ValueError(f"axes {axes!r}: 'xyz' (isotropic) or a subset of x, y, z (each axis on its own)")
        if deform is not None and (deform[0] not in "xyz" or (deform[0] in axes and axes != "xyz")):
            raise ValueError(f"deform axis {deform[0]!r} must be one of x, y, z and not barostatted")
        self.p0 = pressure_bar * units.bar
        self.tau = tau_fs * units.fs
        self.beta = compressibility_per_bar / units.bar
        self.kT = units.kB * temperature_K
        self.axes = axes
        # deformation (2026-11-14, mechanics, further_expansions_beyond.md 7.5): one axis stretched at a constant true
        # strain rate (per fs), L(t) = L0 exp(rate t), the barostatted axes free under P0 (uniaxial stress)
        self.deform = deform
        self.gen = torch.Generator().manual_seed(int(seed) + 7919)  # one scalar per move: a CPU stream

    def scale(self, volume: float, pressure: float, dt: float) -> float:
        """The linear factor mu for a move of duration dt from the instantaneous pressure (eV/A^3)."""
        xi = float(torch.randn(1, generator=self.gen, dtype=torch.float64))
        deps = (-(self.beta / self.tau) * (self.p0 - pressure - self.kT / volume) * dt
                + math.sqrt(2.0 * self.kT * self.beta * dt / (volume * self.tau)) * xi)
        return math.exp(deps / 3.0)

    @staticmethod
    def pressure(kinetic: float, virial: float, volume: float) -> float:
        """P = (2 K + W) / (3 V), W = -sum over edges of r . dE/dr (eV/A^3)."""
        return (2.0 * kinetic + virial) / (3.0 * volume)

    # NP_zAT (2026-11-13, melting by interface pinning, further_expansions_beyond.md 7.2): the lateral cell stays fixed
    # (a crystal slab's in-plane lattice) and only L_z moves. With eps = ln V and V = A L_z at fixed A, the stationary
    # distribution in eps has the isotropic form with the internal pressure replaced by the normal stress
    # P_zz = (2 K_zz + W_zz) / V (the work of a change of L_z), and the configurational Jacobian is still V^N (only z
    # coordinates scale), so the same equation moves eps and the whole d eps goes to z: mu = (1, 1, exp(d eps)).
    def scale_vector(self, volume: float, pressure, dt: float) -> np.ndarray:
        """Per-axis factors for a move of duration dt: isotropic (mu, mu, mu), or each barostatted axis on its own,
        exp(d l_a) with d l_a = -(beta_T / tau_p) (P0 - P_aa - k_B T / V) dt + sqrt(2 k_B T beta_T dt / (V tau_p)) xi_a
        (NP_zAT's equation per axis: with V = L_x L_y L_z, each axis's log length has the isotropic form with its own
        normal stress, and its measure contributes one k_B T / V), and a deformed axis exp(rate dt). ``pressure`` is
        the barostatted axis's P (one axis) or the (3,) diagonal (several)."""
        if self.axes == "xyz":
            mu = np.full(3, self.scale(volume, pressure, dt))
        else:
            mu = np.ones(3)
            p = np.atleast_1d(np.asarray(pressure, dtype=np.float64))
            for a in self.axes:
                i = "xyz".index(a)
                p_a = float(p[0]) if p.size == 1 else float(p[i])
                xi = float(torch.randn(1, generator=self.gen, dtype=torch.float64))
                dl = (-(self.beta / self.tau) * (self.p0 - p_a - self.kT / volume) * dt
                      + math.sqrt(2.0 * self.kT * self.beta * dt / (volume * self.tau)) * xi)
                mu[i] = math.exp(dl)
        if self.deform is not None:
            mu["xyz".index(self.deform[0])] = math.exp(self.deform[1] * dt / units.fs)
        return mu

    def pressure_axes(self, kinetic_axes, virial_tensor, volume: float):
        """The pressure that drives this barostat: (2 K + tr W) / (3 V) isotropic, the one barostatted axis's
        (2 K_aa + W_aa) / V, or the (3,) diagonal for several axes.
        kinetic_axes: (K_xx, K_yy, K_zz) = 1/2 sum m v_a^2; virial_tensor: W_ab = -sum_e r_a dE/dr_b (eV)."""
        k = np.asarray(kinetic_axes, dtype=np.float64)
        w = np.asarray(virial_tensor, dtype=np.float64)
        if self.axes == "xyz":
            return (2.0 * k.sum() + np.trace(w)) / (3.0 * volume)
        diag = (2.0 * k + np.diag(w)) / volume
        return float(diag["xyz".index(self.axes)]) if len(self.axes) == 1 else diag

    @staticmethod
    def stress_diagonal(kinetic_axes, virial_tensor, volume: float) -> np.ndarray:
        """P_aa = (2 K_aa + W_aa) / V for every axis (eV/A^3); the tensile stress along a is -P_aa."""
        return (2.0 * np.asarray(kinetic_axes, dtype=np.float64) + np.diag(np.asarray(virial_tensor, dtype=np.float64))) / volume


class BandGLE:
    """A generalised-Langevin O step whose friction is peaked on one frequency band.

    The lagged-teacher pipeline (C2) pumps specific bands: on Li3PO4 +7.2% kinetic
    excess at 30-35 THz and nothing elsewhere (new_directions.md section 12).
    Uniform Langevin friction damps everything, including the slow modes that carry
    transport. This damps mostly the pumped band.

    Each velocity component is extended with two auxiliary momenta, y = (v/sigma,
    s1, s2), evolving as dy = -A y dt + B dW with

        A = [[base,  c,    0  ],
             [ -c,  Gamma, w0 ],
             [  0,  -w0, Gamma]],        B B^T = A + A^T = diag(2 base, 2 Gamma, 2 Gamma).

    The stationary covariance is the identity, so v is Maxwell-Boltzmann at the
    target temperature for any positions, and the configurational distribution is
    canonical. This is the fluctuation-dissipation construction of Ceriotti's
    colored-noise thermostats. The friction felt by v at angular frequency w is

        gamma(w) = base + c^2 Re[(i w + Gamma) / ((i w + Gamma)^2 + w0^2)],

    a band-pass peaked near w0, of half-width about Gamma and height about
    c^2 / (2 Gamma). This is a friction only while the peak is well below the band's
    own width, peak <~ Gamma / 2. Beyond that the oscillator and the auxiliary modes
    hybridise and the damping saturates near Gamma: a band of width Gamma cannot damp
    faster than Gamma (measured: peak 0.01 against Gamma 0.0025 per fs gave 0.0023).
    So a band damping a rate g needs a width of at least 2 g. The O step is exact
    for the linear OU process:
    y <- T y + S xi, with T = exp(-A dt) and S S^T = I - T T^T. All rates are per fs,
    as in ``Langevin``.
    """

    def __init__(self, dt_fs: float, temperature_K: float, masses: torch.Tensor,
                 center_thz: float, width_thz: float, peak_per_fs: float,
                 base_per_fs: float = 0.0, seed: int = 0):
        w0 = 2.0 * math.pi * center_thz * 1e-3  # rad/fs
        gam = 2.0 * math.pi * width_thz * 1e-3  # 1/fs
        c = math.sqrt(2.0 * gam * max(peak_per_fs - base_per_fs, 0.0))
        a = torch.tensor([[base_per_fs, c, 0.0], [-c, gam, w0], [0.0, -w0, gam]],
                         dtype=torch.float64)
        t = torch.linalg.matrix_exp(-a * dt_fs)
        cov = torch.eye(3, dtype=torch.float64) - t @ t.T
        evals, evecs = torch.linalg.eigh(0.5 * (cov + cov.T))
        sqrt_cov = evecs @ torch.diag(evals.clamp(min=0.0).sqrt()) @ evecs.T
        dev = masses.device
        self.T = t.to(dev)
        self.S = sqrt_cov.to(dev)
        self.params = {"center_thz": center_thz, "width_thz": width_thz,
                       "peak_per_fs": peak_per_fs, "base_per_fs": base_per_fs,
                       "c": c, "gamma_aux": gam, "w0": w0}
        self.sigma = torch.sqrt(units.kB * temperature_K / masses)
        self.gen = torch.Generator(device=dev).manual_seed(seed)
        self.aux = None

    def friction(self, freq_thz: float) -> float:
        """gamma(w) per fs at a frequency in THz."""
        p = self.params
        w = 2.0 * math.pi * freq_thz * 1e-3
        z = complex(p["gamma_aux"], w)
        return p["base_per_fs"] + p["c"] ** 2 * (z / (z * z + p["w0"] ** 2)).real

    def __call__(self, v: torch.Tensor) -> torch.Tensor:
        sigma = self.sigma.to(v.dtype)
        if self.aux is None:  # start the auxiliary momenta in equilibrium
            self.aux = torch.randn((*v.shape, 2), generator=self.gen, device=v.device,
                                   dtype=v.dtype)
        y = torch.cat([(v / sigma).unsqueeze(-1), self.aux], dim=-1)  # (N, 3, 3)
        noise = torch.randn(y.shape, generator=self.gen, device=v.device, dtype=v.dtype)
        y = y @ self.T.to(v.dtype).T + noise @ self.S.to(v.dtype).T
        self.aux = y[..., 1:]
        return y[..., 0] * sigma


class Integrator:
    def __init__(self, state: MDState, dt_fs: float, thermostat: Langevin | None = None):
        self.state = state
        self.dt = dt_fs * units.fs
        self.thermostat = thermostat
        self.heat = None  # with track_heat: kinetic energy the O steps added, eV (a device scalar, no sync)

    def track_heat(self) -> None:
        """Accumulate the heat each thermostat step exchanges, so that E - Q is a conserved quantity.

        The O step of BAOAB changes only the kinetic energy, by exactly the heat it exchanges with the bath; the B
        and A steps conserve the total energy up to integration error. So the effective energy E_total - Q is
        conserved by exact dynamics whatever the bath does, and its drift is the shadow work of the discretisation
        (Sivak, Chodera and Crooks, PRX 2013; the effective energy of Bussi, Donadio and Parrinello 2007). It
        measures a thermostatted run's integration error the way the total energy measures an NVE run's.
        """
        self.heat = torch.zeros((), dtype=torch.float64, device=self.state.v.device)

    def kick(self, force: torch.Tensor, duration: float) -> None:
        s = self.state
        s.v = s.v + duration * force.to(s.v.dtype) / s.masses

    def inner_step(self, fast: ForceFn, f_fast: torch.Tensor):
        """One step of the fast force, given that force at the current positions.

        Returns the fast energy and force at the new positions.
        """
        s, h = self.state, self.dt
        self.kick(f_fast, 0.5 * h)
        if self.thermostat is None:
            s.x = s.x + h * s.v
        else:
            s.x = s.x + 0.5 * h * s.v
            v_before = s.v
            s.v = self.thermostat(s.v)
            if self.heat is not None:
                self.heat += 0.5 * (s.masses * (s.v * s.v - v_before * v_before)).sum().to(torch.float64)
            s.x = s.x + 0.5 * h * s.v
        energy, f_new, s.x = fast(s.x)
        self.kick(f_new, 0.5 * h)
        return energy, f_new.to(s.v.dtype)

    def respa_outer(self, fast: ForceFn, f_fast: torch.Tensor, slow: torch.Tensor, k: int,
                    on_step: Callable[[], None] | None = None):
        """Half kick with the slow force, then k inner steps of the fast force.

        Close the outer step with ``kick(slow_new, 0.5 * k * dt)`` once the slow
        force has been evaluated at the new positions. ``on_step`` runs after each
        inner step, which is how an observable gets sampled at the inner rate.
        """
        self.kick(slow, 0.5 * k * self.dt)
        energy = None
        for _ in range(k):
            energy, f_fast = self.inner_step(fast, f_fast)
            if on_step is not None:
                on_step()
        return energy, f_fast

    def respa_midpoint(self, fast: ForceFn, f_fast: torch.Tensor, slow_at, k: int,
                       on_step: Callable[[], None] | None = None):
        """r-RESPA with the slow force evaluated once, at the interval's midpoint.

        The boundary form above measures the residual where the interval starts and
        ends, so half of every kick is applied k*dt/2 away from where it was
        measured. Drift grows steeply with that distance — on silica it is 0.004,
        0.038 and 0.419 meV/atom/ps at k = 4, 8 and 12 — so halving it is worth more
        than it costs, which is nothing: drift k/2 steps, evaluate the slow force
        there, kick once for the whole interval, drift the remaining steps.

        ``slow_at(x, f_fast)`` is called at the midpoint and returns the slow force
        there, plus whatever the caller wants to record at those positions. It is
        handed the fast force already computed at x, so the two halves of a residual
        are evaluated at the same configuration without a second student call. The
        splitting stays symmetric, so the scheme is still time reversible.
        """
        first = k // 2
        energy = None
        for _ in range(first):
            energy, f_fast = self.inner_step(fast, f_fast)
            if on_step is not None:
                on_step()
        slow, extra = slow_at(self.state.x, f_fast)
        self.kick(slow, k * self.dt)
        for _ in range(k - first):
            energy, f_fast = self.inner_step(fast, f_fast)
            if on_step is not None:
                on_step()
        return energy, f_fast, extra

    def respa_interp(self, fast: ForceFn, f_fast: torch.Tensor, slow_start: torch.Tensor,
                     slow_end: torch.Tensor, k: int, on_step: Callable[[], None] | None = None):
        """Non-impulsive multirate with a known end value: the slow force ramps linearly.

        The impulse form kicks with the whole interval's slow force at its two ends, a
        train of impulses whose flat spectrum drives the fast modes at every multiple of
        the interval's frequency; that is what caps the interval near 0.22 of the
        fastest period. ``respa_spread`` removes the impulse but holds the start value
        across the interval, which is forward Euler on the slow force (first order, and
        measured 186x worse). Ramping from the start value to the *end* value applies
        the trapezoid rule continuously instead: second order like the impulse, but the
        forcing is piecewise linear in time, so its harmonics fall as 1/m^2 instead of
        staying flat.

        The end value is the slow force at the interval's far boundary, which is not
        known before the interval is integrated. Speculative refresh supplies it: the
        teacher is evaluated at a *predicted* boundary state and its residual is
        transported to the true one (``validate.py --respa spec``).

        The slow force at inner step j (time j * dt into the interval) is
        slow_start + (j / k) (slow_end - slow_start); each velocity Verlet half kick uses
        the value at its own time, so the scheme is velocity Verlet on a time-dependent
        force. Returns the fast energy and the fast force at the interval's end.
        """
        count = {"j": 0}

        def combined(x):
            count["j"] += 1
            energy, f, x_new = fast(x)
            return energy, f + slow_start + (count["j"] / k) * (slow_end - slow_start), x_new

        energy = None
        f_total = f_fast + slow_start
        for _ in range(k):
            energy, f_total = self.inner_step(combined, f_total)
            if on_step is not None:
                on_step()
        return energy, f_total - slow_end

    def respa_spread(self, fast: ForceFn, f_fast: torch.Tensor, slow: torch.Tensor, k: int,
                     on_step: Callable[[], None] | None = None):
        """Non-impulsive multirate: the frozen slow force acts at every inner step.

        The impulse form concentrates a whole interval's slow force into one kick
        every k steps, which pumps the fast modes: on silica the drift grows as k^3
        (0.004, 0.038, 0.419 at k = 4, 8, 12) while the residual's staleness grows
        only linearly (0.41, 0.70, 0.91), and evaluating the residual at the
        interval's midpoint instead of its boundaries changes nothing. Spreading it
        instead means the integrator always sees F_S + R, which is the teacher's
        force up to the residual's own drift.

        This is not symplectic, so it trades a bounded impulse error for a slow one
        that depends on how fast the residual changes — the quantity E4 measured as
        7-50x slower than the force itself, and the quantity the certificate bounds.
        """
        def combined(x):
            energy, f, x_new = fast(x)
            return energy, f + slow, x_new

        energy = None
        f_total = f_fast + slow
        for _ in range(k):
            energy, f_total = self.inner_step(combined, f_total)
            if on_step is not None:
                on_step()
        return energy, f_total - slow
