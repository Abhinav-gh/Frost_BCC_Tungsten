"""Path-integral core (frost.pimd): exact transforms, contraction identities, and the quantum harmonic oscillator."""

from __future__ import annotations

import math

import torch
from ase import units

from frost.pimd import HBAR, Contraction, NormalModes, RingPolymer


def test_normal_modes_are_orthonormal_and_the_free_ring_conserves_its_energy():
    m = NormalModes(16, 300.0)
    assert torch.allclose(m.C.T @ m.C, torch.eye(16, dtype=torch.float64), atol=1e-12)
    g = torch.Generator().manual_seed(0)
    q, v = torch.randn(16, 3, 3, generator=g, dtype=torch.float64) * 0.1, torch.randn(16, 3, 3, generator=g, dtype=torch.float64)
    mass = 2.0

    def energy(q, v):  # springs 1/2 m wP^2 (q_j - q_j+1)^2 plus kinetic, per unit mass scaled consistently
        return 0.5 * mass * (v * v).sum() + 0.5 * mass * m.omega_p ** 2 * ((q - q.roll(-1, 0)) ** 2).sum()

    e0 = energy(q, v)
    for _ in range(100):
        q, v = m.free_ring(q, v, 0.3 * units.fs)
    assert abs(float(energy(q, v) - e0)) < 1e-9 * float(e0)


def test_contraction_identities():
    full = Contraction(8, 8)
    assert torch.allclose(full.T, torch.eye(8, dtype=torch.float64), atol=1e-12)
    for pc in (1, 3, 4):
        c = Contraction(16, pc)
        const = torch.ones(16, 2, 3, dtype=torch.float64) * 1.7
        assert torch.allclose(c.contract(const), torch.full((pc, 2, 3), 1.7, dtype=torch.float64), atol=1e-12)
        f_c = torch.full((pc, 2, 3), -0.4, dtype=torch.float64)
        assert torch.allclose(c.expand_force(f_c), torch.full((16, 2, 3), -0.4, dtype=torch.float64), atol=1e-12)
        q = torch.randn(16, 2, 3, dtype=torch.float64)
        assert torch.allclose(c.contract(q).mean(0), q.mean(0), atol=1e-12)  # the centroid is kept
    # a smooth path (lowest modes only) is reproduced exactly at the contracted points
    j = torch.arange(16, dtype=torch.float64)
    smooth = torch.cos(2 * math.pi * j / 16).view(16, 1, 1).expand(16, 1, 3).clone()
    c = Contraction(16, 4)
    jc = torch.arange(4, dtype=torch.float64)
    assert torch.allclose(c.contract(smooth)[:, 0, 0], torch.cos(2 * math.pi * jc / 4), atol=1e-12)


def test_quantum_harmonic_oscillator_spread():
    kT = units.kB * 300.0
    hw = 4.0 * kT  # beta hbar omega = 4: strongly quantum
    mass = 1.0
    omega = hw / HBAR
    kspring = mass * omega ** 2
    n, p = 64, 16
    torch.manual_seed(0)
    masses = torch.full((n, 1), mass, dtype=torch.float64)
    q = torch.zeros(p, n, 3, dtype=torch.float64)
    v = torch.zeros(p, n, 3, dtype=torch.float64)
    ring = RingPolymer(q, v, masses, 300.0, dt_fs=0.5, tau0_fs=20.0, seed=1)

    def force(qq):
        return 0.5 * kspring * (qq * qq).sum((1, 2)), -kspring * qq

    _, f = force(ring.q)
    x2 = []
    for step in range(16000):
        _, f = ring.step(force, f)
        if step > 2000 and step % 10 == 0:
            x2.append(float((ring.q ** 2).mean()))
    measured = sum(x2) / len(x2)
    exact = HBAR / (2 * mass * omega) * (1.0 / math.tanh(hw / (2 * kT)))
    classical = kT / kspring
    assert abs(measured / exact - 1.0) < 0.05, (measured, exact, classical)
    assert measured / classical > 1.8  # far from the classical value (2.07x here)


def test_ring_curvature_fit_recovers_a_symmetric_curvature_and_conserves_the_centroid():
    """fit_ring_curvature (2026-11-17): a residual exactly linear in each atom's own bead offset, with a known symmetric
    J_i, is recovered with all its variation explained; the force J_i dq sums to zero over a ring's beads."""
    from frost.pimd import fit_ring_curvature

    g = torch.Generator().manual_seed(0)
    p, n = 16, 5
    m = torch.randn(n, 3, 3, generator=g, dtype=torch.float64)
    j_true = -(m @ m.transpose(1, 2)) - torch.eye(3, dtype=torch.float64)  # negative definite, like a bond's stiffness
    q = torch.randn(p, n, 3, generator=g, dtype=torch.float64) * 0.1
    dq = q - q.mean(0, keepdim=True)
    dr = torch.einsum("nij,bnj->bni", j_true, dq)
    j, explained = fit_ring_curvature(dr, dq)
    assert torch.allclose(j, j_true, atol=1e-4) and explained > 0.999999
    assert float(torch.einsum("nij,bnj->bni", j, dq).sum(0).abs().max()) < 1e-12
    noisy = dr + 0.05 * torch.randn(p, n, 3, generator=g, dtype=torch.float64)
    _, part = fit_ring_curvature(noisy - noisy.mean(0, keepdim=True), dq)
    assert 0.5 < part < 1.0

