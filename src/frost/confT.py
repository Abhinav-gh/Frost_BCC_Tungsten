"""Configurational temperature: is a set of configurations canonical at the thermostat's temperature? (2026-11-14)

kT_conf = <|grad U|^2> / <laplacian U> for configurations drawn from exp(-U / kT) (Rugh, PRL 78, 772, 1997; Butler,
Ayton, Jepps and Evans, J. Chem. Phys. 109, 6519, 1998): an integration by parts, so it holds for any potential and
needs no velocities. A thermostat holds the kinetic temperature whatever the force; a held force that biases the
configurational distribution (an impulse integrator's shadow potential, O((k dt)^2) in the held force) moves T_conf
away from T. The melting point shifted 10-19 K at k >= 20 with every other clause passing (gaps.md, melting), which is
the case this is for.

The Laplacian is the trace of the Hessian, estimated by Hutchinson's method: tr(H) = E[v^T H v] over random sign
vectors v, each H v one double-backward pass (grad of (grad U . v)), so no Hessian is ever formed.
"""

from __future__ import annotations

import math

import torch

KB_EV = 8.617333262e-5


def grad_and_trace(energy_fn, x: torch.Tensor, probes: int = 2, generator: torch.Generator | None = None):
    """(|grad U|^2 summed over all components, the Hutchinson estimate of tr(H), its per-probe values) at positions x.

    ``energy_fn(x)`` returns the scalar energy and must be twice differentiable in x (no detach inside)."""
    x = x.detach().clone().requires_grad_(True)
    e = energy_fn(x)
    (g,) = torch.autograd.grad(e, x, create_graph=True)
    g2 = float((g * g).sum())
    values = []
    for p in range(probes):
        v = (torch.randint(0, 2, x.shape, generator=generator, device="cpu").to(x.device, x.dtype) * 2 - 1)
        (hv,) = torch.autograd.grad(g, x, grad_outputs=v, retain_graph=p < probes - 1)
        values.append(float((v * hv).sum()))
    return g2, sum(values) / len(values), values


def conf_temperature(g2: list[float], trace: list[float], blocks: int = 8) -> tuple[float, float]:
    """T_conf (K) as the ratio of means over frames, and its standard error from block means (frames in order)."""
    n = len(g2)
    t = sum(g2) / sum(trace) / KB_EV
    if n < 2 * blocks:
        return t, float("nan")
    size = n // blocks
    est = []
    for b in range(blocks):
        sg, st = sum(g2[b * size:(b + 1) * size]), sum(trace[b * size:(b + 1) * size])
        est.append(sg / st / KB_EV)
    mean = sum(est) / blocks
    se = math.sqrt(sum((e - mean) ** 2 for e in est) / (blocks - 1) / blocks)
    return t, se
