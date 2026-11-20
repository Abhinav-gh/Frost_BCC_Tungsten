"""What reusing an intermediate activation does to a gradient force (research_direction.md, Theorem A).

The energy is E(R) = G(R, h(R)): a head G that sees positions directly and through
an intermediate activation h. A reuse scheme substitutes some h~ for h, and autograd
then returns F~ = -dG/dR - J~^T lambda~, where J~ is whatever Jacobian of h~ the
graph still carries. With the value error e = h - h~ and the carried-Jacobian error
D = J - J~, the exact decomposition is

    F~ - F = D^T lambda + O(|e|),        lambda = dG/dh,

so the force error is set by the derivative error projected on the adjoint, and the
value error enters only at first order. These tests check each consequence on a
tiny message-passing model, in float64 on CPU (six atoms: the unit-test exception to
the no-CPU-models rule):

- detaching h (value exact, derivative deleted) gives exactly the deleted path;
- freezing h at a reference leaves an O(1) force error however close the
  configuration is, so refreshing, even every step, cannot remove it;
- the energy error vanishes linearly while the force error does not;
- a first-order (affine) surrogate is exact at the reference and O(delta) away;
- a surrogate whose values converge but whose Jacobian does not keeps an O(1)
  force error: value accuracy does not control force accuracy;
- freezing an invariant (l = 0) activation keeps the surrogate rotation-invariant,
  so the net torque stays zero; freezing an equivariant (l = 1) one produces a net
  torque equal to sum_j v0_j x lambda_j, exactly.
"""

from __future__ import annotations

import pytest
import torch

RC = 3.0
CENTRES = torch.tensor([1.0, 1.6, 2.2], dtype=torch.float64)


@pytest.fixture(autouse=True)
def float64_one_thread():
    """float64 for these tests only: a module-level default would leak into every later test."""
    threads, dtype = torch.get_num_threads(), torch.get_default_dtype()
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(dtype)
    torch.set_num_threads(threads)


def geometry(x: torch.Tensor):
    """Pair vectors r_ij = x_j - x_i, distances, a smooth cutoff and an off-diagonal mask."""
    n = x.shape[0]
    rij = x[None, :, :] - x[:, None, :]
    eye = torch.eye(n, dtype=x.dtype)
    dist = (rij.pow(2).sum(-1) + eye).sqrt()  # the +eye keeps the diagonal away from 0
    mask = 1.0 - eye
    fc = torch.clamp(1.0 - (dist / RC) ** 2, min=0.0) ** 2 * mask
    return rij, dist, fc


def features(x: torch.Tensor):
    """Layer 1: an invariant s (l = 0) and an equivariant v (l = 1), both sums over neighbours."""
    rij, dist, fc = geometry(x)
    phi = torch.exp(-((dist[..., None] - CENTRES) ** 2)) * fc[..., None]  # (N, N, C)
    s = phi.sum(1)  # (N, C)
    unit = rij / dist[..., None]
    v = (phi[..., None] * unit[:, :, None, :]).sum(1)  # (N, C, 3)
    return s, v


class Head:
    """Layer 2 and readout, G(R, s, v): sees positions directly and through s and v."""

    def __init__(self, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        c = len(CENTRES)
        self.w1 = torch.randn(c, c, generator=g) * 0.5
        self.w2 = torch.randn(c, generator=g) * 0.3
        self.w3 = torch.randn(c, generator=g) * 0.3

    def __call__(self, x, s, v):
        rij, dist, fc = geometry(x)
        psi = torch.exp(-dist) * fc
        a = torch.tanh(s @ self.w1)  # (N, C)
        e_pair = (psi[..., None] * a[None, :, :]).sum()  # neighbours' features, as in MACE layer 2
        unit = rij / dist[..., None]
        # v_j . r_ij: invariant only while v rotates with the positions
        proj = torch.einsum("jcd,ijd->ijc", v, unit)
        e_dir = (psi[..., None] * proj * self.w2).sum()
        e_self = ((s**2) @ self.w3).sum()
        return e_pair + e_dir + e_self


def force(energy_fn, x):
    x = x.detach().requires_grad_(True)
    e = energy_fn(x)
    (g,) = torch.autograd.grad(e, x)
    return e.detach(), -g


@pytest.fixture
def system():
    g = torch.Generator().manual_seed(1)
    x0 = torch.randn(6, 3, generator=g) * 1.1
    u = torch.randn(6, 3, generator=g)
    return x0, u / u.norm(), Head()


def exact(head):
    return lambda x: head(x, *features(x))


def deleted_path(head, x):
    """J^T lambda through s alone: backpropagate dG/ds (held) into s(R)."""
    x = x.detach().requires_grad_(True)
    s, v = features(x)
    s_leaf = s.detach().requires_grad_(True)
    (lam,) = torch.autograd.grad(head(x, s_leaf, v.detach()), s_leaf)
    (jt_lam,) = torch.autograd.grad((s * lam).sum(), x)
    return jt_lam


def test_detaching_deletes_exactly_the_adjoint_path(system):
    x0, _, head = system

    def detached(x):
        s, v = features(x)
        return head(x, s.detach(), v)

    e_ex, f_ex = force(exact(head), x0)
    e_det, f_det = force(detached, x0)
    assert torch.allclose(e_det, e_ex, atol=0, rtol=0)  # the value is exact...
    jt_lam = deleted_path(head, x0)
    assert torch.allclose(f_det - f_ex, jt_lam, atol=1e-12)  # ...the error is J^T lambda
    assert jt_lam.norm() > 0.05 * f_ex.norm()  # and it is not small


def test_freezing_error_does_not_vanish_as_the_reference_approaches(system):
    x0, u, head = system
    s0 = features(x0)[0].detach()
    jt_lam0 = deleted_path(head, x0)

    def frozen(x):
        return head(x, s0, features(x)[1])

    energy_errors, force_gaps = [], []
    for delta in (1e-2, 1e-3, 1e-4):
        x = x0 + delta * u
        e_ex, f_ex = force(exact(head), x)
        e_fr, f_fr = force(frozen, x)
        energy_errors.append(abs(float(e_fr - e_ex)) / delta)
        force_gaps.append(float((f_fr - f_ex - jt_lam0).norm()) / delta)
        # the force error tends to the deleted path at the reference, not to zero
        assert (f_fr - f_ex).norm() > 0.9 * jt_lam0.norm()
    # energy error and the approach to the limit are both linear in delta
    assert max(energy_errors) < 2.0 * min(energy_errors)
    assert max(force_gaps) < 2.0 * min(force_gaps)


def test_first_order_surrogate_restores_the_force(system):
    x0, u, head = system
    s0 = features(x0)[0].detach()
    jac = torch.autograd.functional.jacobian(lambda x: features(x)[0], x0)  # (N, C, N, 3)

    def affine(x):
        s_lin = s0 + torch.einsum("icjd,jd->ic", jac, x - x0)
        return head(x, s_lin, features(x)[1])

    _, f0 = force(exact(head), x0)
    _, f0_aff = force(affine, x0)
    assert (f0_aff - f0).norm() < 1e-12  # exact at the reference
    slopes = []
    for delta in (1e-2, 1e-3, 1e-4):
        x = x0 + delta * u
        _, f_ex = force(exact(head), x)
        _, f_aff = force(affine, x)
        slopes.append(float((f_aff - f_ex).norm()) / delta)
    assert max(slopes) < 1.5 * min(slopes)  # O(delta), v1's measured exponent 1.00


def test_value_accuracy_does_not_control_force_accuracy(system):
    """A surrogate within eps of h in value, but with an O(1) Jacobian error."""
    x0, _, head = system
    g = torch.Generator().manual_seed(7)
    k = torch.randn(6, 3, len(CENTRES), generator=g)
    _, f_ex = force(exact(head), x0)
    errors = []
    for eps in (1e-2, 1e-4, 1e-6):

        def wiggly(x, eps=eps):
            s, v = features(x)
            wiggle = eps * torch.sin(torch.einsum("jdc,jd->c", k, x) / eps)
            return head(x, s + wiggle, v)

        value_error = float((eps * torch.sin(torch.einsum("jdc,jd->c", k, x0) / eps)).abs().max())
        assert value_error <= eps
        _, f_w = force(wiggly, x0)
        errors.append(float((f_w - f_ex).norm()))
    # the value error fell 10^4-fold; the force error did not fall at all
    assert min(errors) > 0.1 * max(errors)
    assert min(errors) > 1e-2


@pytest.mark.parametrize("which", ["s", "v"])
def test_freezing_an_equivariant_activation_breaks_rotation_invariance(system, which):
    """Net torque of the surrogate is exactly sum_j v0_j x lambda_j for a frozen l = 1 v0.

    G(QR, s, Qv) = G(R, s, v) for every rotation Q. Differentiating at the identity
    along omega gives sum_i (omega x x_i) . dG/dx_i + sum_j (omega x v_j) . lambda_j = 0,
    and the net torque of -dG/dR is therefore sum_j v_j x lambda_j when v is held
    fixed. A frozen invariant (l = 0) contributes no such term.
    """
    x0, u, head = system
    s0, v0 = (t.detach() for t in features(x0))

    def frozen(x):
        s, v = features(x)
        return head(x, s0, v) if which == "s" else head(x, s, v0)

    x = x0 + 0.05 * u
    _, f = force(frozen, x)
    torque = torch.cross(x - x.mean(0), f, dim=1).sum(0)
    assert f.sum(0).norm() < 1e-12  # translation invariance survives either freeze
    if which == "s":
        assert torque.norm() < 1e-12
        return
    v_leaf = v0.clone().requires_grad_(True)
    (lam,) = torch.autograd.grad(head(x, features(x)[0], v_leaf), v_leaf)
    predicted = torch.cross(v0.reshape(-1, 3), lam.reshape(-1, 3), dim=1).sum(0)
    assert predicted.norm() > 1e-5
    assert torch.allclose(torque, predicted, atol=1e-14, rtol=1e-10)
