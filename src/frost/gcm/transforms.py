"""Spherical-harmonic transforms for NeuralGCM's dynamical core on GPUs.

dinosaur's ``RealSphericalHarmonics`` (the implementation NeuralGCM's checkpoints are
configured with) computes the longitude Fourier transform as a dense DFT matmul, noting that
on the TPUs it was tuned for this beat an explicit FFT. On an RTX A6000 the same matmuls,
resolved to 6-pass bfloat16 with their operand splits and layout transposes, are most of
the dynamical core's time (``experiments/e11_gcm/kernel_profile.py``).

``FFTSphericalHarmonics`` keeps ``RealSphericalHarmonics`` unchanged in every respect (the
modal layout [0, +1, -1, ..., +M, -M] x l, the basis, the quadrature weights, the Legendre
einsum and its precision) except the Fourier step, which it computes with ``jnp.fft``:

    inverse:  f(x_i) = a_0 / sqrt(2 pi) + sum_j (a_j cos j x_i + b_j sin j x_i) / sqrt(pi)
              = irfft(C)_i,  C_0 = N a_0 / sqrt(2 pi),  C_j = N (a_j - i b_j) / (2 sqrt(pi))
    forward:  G_0 = Re R_0 / sqrt(2 pi),  G_{2j-1} = Re R_j / sqrt(pi),  G_{2j} = -Im R_j / sqrt(pi),
              R = rfft(w x)

(``dinosaur.fourier.real_basis`` columns: 1/sqrt(2 pi), then cos and sin of j x / sqrt(pi)).
It is a subclass passed through dinosaur's documented ``spherical_harmonics_impl`` option,
registered with gin under its own name; nothing in dinosaur is modified. Equivalence to
``RealSphericalHarmonics`` is checked in ``tests/jax/check_gcm.py``.
"""

from __future__ import annotations

import gin
import jax
import jax.numpy as jnp
import numpy as np
from dinosaur import jax_numpy_utils, spherical_harmonic


class FFTSphericalHarmonics(spherical_harmonic.RealSphericalHarmonics):
    """``RealSphericalHarmonics`` with the longitude transform done by FFT (axis -2: longitude)."""

    def _from_real_basis(self, a):
        """Real-basis coefficients [..., 2M-1, lat] -> rfft coefficients [..., N//2+1, lat]."""
        n, m = self.longitude_nodes, self.longitude_wavenumbers
        c0 = a[..., :1, :] * (n / np.sqrt(2.0 * np.pi))
        cj = (a[..., 1::2, :] - 1j * a[..., 2::2, :]) * (n / (2.0 * np.sqrt(np.pi)))
        pad = n // 2 + 1 - m
        zeros = jnp.zeros((*a.shape[:-2], pad, a.shape[-1]), cj.dtype)
        return jnp.concatenate([c0.astype(cj.dtype), cj, zeros], axis=-2)

    def _to_real_basis(self, r):
        """rfft coefficients [..., N//2+1, lat] -> real-basis coefficients [..., 2M-1, lat]."""
        m = self.longitude_wavenumbers
        g0 = jnp.real(r[..., :1, :]) / np.sqrt(2.0 * np.pi)
        rj = r[..., 1:m, :]
        pairs = jnp.stack([jnp.real(rj), -jnp.imag(rj)], axis=-2) / np.sqrt(np.pi)  # [..., M-1, 2, lat]
        return jnp.concatenate([g0, pairs.reshape(*rj.shape[:-2], 2 * (m - 1), rj.shape[-1])], axis=-2)

    def inverse_transform(self, x):
        px = jax.named_call(jax_numpy_utils.precise_einsum, name="inv_legendre")("mjl,...ml->...mj", self.basis.p, x)
        c = self._from_real_basis(px)
        return jax.named_call(jnp.fft.irfft, name="inv_fourier")(c, n=self.longitude_nodes, axis=-2).astype(x.dtype)

    def transform(self, x):
        wx = self.basis.w * x
        r = jax.named_call(jnp.fft.rfft, name="fwd_fourier")(wx, axis=-2)
        fwx = self._to_real_basis(r).astype(x.dtype)
        return jax.named_call(jax_numpy_utils.precise_einsum, name="fwd_legendre")("mjl,...mj->...ml", self.basis.p,
                                                                                    fwx)


class F32RealSphericalHarmonics(spherical_harmonic.RealSphericalHarmonics):
    """``RealSphericalHarmonics`` with its four transform einsums in native float32.

    dinosaur resolves float32 einsums to 6-pass bfloat16 emulation on Ampere and newer
    (``jax_numpy_utils.FLOAT32_DOT_ALGORITHM``); XLA:GPU lowers that by splitting each
    operand into three bfloat16 parts, and the splits are written out as concatenate
    fusions before the GEMM. At these shapes the GEMMs run at 5-17 TFLOP/s in any precision
    (``experiments/e11_gcm/ampere_probe.py``), so true float32 on CUDA cores removes the
    splits without slowing the products. Same modal layout, basis and contraction order.
    """

    algorithm = jax.lax.DotAlgorithmPreset.F32_F32_F32

    def inverse_transform(self, x):
        px = jax.named_call(jnp.einsum, name="inv_legendre")("mjl,...ml->...mj", self.basis.p, x,
                                                             precision=self.algorithm)
        return jax.named_call(jnp.einsum, name="inv_fourier")("im,...mj->...ij", self.basis.f, px,
                                                              precision=self.algorithm)

    def transform(self, x):
        wx = self.basis.w * x
        fwx = jax.named_call(jnp.einsum, name="fwd_fourier")("im,...ij->...mj", self.basis.f, wx,
                                                             precision=self.algorithm)
        return jax.named_call(jnp.einsum, name="fwd_legendre")("mjl,...mj->...ml", self.basis.p, fwx,
                                                               precision=self.algorithm)


gin.external_configurable(FFTSphericalHarmonics, "FFTSphericalHarmonics")
gin.external_configurable(F32RealSphericalHarmonics, "F32RealSphericalHarmonics")
