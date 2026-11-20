"""NeuralGCM split into its two rates (workload G1).

NeuralGCM (Kochkov et al., Nature 2024) advances the atmosphere in outer steps of 30-60
minutes. Each outer step calls the learned physics once (a per-column network) and then
runs several dynamical-core substeps (implicit-explicit SIL3 on spherical harmonics) with
those physics tendencies held constant. Its public API, ``PressureLevelModel.advance``,
fuses the two, and the docs state there is no public API that separates them.

This module composes the same public pieces into two functions, so a runtime can call the
physics on its own schedule:
- ``physics(x, forcing) -> tendency``: the step module's ``physics_parameterization_fn``
  and ``perturbation_fn``;
- ``finish(x, tendency, forcing) -> x_next``: its ``corrector_fn`` (the dycore substeps
  with the tendency held) and the randomness, memory and diagnostics updates.

Nothing in NeuralGCM or dinosaur is modified or reassigned. The model is rebuilt from the
checkpoint exactly as ``PressureLevelModel.from_checkpoint`` builds it (``gin_utils``,
``physics_specifications``, ``model_builder``), and our two functions are ``hk.transform``
of calls to the attributes of the step module that the model constructs, with the
checkpoint's parameters. ``finish(x, physics(x, f), f)`` reproduces ``advance`` (checked in
``experiments/e11_gcm/profile_split.py``).

``overrides`` appends gin bindings to the checkpoint's configuration, the same mechanism
``from_checkpoint`` uses for its radius fix. ``FAST_GPU`` selects dinosaur's
``FastSphericalHarmonics`` (registered as ``RealSphericalHarmonicsWithZeroImag``) for every
grid instead of the released ``RealSphericalHarmonics``, which dinosaur documents as its
pedagogical implementation; ``transform_precision`` then picks the matmul algorithm.
"""

from __future__ import annotations

import haiku as hk
import jax
import jax.numpy as jnp
import xarray
from dinosaur import typing as dtyping
from dinosaur import xarray_utils
from neuralgcm.legacy import api, gin_utils, model_builder, physics_specifications

from frost.gcm import transforms  # noqa: F401  (registers our transforms with gin)


# the model and dynamical-core grids; the data grid (``coordinate_system_from_dataset``, used by
# the encoder and decoder) keeps its released implementation
GRIDS = ("GridTL63", "GridTL127", "GridTL255", "smaller/GridWithWavenumbers", "GridWithWavenumbers")


def fast_transforms(precision: str | None = None) -> str:
    """Gin bindings: dinosaur's fast spherical harmonics on every grid, at ``precision``
    (None: dinosaur's platform default; 'highest', 'tensorfloat32' or 'bfloat16')."""
    lines = [f"{g}.spherical_harmonics_impl = @RealSphericalHarmonicsWithZeroImag" for g in GRIDS]
    if precision is not None:
        lines.append(f"RealSphericalHarmonicsWithZeroImag.transform_precision = '{precision}'")
    return "\n".join(lines)


def fft_transforms() -> str:
    """Gin bindings: our ``FFTSphericalHarmonics`` (``frost.gcm.transforms``) on the model and
    dynamical-core grids. Only for checkpoints configured with ``RealSphericalHarmonics`` (the
    released v1 models), whose modal layout it keeps; ``SplitModel`` refuses the others."""
    return "\n".join(f"{g}.spherical_harmonics_impl = @FFTSphericalHarmonics" for g in GRIDS)


def f32_transforms() -> str:
    """Gin bindings: ``F32RealSphericalHarmonics`` (``frost.gcm.transforms``), the released
    transform in native float32, on the model and dynamical-core grids (released v1 models)."""
    return "\n".join(f"{g}.spherical_harmonics_impl = @F32RealSphericalHarmonics" for g in GRIDS)


def _config(checkpoint: dict) -> str:
    """The checkpoint's gin config, with the radius fix ``PressureLevelModel.from_checkpoint`` applies."""
    return (checkpoint["model_config_str"].replace("GridWithWavenumbers.radius = None",
                                                   "GridWithWavenumbers.radius = 1.0")
            + "\n\n" + "\n".join(["GridTL63.radius = 1.0", "GridTL127.radius = 1.0", "GridTL255.radius = 1.0"]))


class SplitModel:
    """A NeuralGCM checkpoint with its physics and dynamics callable separately.

    ``api`` is a ``PressureLevelModel`` on the same structure (encode, decode, xarray
    conversion and the fused ``advance`` used as the reference). ``physics``, ``finish``, ``forcing`` and
    ``advance`` are jit-able functions of (params, ...).
    """

    def __init__(self, checkpoint: dict, overrides: str = ""):
        """``overrides``: gin bindings appended to the checkpoint's config (e.g. the dynamical
        core's spherical-harmonic implementation and its transform precision, both public
        options of dinosaur). Empty: the model exactly as released."""
        ours = [c for c in ("FFTSphericalHarmonics", "F32RealSphericalHarmonics") if f"@{c}" in overrides]
        if ours and "RealSphericalHarmonicsWithZeroImag" in checkpoint["model_config_str"]:
            raise ValueError(f"{ours[0]} keeps RealSphericalHarmonics' modal layout; this checkpoint "
                             "is configured with FastSphericalHarmonics (RealSphericalHarmonicsWithZeroImag)")
        self.params = checkpoint["params"]
        self.config = _config(checkpoint) + ("\n" + overrides if overrides else "")
        with gin_utils.specific_config(self.config):
            physics_specs = physics_specifications.get_physics_specs()
            aux_ds = xarray.Dataset.from_dict(checkpoint["aux_ds_dict"])
            data_coords = model_builder.coordinate_system_from_dataset(aux_ds)
            specs = model_builder.get_model_specs(data_coords, physics_specs, {xarray_utils.XARRAY_DS_KEY: aux_ds})
            self.whirl = model_builder.WhirlModel(coords=specs.coords, dt=specs.dt, physics_specs=specs.physics_specs,
                                                  aux_features=specs.aux_features, input_coords=data_coords,
                                                  output_coords=data_coords)
        # the public constructor, with our (possibly overridden) config: encode, decode and the
        # fused advance then share the dycore's transform implementation and modal layout
        self.api = api.PressureLevelModel(self.whirl, self.params, self.config)
        model_cls = self.whirl.model_cls

        def physics_fwd(x: dtyping.ModelState, forcing):
            step = model_cls().advance_fn
            tendency = step.physics_parameterization_fn(x.state, x.memory, x.diagnostics, x.randomness.nodal_value,
                                                        forcing)
            return step.perturbation_fn(tendency, state=x.state, randomness=x.randomness.nodal_value)

        def finish_fwd(x: dtyping.ModelState, tendency, forcing):
            step = model_cls().advance_fn
            next_state = step.corrector_fn(x.state, tendency, forcing)
            return dtyping.ModelState(state=next_state, memory=x.state if x.memory is not None else None,
                                      diagnostics=step.diagnostics_fn(x, tendency, forcing),
                                      randomness=step.randomness_fn.advance(x.randomness))

        def forcing_fwd(forcing_data, sim_time):
            return model_cls().forcing_fn(forcing_data, sim_time)

        def advance_fwd(x, forcing):
            return model_cls().advance(x, forcing)

        self._physics = hk.transform(physics_fwd).apply
        self._finish = hk.transform(finish_fwd).apply
        self._forcing = hk.transform(forcing_fwd).apply
        self._advance = hk.transform(advance_fwd).apply
        self.dt = specs.dt
        self.num_substeps = self._num_substeps(checkpoint)

    @staticmethod
    def _num_substeps(checkpoint: dict) -> int:
        """Physics calls per model step (``StochasticPhysicsParameterizationStep.num_substeps``, often a macro)."""
        lines = checkpoint["model_config_str"].splitlines()
        value = next((line.split("=", 1)[1].strip() for line in lines
                      if line.startswith("StochasticPhysicsParameterizationStep.num_substeps")), "1")
        if value.startswith("%"):
            value = next(line.split("=", 1)[1].strip() for line in lines if line.startswith(value[1:] + " "))
        return int(value)

    def forcing_at(self, params, forcings: dict, x: dtyping.ModelState):
        """Model forcing at ``x``'s time from pressure-level forcing data, as ``advance`` prepares it."""
        forcings = {k: (v.squeeze(axis=-3) if k in self.api.forcing_variables else v) for k, v in forcings.items()}
        forcings = jax.tree_util.tree_map(lambda a: jnp.expand_dims(a, axis=0), forcings)
        return self.forcing(params, forcings, x.state.sim_time)

    # gin bindings are read when the Haiku modules are built, i.e. while tracing
    def physics(self, params, x, forcing):
        with gin_utils.specific_config(self.config):
            return self._physics(params, None, x, forcing)

    def finish(self, params, x, tendency, forcing):
        with gin_utils.specific_config(self.config):
            return self._finish(params, None, x, tendency, forcing)

    def forcing(self, params, forcing_data, sim_time):
        with gin_utils.specific_config(self.config):
            return self._forcing(params, None, forcing_data, sim_time)

    def advance(self, params, x, forcing):
        with gin_utils.specific_config(self.config):
            return self._advance(params, None, x, forcing)
