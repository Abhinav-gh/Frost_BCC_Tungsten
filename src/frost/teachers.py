"""The teacher models by name, and hydrogen mass repartitioning.

Every trajectory file records the teacher that produced it in its ``teacher`` attribute, and
every script that relabels, validates or replays a run builds that teacher from the file, so a
run cannot be scored against a model it was not made with. Files written before the attribute
existed were all made with MACE-MP-0 medium.

- ``mp-medium``: MACE-MP-0 medium (Batatia et al. 2023), the materials teacher.
- ``off-small``, ``off-medium``: MACE-OFF23 (Kovacs et al. 2023), the organic and water teacher
  that DMTS-NC distils (Gouraud et al. 2026, Table 6 uses the small model).

All are built through mace's own constructors (``mace_mp`` / ``mace_off``) with cuEquivariance
kernels, the strongest stack for the baseline and the multirate arm alike.
"""

from __future__ import annotations

import numpy as np

TEACHERS = ("mp-medium", "off-small", "off-medium")
DEFAULT = "mp-medium"


def load_calc(name: str = DEFAULT, device: str = "cuda", dtype: str = "float32", enable_cueq: bool = True,
              **kwargs):
    """The named teacher as a mace calculator."""
    if name == "mp-medium":
        from mace.calculators import mace_mp

        return mace_mp(model="medium", device=device, default_dtype=dtype, enable_cueq=enable_cueq, **kwargs)
    if name in ("off-small", "off-medium"):
        from mace.calculators import mace_off

        return mace_off(model=name.split("-")[1], device=device, default_dtype=dtype, enable_cueq=enable_cueq,
                        **kwargs)
    raise ValueError(f"unknown teacher {name!r}; known: {TEACHERS}")


def teacher_of(attrs) -> str:
    """The teacher a trajectory file was made with (its ``teacher`` attribute)."""
    value = attrs.get("teacher", DEFAULT)
    return value.decode() if isinstance(value, bytes) else str(value)


def repartition_hydrogen(numbers, masses, added_da: float) -> np.ndarray:
    """Hydrogen mass repartitioning, as FeNNol's ``hmr`` keyword and DMTS define it.

    ``added_da`` is added to every hydrogen, and the same total is taken from the other atoms in
    proportion to their masses, so the total mass (and so the density) is unchanged. In water,
    3 Da gives H 4.008 and O 9.999.
    """
    m = np.asarray(masses, dtype=np.float64).copy()
    if added_da <= 0:
        return m
    h = np.asarray(numbers) == 1
    total = m.sum()
    m[h] += added_da
    m[~h] -= added_da * h.sum() * m[~h] / m[~h].sum()
    assert np.isclose(m.sum(), total) and (m > 0).all()
    return m
