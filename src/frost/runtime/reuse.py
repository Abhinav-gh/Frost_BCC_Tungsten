"""Reuse typing: which held forms a runtime may use, from how the consumer uses the tensor.

The rule (new_directions.md S1, Theorem A): an intermediate may be held if and only if no
consumer differentiates through it.

| class | consumer | held forms allowed | error of a hold |
|---|---|---|---|
| structural | nothing state-dependent | any | none |
| solver | an iterative solve to tolerance | value | none beyond the tolerance |
| operator | an operator applied to the fresh state | operator, value | (A(u) - A(u0)) u |
| value | added to a tendency or force | value (held, ramped, transported) | C(u) - C(u0) |
| gradient | differentiated (forces from an energy) | output only: the model's output or a residual of it; never an intermediate | deletes the adjoint term, 15-68% of the force |

``classify_torch`` derives the class of every submodule output of a PyTorch model whose forces
come from autograd, with forward hooks only (public API, nothing patched): an output that
does not depend on the positions is structural, one that does is gradient-consumed.
"""

from __future__ import annotations

from enum import Enum


class ReuseClass(str, Enum):
    STRUCTURAL = "structural"
    SOLVER = "solver"
    OPERATOR = "operator"
    VALUE = "value"
    GRADIENT = "gradient"


# what may be held, per class; "output" means the operator's final output (or a residual of
# it), "intermediate" any tensor inside it
_LEGAL = {
    ReuseClass.STRUCTURAL: {"output", "intermediate"},
    ReuseClass.SOLVER: {"output", "intermediate"},
    ReuseClass.OPERATOR: {"output", "intermediate"},
    ReuseClass.VALUE: {"output"},
    ReuseClass.GRADIENT: {"output"},
}


def check_hold(cls: ReuseClass, where: str) -> None:
    """Raise if holding at ``where`` ("output" or "intermediate") is illegal for ``cls``."""
    if where not in _LEGAL[ReuseClass(cls)]:
        raise ValueError(f"a {cls.value}-consumed tensor may not be held as an {where}: its consumer "
                         "differentiates through it, so the hold deletes the adjoint term (Theorem A)")


def classify_torch(model, inputs: dict, positions_key: str = "x", modules=None) -> dict:
    """Class of every submodule output of ``model(**inputs)``, structural or gradient-consumed.

    ``inputs[positions_key]`` must require grad and the parameters must not, so an output requires
    grad exactly when it depends on the positions. Forward hooks are registered and removed; the
    model is not modified.
    """
    import torch

    found, handles = {}, []
    named = dict(model.named_modules())
    for name, module in named.items():
        if modules is not None and name not in modules:
            continue

        def hook(_m, _a, out, name=name):
            t = out[0] if isinstance(out, (tuple, list)) else out
            if isinstance(t, torch.Tensor):
                found[name] = ReuseClass.GRADIENT if t.requires_grad else ReuseClass.STRUCTURAL

        handles.append(module.register_forward_hook(hook))
    try:
        model(**inputs)
    finally:
        for h in handles:
            h.remove()
    return found
