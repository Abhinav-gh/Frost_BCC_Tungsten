"""A competitor's value projected onto our hardware, with its competitor-favourable bound (skill: fair-comparison).

Each parameter is given as [low, nominal, high]. The expression is evaluated at the nominal point and at every
corner of the parameter box (the extremes of a monotone model; for a non-monotone one, add interior points to the
lists). The bound favourable to the competitor is the box's maximum when higher is better, its minimum otherwise.
The verdict compares our measured interval with the competitor's range:
- win (robust): our pessimistic end beats their favourable bound;
- win (nominal): ours beats their nominal but not their favourable bound;
- tie: the intervals overlap otherwise;
- loss: their unfavourable end beats our optimistic end.

Usage
  python scripts/compare_projection.py spec.json [spec2.json ...]
"""

from __future__ import annotations

import ast
import itertools
import json
import math
import operator
import sys
from pathlib import Path

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos}
_FUNCS = {"min": min, "max": max, "sqrt": math.sqrt, "log": math.log, "exp": math.exp}


def evaluate(expression: str, env: dict) -> float:
    """Arithmetic only: numbers, the named parameters, + - * / **, and min / max / sqrt / log / exp."""
    def walk(node):
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name):
            return float(env[node.id])
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](walk(node.operand))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
            return _FUNCS[node.func.id](*(walk(a) for a in node.args))
        raise ValueError(f"unsupported expression element: {ast.dump(node)}")

    return walk(ast.parse(expression, mode="eval"))


def project(spec: dict) -> dict:
    params = spec["params"]
    names = list(params)
    nominal = evaluate(spec["expression"], {k: v[1] for k, v in params.items()})
    corners = [evaluate(spec["expression"], dict(zip(names, combo)))
               for combo in itertools.product(*([v[0], v[2]] for v in params.values()))]
    lo, hi = min(corners + [nominal]), max(corners + [nominal])
    higher = spec.get("higher_is_better", True)
    favourable, unfavourable = (hi, lo) if higher else (lo, hi)
    ours = spec["ours"]
    o_val, o_lo, o_hi = ours["value"], ours.get("low", ours["value"]), ours.get("high", ours["value"])
    o_bad, o_good = (o_lo, o_hi) if higher else (o_hi, o_lo)
    better = (lambda a, b: a > b) if higher else (lambda a, b: a < b)
    if better(o_bad, favourable):
        verdict = "win (robust)"
    elif better(unfavourable, o_good):
        verdict = "loss"
    elif better(o_val, nominal):
        verdict = "win (nominal)"
    else:
        verdict = "tie"
    sensitivity = {}
    for name in names:
        vals = [evaluate(spec["expression"], {**{k: v[1] for k, v in params.items()}, name: x})
                for x in (params[name][0], params[name][2])]
        sensitivity[name] = [min(vals), max(vals)]
    return {"competitor": spec["competitor"], "tier": spec.get("tier", "T3"), "nominal": nominal, "range": [lo, hi],
            "competitor_favourable": favourable, "ours": [o_lo, o_val, o_hi], "verdict": verdict,
            "one_at_a_time": sensitivity, "notes": spec.get("notes", "")}


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    for path in argv:
        spec = json.loads(Path(path).read_text())
        r = project(spec)
        print(f"{r['competitor']} [{r['tier']}]: nominal {r['nominal']:.3g}, range {r['range'][0]:.3g}-{r['range'][1]:.3g}, "
              f"favourable to them {r['competitor_favourable']:.3g} | ours {r['ours'][1]:.3g} "
              f"({r['ours'][0]:.3g}-{r['ours'][2]:.3g}) -> {r['verdict']}")
        for name, (a, b) in r["one_at_a_time"].items():
            print(f"    {name}: {a:.3g}-{b:.3g}")
        out = Path(path).with_suffix(".result.json")
        out.write_text(json.dumps(r, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
