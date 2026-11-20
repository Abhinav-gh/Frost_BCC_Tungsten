"""Spare slots of a fixed-capacity edge buffer (CUDA graphs need fixed shapes), spread over the atoms.

A captured force call reads an edge list of fixed capacity, so the slots past the real edges must hold harmless
edges: a self-edge pushed beyond the cutoff (the teacher) or masked out (the student). Putting every spare slot on
atom 0 is harmless for the values but not for the time. The force's backward through the edge gather
``x[sender] - x[receiver]`` is PyTorch's sort-based ``index_put_(accumulate=True)``, which walks each run of equal
indices serially (``indexing_backward_kernel_small_stride``), and an edge-force sum (``index_add_``) serialises its
atomics on one address the same way. With a 25% margin at 4,000 Cu atoms that is a run of 133,000 slots on atom 0:
77 ms of a 193 ms MACE call, and 371 ms against 118 ms spread, run to run erratic (Li3PO4 63 against 24 ms; Cu 500,
16,000 slots, unaffected). Slot s joins atom s mod n to itself, so no index repeats more than ceil(spare / n) times.
"""

from __future__ import annotations

import torch


def spread_padding(n_edges: int, capacity: int, n_atoms: int, device) -> torch.Tensor:
    """Atom indices for slots n_edges .. capacity - 1: slot s gets atom (s - n_edges) mod n_atoms."""
    return torch.arange(max(capacity - n_edges, 0), device=device) % n_atoms
