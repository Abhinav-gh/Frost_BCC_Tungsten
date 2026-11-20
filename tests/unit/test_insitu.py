"""Label pairs: consecutive calls by default, any calls within the gap when spread."""

from __future__ import annotations

import torch

from frost.insitu import Frame, LabelBuffer


def _buffer(steps):
    buffer = LabelBuffer(capacity=len(steps))
    empty = torch.zeros(0, dtype=torch.long)
    for s in steps:
        buffer.add(Frame(s, torch.zeros(2, 3), torch.zeros(2, 3), 0.0, empty, empty,
                         torch.zeros(0, 3)))
    return buffer


def test_consecutive_pairs_respect_the_gap():
    buffer = _buffer([0, 1, 2, 10, 11])
    assert buffer.pairs(max_gap=4) == [(0, 1), (1, 2), (3, 4)]


def test_spread_pairs_cover_every_gap_within_range_and_nothing_beyond():
    steps = list(range(0, 40))
    buffer = _buffer(steps)
    gen = torch.Generator().manual_seed(0)
    pairs = buffer.sample_pairs(4000, max_gap=8, generator=gen)
    gaps = {steps[b] - steps[a] for a, b in pairs}
    assert gaps == set(range(1, 9))
    assert all(0 <= a < b < len(steps) for a, b in pairs)


def test_spread_pairs_are_empty_when_no_two_calls_are_in_range():
    buffer = _buffer([0, 20, 40])
    assert buffer.sample_pairs(8, max_gap=4, generator=torch.Generator().manual_seed(0)) == []


def test_ring_pairs_are_the_same_time_on_later_bead_blocks():
    """A bead dump stores each bead's path as a block of `stride` frames: ring partners are whole strides apart."""
    buffer = _buffer([0, 4, 8, 12, 3000, 3004, 3008, 6000, 6004])
    assert sorted(buffer.ring_pairs(3000)) == [(0, 4), (0, 7), (1, 5), (1, 8), (2, 6), (4, 7), (5, 8)]
    assert buffer.ring_pairs(5000) == []
