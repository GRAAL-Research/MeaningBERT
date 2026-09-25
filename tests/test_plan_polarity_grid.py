"""Tests for the experiment 3 grid planner (``src/training/plan_polarity_grid.py``).

The planner exists because of a specific, expensive mistake. There is no shared filesystem,
so each worker decides what to skip by looking at its OWN results tree; a slice naming a
cell finished on another machine makes that machine recompute it. It happened twenty times
during the v2 campaign. So the property that matters most here is not balance, it is that
every cell is placed exactly once.
"""

from __future__ import annotations

import collections

import pytest

from training.plan_polarity_grid import ARCHS, WORKERS, assign, cells, parse_seeds

PASCAL = ["renard-gpu0", "renard-gpu1", "souris-gpu0"]
BASE_ARCHS = [tag for tag, (cost, needed) in ARCHS.items() if cost == 1.0 and needed == 0]


def _grid(archs=None, seeds=(42, 43), conditions=("none", "full")):
    return cells(list(archs or BASE_ARCHS), list(seeds), list(conditions))


# --- seeds ----------------------------------------------------------------------------


def test_a_seed_range_expands_inclusively():
    assert parse_seeds("42-51") == list(range(42, 52))


def test_a_seed_list_is_read_as_written():
    assert parse_seeds("42,45,48") == [42, 45, 48]


# --- the grid -------------------------------------------------------------------------


def test_the_grid_is_the_full_product_of_arch_seed_and_condition():
    grid = _grid(archs=["bert", "deberta-v3-base"], seeds=(42, 43, 44))
    assert len(grid) == 2 * 3 * 2
    assert len({(a, s, c) for a, s, c, _ in grid}) == len(grid)


def test_the_heaviest_cells_come_first():
    # Longest-processing-time first: the expensive cells are placed while there is still
    # room to balance around them.
    grid = _grid(archs=["bert", "deberta-v3-large"])
    costs = [cost for *_, cost in grid]
    assert costs == sorted(costs, reverse=True)


def test_the_augmented_condition_costs_more_than_the_plain_one():
    grid = {(a, s, c): cost for a, s, c, cost in _grid(archs=["bert"], seeds=(42,))}
    assert grid[("bert", 42, "full")] > grid[("bert", 42, "none")]


# --- assignment -----------------------------------------------------------------------


def test_every_cell_is_placed_exactly_once():
    # THE property. A dropped cell is a hole nobody notices until the analysis; a duplicated
    # one is the twenty wasted runs of the v2 campaign.
    grid = _grid()
    placed = assign(grid, PASCAL)
    flat = [cell[:3] for slice_ in placed.values() for cell in slice_]
    assert collections.Counter(flat) == collections.Counter(cell[:3] for cell in grid)


def test_the_load_is_balanced_across_the_workers():
    placed = assign(_grid(seeds=range(42, 52)), PASCAL)
    loads = [sum(cost for *_, cost in slice_) for slice_ in placed.values()]
    assert max(loads) - min(loads) <= max(cost for *_, cost in _grid())


def test_a_large_cell_avoids_the_smallest_card():
    # deberta-v3-large at micro-batch 4 wants more than the 1080 Ti's 11 GB is comfortable
    # with. Steering, not a hard rule: the runner halves the batch on out-of-memory.
    placed = assign(_grid(archs=["deberta-v3-large"], seeds=range(42, 52)), PASCAL)
    assert placed["renard-gpu0"] == []
    assert WORKERS["renard-gpu0"][1] < 12


def test_a_base_cell_is_allowed_on_the_smallest_card():
    placed = assign(_grid(archs=["bert"], seeds=range(42, 52)), PASCAL)
    assert placed["renard-gpu0"]


def test_the_plan_is_deterministic():
    grid = _grid()
    assert assign(grid, PASCAL) == assign(grid, PASCAL)


# --- the hardware gate ----------------------------------------------------------------


def test_modernbert_is_refused_on_a_pascal_only_fleet():
    # It is built around FlashAttention, which needs compute 8.0. It would run on Pascal,
    # so degraded that comparing it to anything else would be meaningless. Refusing beats
    # producing a number nobody should trust.
    with pytest.raises(ValueError, match="needs compute"):
        assign(_grid(archs=["modernbert-base"], seeds=(42,)), PASCAL)


def test_modernbert_is_placed_once_a_capable_machine_joins():
    placed = assign(_grid(archs=["modernbert-base"], seeds=(42,)), PASCAL + ["caribou-gpu0"])
    assert len(placed["caribou-gpu0"]) == 2
    assert WORKERS["caribou-gpu0"][0] >= 80


def test_a_capable_machine_still_takes_its_share_of_ordinary_cells():
    # Adding caribou must not turn it into a ModernBERT-only box while the others queue.
    placed = assign(_grid(archs=["bert", "modernbert-base"], seeds=range(42, 52)), PASCAL + ["caribou-gpu0"])
    ordinary = [cell for cell in placed["caribou-gpu0"] if cell[0] == "bert"]
    assert ordinary


def test_every_registered_arch_declares_a_cost_and_a_gate():
    for tag, (cost, needed) in ARCHS.items():
        assert cost > 0, tag
        assert needed in (0, 80), tag
