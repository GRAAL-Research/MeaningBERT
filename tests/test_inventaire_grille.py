"""Tests for the grid recovery inventory (``src/training/inventaire_grille.py``).

It exists because a slice is a single pass: a cell killed by the watchdog or skipped on a
stale claim leaves a hole the driver never revisits. Both happened in the first two nights.
So the property that matters is that the inventory finds every hole and sends none of them
back to a card that wedges on them.
"""

from __future__ import annotations

import pytest

from training.inventaire_grille import done_cells, missing, reassign
from training.plan_polarity_grid import FORBIDDEN

PASCAL = ["renard-gpu0", "renard-gpu1", "souris-gpu0"]


def _write(root, tag, seed, with_metrics=True):
    cell = root / tag / f"seed{seed}"
    cell.mkdir(parents=True, exist_ok=True)
    if with_metrics:
        (cell / "metrics.json").write_text("{}", encoding="utf-8")
    return cell


# --- reading a results tree -----------------------------------------------------------


def test_a_cell_with_metrics_is_read_as_done(tmp_path):
    _write(tmp_path, "deberta-v3-base-none", 42)
    assert done_cells(tmp_path) == {("deberta-v3-base", 42, "none")}


def test_a_cell_without_metrics_is_not_read_as_done(tmp_path):
    _write(tmp_path, "deberta-v3-base-none", 42, with_metrics=False)
    assert done_cells(tmp_path) == set()


def test_an_architecture_tag_containing_dashes_is_split_from_the_right(tmp_path):
    # Splitting from the left reads nli-deberta-v3-base-none as architecture "nli" and
    # quietly reports every cell of that model as missing, which would re-run forty hours
    # of finished work.
    _write(tmp_path, "nli-deberta-v3-base-none", 42)
    _write(tmp_path, "nli-deberta-v3-large-full", 43)
    assert done_cells(tmp_path) == {
        ("nli-deberta-v3-base", 42, "none"),
        ("nli-deberta-v3-large", 43, "full"),
    }


def test_a_directory_that_is_not_a_known_architecture_is_ignored(tmp_path):
    _write(tmp_path, "experiences-perso-none", 42)
    assert done_cells(tmp_path) == set()


# --- finding the holes ----------------------------------------------------------------


def test_a_cell_done_on_any_machine_counts_as_done():
    # There is no shared filesystem, so the inventory is the union of the trees: a cell
    # finished on souris must not be re-run on renard.
    done = {"renard": {("bert", 42, "none")}, "souris": {("bert", 43, "none")}}
    assert missing(done, ["bert"], [42, 43], ["none"]) == []


def test_every_absent_cell_is_reported():
    done = {"renard": {("bert", 42, "none")}}
    holes = missing(done, ["bert"], [42, 43, 44], ["none"])
    assert holes == [("bert", 43, "none"), ("bert", 44, "none")]


def test_the_heaviest_holes_come_first():
    holes = missing({}, ["bert", "deberta-v3-large"], [42], ["none", "full"])
    assert holes[0][0] == "deberta-v3-large"
    assert holes[0][2] == "full"


def test_an_empty_inventory_reports_everything_missing():
    assert len(missing({}, ["bert"], [42, 43], ["none", "full"])) == 4


# --- sending the holes somewhere they will not wedge ----------------------------------


def test_a_hole_is_never_sent_back_to_the_card_that_wedges_it():
    # The whole reason ten stsb cells were lost: re-running them where they died would
    # lose them again.
    assert ("stsb-roberta-base", "renard-gpu0") in FORBIDDEN
    placed = reassign(missing({}, ["stsb-roberta-base"], list(range(42, 52)), ["none"]), PASCAL)
    assert placed["renard-gpu0"] == []
    assert sum(len(v) for v in placed.values()) == 10


def test_every_hole_is_placed_exactly_once():
    holes = missing({}, ["bert", "deberta-v3-base"], [42, 43], ["none", "full"])
    placed = reassign(holes, PASCAL)
    flat = [cell[:3] for slice_ in placed.values() for cell in slice_]
    assert sorted(flat) == sorted(holes)


def test_a_hole_no_worker_can_take_is_refused_rather_than_dropped():
    holes = missing({}, ["bert"], [42], ["none"])
    with pytest.raises(ValueError, match="aucun worker"):
        reassign(holes, ["renard-gpu0"])


def test_a_large_hole_avoids_the_smallest_card():
    placed = reassign(missing({}, ["deberta-v3-large"], list(range(42, 52)), ["none"]), PASCAL)
    assert placed["renard-gpu0"] == []
