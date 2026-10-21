"""Tests for the replay corpus (``build_replay_corpus.py``).

The replay must stay balanced, keep MultiNLI's labels on their pair, and grow the train
split only: a replay row in dev would change model selection for every regime it is
compared with.
"""

from __future__ import annotations

import pytest
from datasets import Dataset, DatasetDict, Features, Value

from data.build_replay_corpus import replay_rows, with_replay


def _mnli(per_class: int) -> list[dict]:
    return [
        {
            "pairID": f"{label}-{i}",
            "premise": f"p{label}{i}",
            "hypothesis": f"h{label}{i}",
            "label": label,
            "genre": "g",
        }
        for label in (0, 1, 2)
        for i in range(per_class)
    ]


def test_rows_are_balanced_and_keep_their_label() -> None:
    rows = replay_rows(_mnli(5), per_class=3, seed=0)
    assert [r["polarity"] for r in rows].count(2.0) == 3
    assert len(rows) == 9
    for row in rows:
        label = int(row["item_id"].split(":")[1].split("-")[0])
        assert row["polarity"] == float(label)
        assert row["original"].startswith(f"p{label}") and row["simplification"].startswith(f"h{label}")
    assert {r["polarity_raw"] for r in rows if r["polarity"] == 2.0} == {"contradiction"}


def test_same_seed_same_rows() -> None:
    # Rows hold NaN, which never equals itself: compare the draws by their ids.
    first, second = replay_rows(_mnli(10), 4, 7), replay_rows(_mnli(10), 4, 7)
    assert [r["item_id"] for r in first] == [r["item_id"] for r in second]
    assert [r["item_id"] for r in first] != [r["item_id"] for r in replay_rows(_mnli(10), 4, 8)]


def test_short_class_is_refused() -> None:
    with pytest.raises(ValueError):
        replay_rows(_mnli(2), per_class=3, seed=0)


def test_only_train_grows() -> None:
    features = Features(
        {
            name: Value("float64") if name in ("label_raw", "label", "label_std", "polarity") else Value("string")
            for name in replay_rows(_mnli(1), 1, 0)[0]
            if name != "n_annotators"
        }
        | {"n_annotators": Value("int64")}
    )
    seed_row = replay_rows(_mnli(1), 1, 0)[0]
    split = Dataset.from_list([seed_row], features=features)
    splits = DatasetDict({"train": split, "dev": split, "test": split})
    out = with_replay(splits, replay_rows(_mnli(4), 2, 0))
    assert len(out["train"]) == 7
    assert len(out["dev"]) == 1 and len(out["test"]) == 1
    assert out["train"].features == split.features
