"""Tests for the multilingual magnitude proxies (``src/diagnostics/magnitude_proxies.py``).

The TextDetox rows are prompts with few-shot examples in front of the evaluated pair: the
parser must take the last pair, never an example. The script grouping stands in for a
language tag the rows do not carry.
"""

from __future__ import annotations

import numpy as np
import pytest

from diagnostics import magnitude_proxies
from diagnostics.magnitude_fr import bar_name
from diagnostics.magnitude_proxies import LATIN, NATIVE, parse_textdetox, script

PROMPT = (
    "You are given with two texts.\n\nExamples:\n"
    'text1: "Why do you behave like an asshole?"\ntext2: "Why do you act so bad?"\nAnswer: yes\n\n'
    'text1: "This political pary is a gang of idiots"\ntext2: "They just dont know their stuff"\nAnswer: no\n\n\n\n'
    "text1: эта сраная фишка встречается\ntext2: эта фишка встречается\nAnswer:"
)


def test_parse_textdetox_takes_the_evaluated_pair_not_an_example() -> None:
    assert parse_textdetox(PROMPT) == ("эта сраная фишка встречается", "эта фишка встречается")


def test_parse_textdetox_keeps_a_multiline_sentence() -> None:
    assert parse_textdetox("text1: a\nb\ntext2: c\nAnswer: ") == ("a\nb", "c")


def test_parse_textdetox_refuses_a_prompt_without_a_pair() -> None:
    with pytest.raises(ValueError):
        parse_textdetox("text1: seul\nAnswer:")


@pytest.mark.parametrize(
    ("text", "group"),
    [
        ("эта фишка", "cyrillique (ru, uk)"),
        ("هذا نص", "arabe"),
        ("यह पाठ", "devanagari (hi)"),
        ("ይህ ጽሑፍ", "éthiopien (am)"),
        ("这是文本", "chinois"),
        ("Das ist ein Text, ¿verdad?", LATIN),
    ],
)
def test_script_groups_each_textdetox_language(text: str, group: str) -> None:
    assert script(text) == group


def test_block_tests_every_candidate_against_the_native_bertscore(monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(1)
    gold = rng.normal(size=120)
    bar = bar_name(NATIVE)
    fake = {bar: gold + rng.normal(size=120), "fort": gold + rng.normal(scale=0.2, size=120)}
    seen = {}

    def candidates(left, right, left_en, right_en, en_label, trained_root, native):  # noqa: ANN001, ANN202
        seen.update(left_en=left_en, native=native)
        return fake

    monkeypatch.setattr(magnitude_proxies, "candidates", candidates)
    out = magnitude_proxies.block(["a"] * 120, ["b"] * 120, gold, "racine")
    assert seen == {"left_en": None, "native": NATIVE}
    assert out["n"] == 120 and set(out["candidates"]) == {bar, "fort"}
    assert set(out["versus_bar"]) == {"fort"} and out["versus_bar"]["fort"]["diff"] > 0
