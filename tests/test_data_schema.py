"""Tests for the CSMD v2 loader contract validator."""

import math

import pytest
from datasets import Dataset

from data.schema import SCALES, ContractError, build, validate


def _row(**overrides) -> dict:
    """A minimal contract-valid row, overridable field by field."""
    row = {
        "item_id": "1",
        "original": "The cat sat on the mat.",
        "simplification": "The cat was on the mat.",
        "label_raw": 87.0,
        "scale": "da100",
        "domain": "wiki",
        "license": "CC-BY-4.0",
    }
    row.update(overrides)
    return row


def _dataset(rows: list[dict], corpus: str = "demo") -> Dataset:
    return build(rows, corpus)


# --- build ---------------------------------------------------------------------------


def test_build_leaves_label_nan_so_harmonize_owns_it():
    dataset = _dataset([_row()])
    assert math.isnan(dataset["label"][0])


def test_build_prefixes_item_id_with_corpus():
    dataset = _dataset([_row(item_id="42")], corpus="simpeval")
    assert dataset["item_id"] == ["simpeval:42"]


def test_build_does_not_double_prefix_an_already_prefixed_id():
    dataset = _dataset([_row(item_id="simpeval:42")], corpus="simpeval")
    assert dataset["item_id"] == ["simpeval:42"]


def test_build_strips_surrounding_whitespace_from_texts():
    dataset = _dataset([_row(original="  a sentence  ", simplification="\tanother\n")])
    assert dataset["original"] == ["a sentence"]
    assert dataset["simplification"] == ["another"]


def test_build_applies_documented_defaults():
    dataset = _dataset([_row()])
    assert dataset["source"] == ["original"]
    assert dataset["system"] == [""]
    assert dataset["split_hint"] == [""]
    assert dataset["n_annotators"] == [0]
    assert math.isnan(dataset["label_std"][0])


def test_build_rejects_a_row_missing_a_required_key():
    with pytest.raises(ContractError, match="missing required keys"):
        _dataset([{"item_id": "1", "original": "a", "simplification": "b"}])


# --- validate: happy path ------------------------------------------------------------


def test_validate_accepts_a_well_formed_dataset():
    validate(_dataset([_row(item_id="1"), _row(item_id="2", label_raw=12.5)]))


def test_validate_accepts_every_declared_scale_at_its_bounds():
    # 'none' is excluded on purpose: it declares the ABSENCE of a meaning-preservation
    # annotation, so it has no bounds to sit at. Its own rule is tested just below.
    for scale, spec in SCALES.items():
        if scale == "none":
            continue
        high = spec.high if math.isfinite(spec.high) else 99.0
        rows = [
            _row(item_id="lo", scale=scale, label_raw=spec.low),
            _row(item_id="hi", scale=scale, label_raw=high),
        ]
        validate(_dataset(rows))


# --- validate: the rules that actually matter ----------------------------------------


def test_validate_rejects_a_filled_label():
    dataset = _dataset([_row()])
    dataset = dataset.map(lambda _: {"label": 87.0})
    with pytest.raises(ContractError, match="harmonize.py owns that column"):
        validate(dataset)


def test_validate_rejects_label_raw_above_the_scale_bound():
    with pytest.raises(ContractError, match=r"outside \[1.0, 5.0\]"):
        validate(_dataset([_row(scale="likert5", label_raw=7.0)]))


def test_validate_rejects_label_raw_below_the_scale_bound():
    with pytest.raises(ContractError, match=r"outside \[0.0, 100.0\]"):
        validate(_dataset([_row(scale="da100", label_raw=-1.0)]))


def test_validate_rejects_a_non_finite_label_raw():
    with pytest.raises(ContractError, match="non-finite"):
        validate(_dataset([_row(label_raw=float("nan"))]))


def test_validate_rejects_an_unknown_scale():
    with pytest.raises(ContractError, match="unknown scale"):
        validate(_dataset([_row(scale="likert11", label_raw=3.0)]))


def test_validate_rejects_a_loader_mixing_two_scales():
    rows = [_row(item_id="1", scale="da100", label_raw=50.0), _row(item_id="2", scale="likert5", label_raw=3.0)]
    with pytest.raises(ContractError, match="single scale"):
        validate(_dataset(rows))


def test_validate_rejects_duplicate_item_ids():
    with pytest.raises(ContractError, match="item_id is not unique"):
        validate(_dataset([_row(item_id="1"), _row(item_id="1", label_raw=20.0)]))


def test_validate_rejects_an_empty_simplification():
    with pytest.raises(ContractError, match="simplification: 1 empty"):
        validate(_dataset([_row(simplification="   ")]))


def test_validate_rejects_an_unknown_domain():
    with pytest.raises(ContractError, match="domain: unknown value"):
        validate(_dataset([_row(domain="legal")]))


def test_validate_rejects_an_unknown_source_tag():
    with pytest.raises(ContractError, match="source: unknown value"):
        validate(_dataset([_row(source="paraphrase")]))


def test_validate_rejects_an_unknown_split_hint():
    with pytest.raises(ContractError, match="split_hint: unknown value"):
        validate(_dataset([_row(split_hint="validation")]))


def test_validate_rejects_negative_annotator_counts():
    with pytest.raises(ContractError, match="n_annotators: 1 negative"):
        validate(_dataset([_row(n_annotators=-1)]))


def test_validate_rejects_a_negative_std_when_annotators_are_declared():
    with pytest.raises(ContractError, match="label_std: 1 negative"):
        validate(_dataset([_row(n_annotators=3, label_std=-0.5)]))


def test_validate_rejects_an_empty_dataset():
    with pytest.raises(ContractError, match="dataset is empty"):
        validate(build([], "demo"))


def test_validate_rejects_a_missing_column():
    dataset = _dataset([_row()]).remove_columns(["license"])
    with pytest.raises(ContractError, match=r"missing columns: \['license'\]"):
        validate(dataset)


def test_validate_rejects_an_extra_column():
    dataset = _dataset([_row()]).add_column("annotator_notes", ["whatever"])
    with pytest.raises(ContractError, match=r"unexpected columns: \['annotator_notes'\]"):
        validate(dataset)


def test_validate_rejects_two_corpora_in_one_loader_output():
    dataset = _dataset([_row(item_id="1"), _row(item_id="2")])
    dataset = Dataset.from_dict({**dataset.to_dict(), "corpus": ["demo", "other"]})
    with pytest.raises(ContractError, match="single corpus"):
        validate(dataset)


def test_validate_reports_every_problem_at_once_not_just_the_first():
    rows = [_row(item_id="1", domain="legal", source="paraphrase", n_annotators=-2)]
    with pytest.raises(ContractError) as excinfo:
        validate(_dataset(rows))
    message = str(excinfo.value)
    assert "domain: unknown value" in message
    assert "source: unknown value" in message
    assert "n_annotators: 1 negative" in message


# --- likert3_signed, added for the PLABA expert judgements ---------------------------


def test_the_signed_likert_scale_accepts_its_three_native_values():
    for value in (-1.0, 0.0, 1.0):
        validate(_dataset([_row(scale="likert3_signed", label_raw=value)]))


def test_the_signed_likert_scale_rejects_a_value_outside_minus_one_to_one():
    with pytest.raises(ContractError, match=r"outside \[-1.0, 1.0\]"):
        validate(_dataset([_row(scale="likert3_signed", label_raw=2.0)]))


def test_the_signed_likert_scale_is_oriented_higher_is_better():
    """Unlike severity3, which shares its cardinality but not its orientation."""
    assert SCALES["likert3_signed"].higher_is_better
    assert not SCALES["severity3"].higher_is_better


# --- v3: the polarity half of the contract -------------------------------------------


def _polar(**overrides) -> dict:
    """A row from a polarity-only corpus: no preservation label, an NLI class."""
    defaults = {
        "scale": "none",
        "label_raw": float("nan"),
        "polarity_scheme": "nli3",
        "polarity_raw": "contradiction",
    }
    defaults.update(overrides)
    return _row(**defaults)


def test_a_polarity_only_corpus_is_valid_without_any_preservation_label():
    validate(_dataset([_polar(item_id="a"), _polar(item_id="b", polarity_raw="entailment")]))


def test_scale_none_rejects_a_label_raw_that_is_not_nan():
    # The failure this guards against is a loader defaulting label_raw to 0.0, which reads
    # as "no meaning preserved" instead of "not measured" and trains the model on a lie.
    with pytest.raises(ContractError, match="not NaN"):
        validate(_dataset([_polar(label_raw=0.0)]))


def test_a_v2_loader_that_says_nothing_about_polarity_stays_valid():
    # The whole point of the defaults: the four v2 loaders were not touched for v3.
    dataset = _dataset([_row()])
    assert set(dataset["polarity_scheme"]) == {"none"}
    assert set(dataset["polarity_raw"]) == {""}
    validate(dataset)


def test_build_leaves_polarity_nan_so_harmonize_owns_it():
    dataset = _dataset([_polar()])
    assert all(math.isnan(value) for value in dataset["polarity"])


def test_validate_rejects_a_filled_polarity():
    dataset = _dataset([_polar()]).map(lambda _: {"polarity": 2.0})
    with pytest.raises(ContractError, match="harmonize.py owns that column"):
        validate(dataset)


def test_validate_rejects_a_raw_value_outside_its_declared_scheme():
    # REFUTES is a real polarity class, but it belongs to fact3 and not to nli3. Accepting
    # it here is how two label spaces quietly merge into one incoherent one.
    with pytest.raises(ContractError, match="outside scheme 'nli3'"):
        validate(_dataset([_polar(polarity_raw="REFUTES")]))


def test_validate_rejects_an_unknown_polarity_scheme():
    with pytest.raises(ContractError, match="unknown polarity_scheme"):
        validate(_dataset([_polar(polarity_scheme="entail_or_not")]))


def test_validate_rejects_a_loader_mixing_two_polarity_schemes():
    rows = [_polar(item_id="a"), _polar(item_id="b", polarity_scheme="fact3", polarity_raw="REFUTES")]
    with pytest.raises(ContractError, match="single polarity_scheme"):
        validate(_dataset(rows))


def test_validate_rejects_a_corpus_that_annotates_neither_target():
    rows = [_row(scale="none", label_raw=float("nan"))]
    with pytest.raises(ContractError, match="at least one target"):
        validate(_dataset(rows))
