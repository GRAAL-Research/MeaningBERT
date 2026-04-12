import argparse

import pytest

from src.training.tools import bool_parse


class TestBoolParse:
    @pytest.mark.parametrize("value", ["true", "True", "TRUE", "t", "T", "yes", "Yes", "y", "Y", "1"])
    def test_true_values(self, value):
        assert bool_parse(value) is True

    @pytest.mark.parametrize("value", ["false", "False", "FALSE", "f", "F", "no", "No", "n", "N", "0"])
    def test_false_values(self, value):
        assert bool_parse(value) is False

    @pytest.mark.parametrize("value", ["maybe", "2", "oui", "non", ""])
    def test_invalid_values_raise(self, value):
        with pytest.raises(argparse.ArgumentTypeError):
            bool_parse(value)
