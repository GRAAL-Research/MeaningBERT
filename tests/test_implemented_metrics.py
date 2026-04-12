import numpy as np
import pytest

from src.training.metrics.implemented_metrics import sigmoid


class TestSigmoid:
    def test_zero(self):
        assert sigmoid(0) == pytest.approx(0.5)

    def test_large_positive(self):
        assert sigmoid(100) == pytest.approx(1.0, abs=1e-6)

    def test_large_negative(self):
        assert sigmoid(-100) == pytest.approx(0.0, abs=1e-6)

    def test_symmetry(self):
        assert sigmoid(2) + sigmoid(-2) == pytest.approx(1.0, abs=1e-10)

    def test_array(self):
        result = sigmoid(np.array([-1, 0, 1]))
        assert result[1] == pytest.approx(0.5)
        assert result[0] < 0.5
        assert result[2] > 0.5
