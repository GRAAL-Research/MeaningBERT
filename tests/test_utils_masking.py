import numpy as np
import pytest

from src.training.metrics.utils_masking import NonStopMasker


class TestNonStopMaskerComputeEffectiveMaskRatio:
    def test_all_masked(self):
        is_masked = [[1, 1, 1], [1, 1, 1]]
        result = NonStopMasker.compute_effective_mask_ratio(is_masked)
        assert result == pytest.approx(1.0)

    def test_none_masked(self):
        is_masked = [[0, 0, 0], [0, 0, 0]]
        result = NonStopMasker.compute_effective_mask_ratio(is_masked)
        assert result == pytest.approx(0.0)

    def test_half_masked(self):
        is_masked = [[1, 0], [0, 1]]
        result = NonStopMasker.compute_effective_mask_ratio(is_masked)
        assert result == pytest.approx(0.5)

    def test_empty_list(self):
        is_masked = []
        result = NonStopMasker.compute_effective_mask_ratio(is_masked)
        assert np.isnan(result)
