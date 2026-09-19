"""Tests for GPU capability derivation.

Pure functions of the compute capability, so no GPU and no torch are needed. These
thresholds are what stop the training code from hard-coding ``--bf16`` on a card that has
no bfloat16.
"""

import pytest

from diagnostics.hardware import (
    BF16_MIN_CAPABILITY,
    TRITON_MIN_CAPABILITY,
    DeviceCapability,
    assert_arch_compiled,
    blocked_features,
    describe,
    recommended_attention_backends,
    recommended_compile_backend,
    recommended_precision,
)

P5000 = DeviceCapability(name="Quadro P5000", major=6, minor=1, total_memory_gb=16.0)
V100 = DeviceCapability(name="Tesla V100", major=7, minor=0, total_memory_gb=32.0)
T4 = DeviceCapability(name="Tesla T4", major=7, minor=5, total_memory_gb=15.0)
A100 = DeviceCapability(name="NVIDIA A100", major=8, minor=0, total_memory_gb=80.0)
H100 = DeviceCapability(name="NVIDIA H100", major=9, minor=0, total_memory_gb=80.0)


# --- capability arithmetic -----------------------------------------------------------


def test_capability_reads_as_a_comparable_number():
    assert P5000.capability == pytest.approx(6.1)


def test_arch_tag_matches_what_torch_reports():
    assert P5000.arch == "sm_61"
    assert A100.arch == "sm_80"


def test_a_two_digit_minor_is_not_expected_but_does_not_crash_the_tag():
    assert DeviceCapability("x", 12, 0, 1.0).arch == "sm_120"


# --- the three thresholds ------------------------------------------------------------


def test_pascal_has_none_of_the_three_accelerations():
    assert not P5000.supports_bf16
    assert not P5000.supports_triton
    assert not P5000.supports_flash_attention
    assert not P5000.has_fp16_tensor_cores


def test_volta_unlocks_triton_and_fp16_tensor_cores_but_not_bf16():
    assert V100.supports_triton
    assert V100.has_fp16_tensor_cores
    assert not V100.supports_bf16
    assert not V100.supports_flash_attention


def test_ampere_unlocks_everything():
    assert A100.supports_bf16
    assert A100.supports_triton
    assert A100.supports_flash_attention


def test_hopper_is_still_fully_supported():
    assert H100.supports_bf16 and H100.supports_triton and H100.supports_flash_attention


def test_the_triton_boundary_is_exact():
    assert not DeviceCapability("x", 6, 9, 1.0).supports_triton
    assert DeviceCapability("x", 7, 0, 1.0).supports_triton


def test_the_bf16_boundary_is_exact():
    assert not DeviceCapability("x", 7, 5, 1.0).supports_bf16
    assert DeviceCapability("x", 8, 0, 1.0).supports_bf16


# --- recommendations -----------------------------------------------------------------


def test_pascal_trains_in_fp32():
    assert recommended_precision(P5000) == "fp32"


def test_ampere_trains_in_bf16():
    assert recommended_precision(A100) == "bf16"


def test_fp16_is_never_recommended():
    """Where fp16 is fast, bf16 exists and is safer; where bf16 is missing, fp16 is slow."""
    for device in (P5000, V100, T4, A100, H100):
        assert recommended_precision(device) != "fp16"


def test_turing_gets_fp32_despite_having_fp16_tensor_cores():
    assert T4.has_fp16_tensor_cores
    assert recommended_precision(T4) == "fp32"


def test_compilation_is_skipped_on_pascal():
    assert recommended_compile_backend(P5000) is None


def test_compilation_uses_inductor_from_volta_on():
    assert recommended_compile_backend(V100) == "inductor"


def test_flash_attention_is_offered_only_from_ampere():
    assert "flash" not in recommended_attention_backends(P5000)
    assert "flash" not in recommended_attention_backends(T4)
    assert recommended_attention_backends(A100)[0] == "flash"


def test_the_math_backend_is_always_available_as_a_last_resort():
    for device in (P5000, V100, T4, A100):
        assert "math" in recommended_attention_backends(device)


# --- blocked features ----------------------------------------------------------------


def test_pascal_blocks_all_four_features_with_a_reason_each():
    blocked = blocked_features(P5000)
    assert set(blocked) == {"bf16", "torch.compile", "flash_attention", "fp16_speedup"}
    assert str(BF16_MIN_CAPABILITY) in blocked["bf16"]
    assert str(TRITON_MIN_CAPABILITY) in blocked["torch.compile"]


def test_the_reason_names_the_actual_device_capability():
    assert "6.1" in blocked_features(P5000)["bf16"]


def test_ampere_blocks_nothing():
    assert blocked_features(A100) == {}


def test_volta_blocks_only_bf16_and_flash():
    assert set(blocked_features(V100)) == {"bf16", "flash_attention"}


# --- arch compiled into the wheel ----------------------------------------------------


def test_a_wheel_containing_the_arch_passes():
    assert_arch_compiled(P5000, ["sm_50", "sm_61", "sm_70", "sm_80"])


def test_a_wheel_missing_the_arch_fails_with_an_actionable_message():
    with pytest.raises(RuntimeError, match="no sm_61 kernels"):
        assert_arch_compiled(P5000, ["sm_75", "sm_80", "sm_90"])


def test_that_message_points_at_the_channel_that_still_ships_pascal():
    with pytest.raises(RuntimeError, match="cu126"):
        assert_arch_compiled(P5000, ["sm_80"])


def test_an_empty_arch_list_is_a_failure_not_a_pass():
    with pytest.raises(RuntimeError):
        assert_arch_compiled(P5000, [])


# --- description ---------------------------------------------------------------------


def test_the_description_states_the_precision_and_the_blockers():
    text = describe(P5000)
    assert "fp32" in text
    assert "sm_61" in text
    assert "Triton unavailable" in text


def test_the_description_of_a_modern_card_lists_no_blockers():
    assert "blocked" not in describe(A100)
