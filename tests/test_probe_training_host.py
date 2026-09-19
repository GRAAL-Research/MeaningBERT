"""Tests for the training-host capability derivation.

Collection is IO and is not tested here. What is tested is the part that would otherwise
be asserted from memory: which precision a compute capability allows, and which checkpoint
fits in how much memory.
"""

import pytest

from diagnostics.probe_training_host import (
    ACTIVATION_HEADROOM,
    BYTES_PER_PARAM_ADAMW_FP32,
    CHECKPOINT_PARAMS,
    Gpu,
    HostProbe,
    _parse_gpus,
    derive,
    render_markdown,
    trainable,
)

PASCAL_P5000 = Gpu(index=1, name="Quadro P5000", memory_mib=16384, compute_capability=6.1)
PASCAL_1080TI = Gpu(index=0, name="NVIDIA GeForce GTX 1080 Ti", memory_mib=11264, compute_capability=6.1)
AMPERE_A100 = Gpu(index=0, name="NVIDIA A100", memory_mib=81920, compute_capability=8.0)
TURING_T4 = Gpu(index=0, name="Tesla T4", memory_mib=15360, compute_capability=7.5)


# --- precision support ---------------------------------------------------------------


def test_pascal_has_no_bf16():
    assert not PASCAL_P5000.supports_bf16


def test_pascal_has_no_fp16_tensor_cores():
    """Pascal can store fp16 but gains no throughput from it, which changes the advice."""
    assert not PASCAL_1080TI.has_fp16_tensor_cores


def test_ampere_has_bf16():
    assert AMPERE_A100.supports_bf16


def test_turing_has_fp16_tensor_cores_but_no_bf16():
    assert TURING_T4.has_fp16_tensor_cores
    assert not TURING_T4.supports_bf16


def test_the_bf16_boundary_is_exactly_at_compute_capability_eight():
    assert not Gpu(0, "x", 16384, 7.9).supports_bf16
    assert Gpu(0, "x", 16384, 8.0).supports_bf16


# --- memory fit ----------------------------------------------------------------------


def test_deberta_v3_large_fits_on_the_p5000():
    fits, state = trainable(PASCAL_P5000, CHECKPOINT_PARAMS["microsoft/deberta-v3-large"])
    assert fits
    assert state == pytest.approx(6.5, abs=0.3)


def test_deberta_v2_xlarge_does_not_fit_on_the_p5000():
    """14.4 GB of optimiser state on a 16 GB card leaves nothing for activations."""
    fits, state = trainable(PASCAL_P5000, CHECKPOINT_PARAMS["microsoft/deberta-v2-xlarge"])
    assert not fits
    assert state > 13.0


def test_deberta_v3_large_clears_the_smaller_card_but_only_just():
    """6.5 GB of state against 7.1 GB usable on the 1080 Ti: it screens through with
    roughly 10 percent to spare, which is why the plan puts it on the 16 GB card."""
    fits, state = trainable(PASCAL_1080TI, CHECKPOINT_PARAMS["microsoft/deberta-v3-large"])
    usable_gb = (PASCAL_1080TI.memory_mib / 1024) * (1 - ACTIVATION_HEADROOM)
    assert fits
    assert state / usable_gb > 0.85


def test_deberta_v3_base_fits_on_both_cards():
    params = CHECKPOINT_PARAMS["microsoft/deberta-v3-base"]
    assert trainable(PASCAL_P5000, params)[0]
    assert trainable(PASCAL_1080TI, params)[0]


def test_deberta_v2_xlarge_fits_on_an_a100():
    assert trainable(AMPERE_A100, CHECKPOINT_PARAMS["microsoft/deberta-v2-xlarge"])[0]


def test_state_size_follows_the_documented_bytes_per_parameter():
    _, state = trainable(PASCAL_P5000, 1_000_000_000)
    assert state == pytest.approx(1e9 * BYTES_PER_PARAM_ADAMW_FP32 / 1024**3, rel=1e-9)


def test_headroom_is_actually_reserved():
    """A model needing exactly the full card must be refused, not accepted."""
    params = int(PASCAL_P5000.memory_mib / 1024 * 1024**3 / BYTES_PER_PARAM_ADAMW_FP32)
    assert not trainable(PASCAL_P5000, params)[0]
    just_under = int(params * (1 - ACTIVATION_HEADROOM) * 0.98)
    assert trainable(PASCAL_P5000, just_under)[0]


# --- derive --------------------------------------------------------------------------


def _probe(gpus):
    return HostProbe(host="renard", hostname="renard", gpus=gpus)


def test_derive_restricts_the_verdict_to_the_selected_gpu():
    verdict = derive(_probe([PASCAL_1080TI, PASCAL_P5000]), gpu_index=1)
    assert verdict["selected_gpu_indices"] == [1]
    assert verdict["checkpoint_fit"]["microsoft/deberta-v3-large"][1]["fits"]


def test_selecting_the_smaller_card_changes_the_verdict():
    """v3-large screens through on both cards; v3-base is the one that separates them."""
    small = derive(_probe([PASCAL_1080TI, PASCAL_P5000]), gpu_index=0)
    large = derive(_probe([PASCAL_1080TI, PASCAL_P5000]), gpu_index=1)
    assert small["selected_gpu_indices"] == [0]
    assert 0 in small["checkpoint_fit"]["microsoft/deberta-v3-large"]
    assert 1 not in small["checkpoint_fit"]["microsoft/deberta-v3-large"]
    assert not small["checkpoint_fit"]["microsoft/deberta-v2-xlarge"][0]["fits"]
    assert not large["checkpoint_fit"]["microsoft/deberta-v2-xlarge"][1]["fits"]


def test_derive_recommends_fp32_on_pascal():
    assert derive(_probe([PASCAL_P5000]), gpu_index=1)["recommended_precision"] == "fp32"


def test_derive_recommends_bf16_on_ampere():
    assert derive(_probe([AMPERE_A100]), gpu_index=0)["recommended_precision"] == "bf16"


def test_a_mixed_fleet_falls_back_to_the_weakest_card():
    """Selecting both an Ampere and a Pascal must not advertise bf16."""
    verdict = derive(_probe([PASCAL_P5000, AMPERE_A100]), gpu_index=None)
    assert not verdict["bf16"]


def test_derive_on_a_host_without_gpu_claims_nothing():
    verdict = derive(_probe([]), gpu_index=None)
    assert verdict["selected_gpu_indices"] == []
    assert not verdict["bf16"]
    assert verdict["recommended_precision"] == "fp32"


# --- parsing -------------------------------------------------------------------------


def test_parse_gpus_reads_the_nvidia_smi_csv():
    raw = "0, NVIDIA GeForce GTX 1080 Ti, 11264 MiB, 6.1\n1, Quadro P5000, 16384 MiB, 6.1"
    gpus = _parse_gpus(raw)
    assert [g.index for g in gpus] == [0, 1]
    assert gpus[1].memory_mib == 16384
    assert gpus[1].compute_capability == 6.1


def test_parse_gpus_skips_malformed_lines_instead_of_crashing():
    raw = "garbage\n1, Quadro P5000, 16384 MiB, 6.1\n2, broken, notanumber MiB, 6.1"
    assert [g.index for g in _parse_gpus(raw)] == [1]


def test_parse_gpus_on_empty_input_returns_nothing():
    assert _parse_gpus("") == []


# --- rendering -----------------------------------------------------------------------


def test_the_sheet_warns_about_bf16_flags_when_bf16_is_absent():
    probe = _probe([PASCAL_P5000])
    markdown = render_markdown(probe, derive(probe, gpu_index=1))
    assert "--bf16" in markdown
    assert "CUDA_VISIBLE_DEVICES=1" in markdown


def test_the_sheet_omits_the_bf16_warning_when_bf16_is_available():
    probe = _probe([AMPERE_A100])
    markdown = render_markdown(probe, derive(probe, gpu_index=0))
    assert "silencieusement en fp32" not in markdown


def test_the_sheet_marks_which_gpu_was_retained():
    probe = _probe([PASCAL_1080TI, PASCAL_P5000])
    markdown = render_markdown(probe, derive(probe, gpu_index=1))
    retained = [line for line in markdown.splitlines() if "(retenue)" in line]
    assert len(retained) == 1
    assert "P5000" in retained[0]


def test_the_sheet_surfaces_probe_errors():
    probe = _probe([])
    probe.errors.append("nvidia-smi unavailable")
    assert "nvidia-smi unavailable" in render_markdown(probe, derive(probe))
