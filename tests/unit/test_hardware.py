"""Tests for hardware capability gating (precision / attention resolution)."""

from unittest.mock import patch

import pytest

from baligh.utils.hardware import (
    get_device_capability,
    resolve_attn_implementation,
    resolve_precision,
)


class TestResolvePrecision:
    @patch("baligh.utils.hardware.get_device_capability", return_value=(7, 5))
    def test_bf16_falls_back_on_turing(self, _cap):
        assert resolve_precision("bf16") == "fp16"

    @patch("baligh.utils.hardware.get_device_capability", return_value=(8, 6))
    def test_bf16_kept_on_ampere(self, _cap):
        assert resolve_precision("bf16") == "bf16"

    @patch("baligh.utils.hardware.get_device_capability", return_value=(7, 5))
    def test_auto_selects_fp16_on_turing(self, _cap):
        assert resolve_precision("auto") == "fp16"

    @patch("baligh.utils.hardware.get_device_capability", return_value=(8, 0))
    def test_auto_selects_bf16_on_ampere(self, _cap):
        assert resolve_precision("auto") == "bf16"

    @patch("baligh.utils.hardware.get_device_capability", return_value=None)
    def test_cpu_auto_selects_no(self, _cap):
        assert resolve_precision("auto") == "no"

    def test_explicit_no_respected(self):
        assert resolve_precision("no") == "no"

    def test_explicit_fp16_respected(self):
        assert resolve_precision("fp16") == "fp16"

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError):
            resolve_precision("fp8")


class TestResolveAttnImplementation:
    @patch("baligh.utils.hardware.get_device_capability", return_value=(7, 5))
    def test_fa2_falls_back_to_sdpa_on_turing(self, _cap):
        assert resolve_attn_implementation("flash_attention_2") == "sdpa"

    @patch("baligh.utils.hardware.get_device_capability", return_value=(8, 6))
    def test_fa2_kept_on_ampere(self, _cap):
        assert resolve_attn_implementation("flash_attention_2") == "flash_attention_2"

    @patch("baligh.utils.hardware.get_device_capability", return_value=None)
    def test_cpu_gets_sdpa(self, _cap):
        assert resolve_attn_implementation("flash_attention_2") == "sdpa"

    def test_none_is_sdpa(self):
        assert resolve_attn_implementation(None) == "sdpa"

    def test_explicit_sdpa_passthrough(self):
        assert resolve_attn_implementation("sdpa") == "sdpa"


class TestGetDeviceCapability:
    @patch("torch.cuda.is_available", return_value=False)
    def test_none_without_cuda(self, _avail):
        assert get_device_capability() is None

    @patch("torch.cuda.get_device_capability", return_value=(7, 5))
    @patch("torch.cuda.is_available", return_value=True)
    def test_tuple_when_cuda(self, _avail, _cap):
        assert get_device_capability() == (7, 5)
