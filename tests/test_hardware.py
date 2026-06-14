"""
Tests for scripts.hardware — DeviceProfile detection across simulated platforms.

We monkey-patch the low-level detectors (platform.*, psutil, subprocess,
filesystem) so the same test file can verify every supported profile from a
single machine.
"""

from __future__ import annotations

__author__ = "Luca Ostinelli"

import subprocess
from pathlib import Path

import pytest

from scripts import hardware
from scripts.hardware import DevicePlatform, DeviceProfile, get_profile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_profile_cache():
    """Drop the cached profile before *and* after each test."""
    hardware.reset_profile_cache()
    yield
    hardware.reset_profile_cache()


def _stub_no_gpu(monkeypatch):
    """Make every GPU probe return 'absent'."""
    monkeypatch.setattr(hardware, "_detect_cuda", lambda: (False, None, None))
    monkeypatch.setattr(hardware, "_detect_rocm", lambda: (False, None, None))


def _stub_no_pi(monkeypatch):
    monkeypatch.setattr(hardware, "_detect_raspberry_pi", lambda: None)


def _stub_no_virt(monkeypatch):
    monkeypatch.setattr(hardware, "_detect_virtualization", lambda: (False, None))


def _force_platform(
    monkeypatch,
    *,
    system: str,
    machine: str,
    cpu_logical: int = 8,
    cpu_physical: int = 8,
    ram_gb: float = 16.0,
):
    monkeypatch.setattr(hardware.platform, "system", lambda: system)
    monkeypatch.setattr(hardware.platform, "machine", lambda: machine)
    monkeypatch.setattr(hardware.os, "cpu_count", lambda: cpu_logical)
    monkeypatch.setattr(
        hardware, "_detect_physical_cpu_count", lambda logical: cpu_physical
    )
    monkeypatch.setattr(hardware, "_detect_ram_gb", lambda: ram_gb)


# ---------------------------------------------------------------------------
# Apple Silicon (M-class)
# ---------------------------------------------------------------------------


class TestAppleSilicon:
    def test_m1_pro_16gb(self, monkeypatch):
        _force_platform(
            monkeypatch, system="Darwin", machine="arm64",
            cpu_logical=10, cpu_physical=10, ram_gb=16.0,
        )
        monkeypatch.setattr(
            hardware, "_detect_apple_silicon_cores", lambda: (8, 2)
        )
        _stub_no_gpu(monkeypatch)
        _stub_no_pi(monkeypatch)
        _stub_no_virt(monkeypatch)

        p = get_profile()

        assert p.platform == DevicePlatform.APPLE_SILICON
        assert p.has_metal is True
        assert p.has_cuda is False
        assert p.cpu_perf_cores == 8
        assert p.cpu_count_physical == 10  # P+E
        # faster-whisper has no Metal backend → CPU + int8
        assert p.whisper_device == "cpu"
        assert p.whisper_compute_type == "int8"
        assert p.whisper_cpu_threads == 8  # uses P-cores
        assert p.whisper_model == "large-v3-turbo"
        # llama.cpp uses Metal: full offload
        assert p.n_gpu_layers == -1
        assert p.n_threads == 8
        assert p.use_mlock is True
        assert p.use_mmap is True
        assert p.enable_llm is True
        assert p.llm_tier == "14B"  # 16 GB → 14B tier

    def test_m_class_32gb_picks_32b_tier(self, monkeypatch):
        _force_platform(
            monkeypatch, system="Darwin", machine="arm64",
            cpu_logical=12, cpu_physical=12, ram_gb=32.0,
        )
        monkeypatch.setattr(
            hardware, "_detect_apple_silicon_cores", lambda: (8, 4)
        )
        _stub_no_gpu(monkeypatch)
        _stub_no_pi(monkeypatch)
        _stub_no_virt(monkeypatch)

        p = get_profile()

        assert p.llm_tier == "32B"
        assert p.n_batch == 2048
        assert p.n_ctx == 8192


# ---------------------------------------------------------------------------
# Linux + NVIDIA CUDA
# ---------------------------------------------------------------------------


class TestLinuxCUDA:
    def test_rtx_4090_24gb(self, monkeypatch):
        _force_platform(
            monkeypatch, system="Linux", machine="x86_64",
            cpu_logical=32, cpu_physical=16, ram_gb=64.0,
        )
        monkeypatch.setattr(
            hardware, "_detect_cuda",
            lambda: (True, "NVIDIA GeForce RTX 4090", 24.0),
        )
        monkeypatch.setattr(hardware, "_detect_rocm", lambda: (False, None, None))
        _stub_no_pi(monkeypatch)
        _stub_no_virt(monkeypatch)

        p = get_profile()

        assert p.platform == DevicePlatform.LINUX_CUDA
        assert p.has_cuda is True
        assert p.has_metal is False
        assert p.gpu_vram_gb == 24.0
        assert p.whisper_device == "cuda"
        assert p.whisper_compute_type == "float16"  # VRAM ≥ 8 GB
        assert p.n_gpu_layers == -1
        assert p.n_threads == 15  # physical-1
        assert p.use_mlock is True
        assert p.enable_llm is True
        assert p.llm_tier == "72B"

    def test_small_vram_uses_int8_float16(self, monkeypatch):
        _force_platform(
            monkeypatch, system="Linux", machine="x86_64",
            cpu_logical=8, cpu_physical=4, ram_gb=8.0,
        )
        monkeypatch.setattr(
            hardware, "_detect_cuda",
            lambda: (True, "NVIDIA GeForce GTX 1660", 6.0),
        )
        monkeypatch.setattr(hardware, "_detect_rocm", lambda: (False, None, None))
        _stub_no_pi(monkeypatch)
        _stub_no_virt(monkeypatch)

        p = get_profile()

        assert p.whisper_device == "cuda"
        assert p.whisper_compute_type == "int8_float16"


# ---------------------------------------------------------------------------
# Raspberry Pi 5 / 8 GB
# ---------------------------------------------------------------------------


class TestRaspberryPi5:
    def test_pi5_8gb(self, monkeypatch):
        _force_platform(
            monkeypatch, system="Linux", machine="aarch64",
            cpu_logical=4, cpu_physical=4, ram_gb=8.0,
        )
        _stub_no_gpu(monkeypatch)
        monkeypatch.setattr(
            hardware, "_detect_raspberry_pi",
            lambda: "Raspberry Pi 5 Model B Rev 1.0",
        )
        _stub_no_virt(monkeypatch)

        p = get_profile()

        assert p.platform == DevicePlatform.RASPBERRY_PI
        assert p.pi_model is not None
        assert "Pi 5" in p.pi_model
        assert p.whisper_model == "small"
        assert p.whisper_device == "cpu"
        assert p.whisper_compute_type == "int8"
        assert p.n_gpu_layers == 0
        assert p.n_batch == 256
        assert p.use_mlock is False
        assert p.enable_llm is True
        # 8 GB → 8B tier still chosen (≥6 GB threshold)
        assert p.llm_tier in {"8B", "3B"}

    def test_pi3_1gb_disables_llm(self, monkeypatch):
        _force_platform(
            monkeypatch, system="Linux", machine="armv7l",
            cpu_logical=4, cpu_physical=4, ram_gb=1.0,
        )
        _stub_no_gpu(monkeypatch)
        monkeypatch.setattr(
            hardware, "_detect_raspberry_pi",
            lambda: "Raspberry Pi 3 Model B Plus Rev 1.3",
        )
        _stub_no_virt(monkeypatch)

        p = get_profile()

        assert p.platform == DevicePlatform.RASPBERRY_PI
        assert p.whisper_model == "tiny"
        assert p.enable_llm is False
        assert p.llm_tier == "none"


# ---------------------------------------------------------------------------
# Docker on Linux x86_64
# ---------------------------------------------------------------------------


class TestDockerOnLinuxX86:
    def test_docker_disables_mlock(self, monkeypatch):
        _force_platform(
            monkeypatch, system="Linux", machine="x86_64",
            cpu_logical=8, cpu_physical=4, ram_gb=16.0,
        )
        _stub_no_gpu(monkeypatch)
        _stub_no_pi(monkeypatch)
        monkeypatch.setattr(
            hardware, "_detect_virtualization", lambda: (True, "docker")
        )

        p = get_profile()

        assert p.platform == DevicePlatform.LINUX_X86_64
        assert p.is_virtual is True
        assert p.virt_type == "docker"
        # VMs/containers disable mlock
        assert p.use_mlock is False
        # And reserve a core for the host
        assert p.n_threads >= 2


# ---------------------------------------------------------------------------
# WSL2
# ---------------------------------------------------------------------------


class TestWSL2:
    def test_wsl2_no_gpu(self, monkeypatch):
        _force_platform(
            monkeypatch, system="Linux", machine="x86_64",
            cpu_logical=16, cpu_physical=8, ram_gb=16.0,
        )
        _stub_no_gpu(monkeypatch)
        _stub_no_pi(monkeypatch)
        monkeypatch.setattr(
            hardware, "_detect_virtualization", lambda: (True, "wsl")
        )

        p = get_profile()

        assert p.is_virtual is True
        assert p.virt_type == "wsl"
        assert p.use_mlock is False


# ---------------------------------------------------------------------------
# Intel Mac
# ---------------------------------------------------------------------------


class TestMacIntel:
    def test_intel_mac(self, monkeypatch):
        _force_platform(
            monkeypatch, system="Darwin", machine="x86_64",
            cpu_logical=8, cpu_physical=4, ram_gb=16.0,
        )
        _stub_no_gpu(monkeypatch)
        _stub_no_pi(monkeypatch)
        _stub_no_virt(monkeypatch)

        p = get_profile()

        assert p.platform == DevicePlatform.MAC_INTEL
        assert p.has_metal is False  # only arm64 Macs flagged
        assert p.has_cuda is False
        assert p.whisper_device == "cpu"
        assert p.whisper_compute_type == "int8"
        assert p.n_gpu_layers == 0


# ---------------------------------------------------------------------------
# Unknown platform
# ---------------------------------------------------------------------------


class TestUnknown:
    def test_unknown_system(self, monkeypatch):
        _force_platform(
            monkeypatch, system="FreeBSD", machine="x86_64",
            cpu_logical=4, cpu_physical=4, ram_gb=8.0,
        )
        _stub_no_gpu(monkeypatch)
        _stub_no_pi(monkeypatch)
        _stub_no_virt(monkeypatch)

        p = get_profile()

        assert p.platform == DevicePlatform.UNKNOWN
        # Safe defaults: CPU only, no GPU, LLM still enabled at 8 GB
        assert p.n_gpu_layers == 0
        assert p.enable_llm is True


# ---------------------------------------------------------------------------
# Singleton / caching
# ---------------------------------------------------------------------------


class TestSingleton:
    def test_get_profile_returns_same_instance(self, monkeypatch):
        _force_platform(
            monkeypatch, system="Linux", machine="x86_64",
            cpu_logical=4, cpu_physical=4, ram_gb=8.0,
        )
        _stub_no_gpu(monkeypatch)
        _stub_no_pi(monkeypatch)
        _stub_no_virt(monkeypatch)

        a = get_profile()
        b = get_profile()

        assert a is b

    def test_reset_cache_returns_fresh_object(self, monkeypatch):
        _force_platform(
            monkeypatch, system="Linux", machine="x86_64",
            cpu_logical=4, cpu_physical=4, ram_gb=8.0,
        )
        _stub_no_gpu(monkeypatch)
        _stub_no_pi(monkeypatch)
        _stub_no_virt(monkeypatch)

        a = get_profile()
        hardware.reset_profile_cache()
        b = get_profile()

        assert a is not b
        # but content equal
        assert a.platform == b.platform
        assert a.total_ram_gb == b.total_ram_gb


# ---------------------------------------------------------------------------
# Pure-function tests for helpers that don't need monkey-patching
# ---------------------------------------------------------------------------


def test_pick_llm_tier_ladder():
    """LLM tier ladder picks the largest model that fits the RAM."""
    assert hardware._pick_llm_tier(64, True) == "72B"
    assert hardware._pick_llm_tier(40, True) == "72B"
    assert hardware._pick_llm_tier(24, True) == "32B"
    assert hardware._pick_llm_tier(12, True) == "14B"
    assert hardware._pick_llm_tier(8, True) == "8B"
    assert hardware._pick_llm_tier(3, True) == "3B"
    assert hardware._pick_llm_tier(1.5, True) == "1B"
    assert hardware._pick_llm_tier(0.5, True) == "none"
    assert hardware._pick_llm_tier(64, False) == "none"


def test_to_dict_is_json_safe():
    """to_dict() returns a plain dict with primitive values."""
    import json

    profile = DeviceProfile(
        platform=DevicePlatform.UNKNOWN,
        system="x", machine="y",
        cpu_count_logical=1, cpu_count_physical=1, cpu_perf_cores=1,
        total_ram_gb=1.0,
        is_virtual=False, virt_type=None,
        pi_model=None,
        has_metal=False, has_cuda=False, has_rocm=False,
        gpu_name=None, gpu_vram_gb=None,
        whisper_device="cpu", whisper_compute_type="int8",
        whisper_cpu_threads=1, whisper_model="tiny",
        n_threads=2, n_gpu_layers=0, n_batch=256, n_ctx=1024,
        use_mlock=False, use_mmap=True,
        llm_tier="none", enable_llm=False,
        detection_notes=("a", "b"),
    )
    blob = json.dumps(profile.to_dict())
    assert "platform" in blob
    assert "unknown" in blob


def test_detect_raspberry_pi_handles_missing_path(monkeypatch, tmp_path: Path):
    """Reading a non-existent device-tree model returns None."""
    monkeypatch.setattr(hardware, "_PI_MODEL_PATH", tmp_path / "missing")
    assert hardware._detect_raspberry_pi() is None


def test_detect_raspberry_pi_parses_nul_terminated_string(
    monkeypatch, tmp_path: Path
):
    """device-tree model files end with a NUL byte; we strip it."""
    model_path = tmp_path / "model"
    model_path.write_text("Raspberry Pi 5 Model B Rev 1.0\x00")
    monkeypatch.setattr(hardware, "_PI_MODEL_PATH", model_path)
    assert hardware._detect_raspberry_pi() == "Raspberry Pi 5 Model B Rev 1.0"


def test_detect_cuda_returns_false_when_nvidia_smi_missing(monkeypatch):
    """No nvidia-smi binary and no PyTorch CUDA → has_cuda=False, no crash."""
    def _raise(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(subprocess, "run", _raise)

    # Also neutralise the PyTorch fallback so the test isn't environment-dependent.
    import sys
    import types

    fake_cuda = types.SimpleNamespace(
        is_available=lambda: False,
        device_count=lambda: 0,
    )
    fake_torch = types.SimpleNamespace(cuda=fake_cuda)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    has, name, vram = hardware._detect_cuda()
    assert has is False
    assert name is None
    assert vram is None
