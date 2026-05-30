"""
Tests for scripts.resource_monitor — live runtime governance.

We drive the adaptive logic by crafting ResourceSnapshot values and a fake
DeviceProfile, so the same test verifies "fits / wait / downgrade" behaviour
deterministically on any machine.
"""

from __future__ import annotations

__author__ = "Luca Ostinelli"

import types
from pathlib import Path

import pytest

from scripts import resource_monitor as rm
from scripts.resource_monitor import ResourceSnapshot


@pytest.fixture(autouse=True)
def _no_wait(monkeypatch):
    """These tests exercise the decision logic, not the back-pressure wait;
    collapse the headroom wait so 'wait, then downgrade' resolves instantly."""
    monkeypatch.setattr(rm, "HEADROOM_WAIT_SECONDS", 0.0)
    monkeypatch.setattr(rm, "POLL_SECONDS", 0.01)


def _snap(**kw) -> ResourceSnapshot:
    base = dict(
        ram_total_gb=32.0,
        ram_available_gb=24.0,
        ram_used_percent=25.0,
        swap_used_gb=0.0,
        swap_used_percent=0.0,
        load_per_core=0.2,
        cpu_count=8,
        gpu_total_gb=None,
        gpu_free_gb=None,
        gpu_used_percent=None,
    )
    base.update(kw)
    return ResourceSnapshot(**base)


def _profile(**kw):
    base = dict(
        has_cuda=False,
        has_metal=False,
        n_gpu_layers=0,
        use_mlock=False,
        is_virtual=False,
        whisper_device="cpu",
    )
    base.update(kw)
    return types.SimpleNamespace(**base)


# ---------------------------------------------------------------------------
# snapshot()
# ---------------------------------------------------------------------------


def test_snapshot_real_machine_is_sane():
    s = rm.snapshot(include_gpu=False)
    assert s.ram_total_gb > 0
    assert 0 <= s.ram_available_gb <= s.ram_total_gb + 0.01
    assert 0 <= s.ram_used_percent <= 100
    assert s.cpu_count >= 1
    assert "free" in s.summary()


# ---------------------------------------------------------------------------
# under_pressure()
# ---------------------------------------------------------------------------


class TestUnderPressure:
    def test_calm(self):
        stressed, why = rm.under_pressure(_snap())
        assert stressed is False
        assert why == ""

    def test_high_ram(self):
        stressed, why = rm.under_pressure(_snap(ram_used_percent=95.0))
        assert stressed is True
        assert "RAM" in why

    def test_swap(self):
        stressed, why = rm.under_pressure(_snap(swap_used_percent=40.0))
        assert stressed is True
        assert "swap" in why

    def test_load(self):
        stressed, why = rm.under_pressure(_snap(load_per_core=5.0))
        assert stressed is True
        assert "load" in why

    def test_vram(self):
        stressed, why = rm.under_pressure(
            _snap(gpu_total_gb=24.0, gpu_free_gb=1.0, gpu_used_percent=96.0)
        )
        assert stressed is True
        assert "VRAM" in why


# ---------------------------------------------------------------------------
# plan_llm_load() — RAM pool
# ---------------------------------------------------------------------------


class TestPlanLlmLoadRam:
    def test_no_models(self):
        plan = rm.plan_llm_load(_profile(), [])
        assert plan.model_path is None
        assert "no GGUF" in plan.note

    def test_largest_fits_free(self, monkeypatch):
        monkeypatch.setattr(rm, "snapshot", lambda **_: _snap(ram_available_gb=24.0))
        models = [(Path("big-32B.gguf"), 14.0), (Path("small-8B.gguf"), 5.0)]
        plan = rm.plan_llm_load(_profile(), models)
        assert plan.model_path.name == "big-32B.gguf"
        assert plan.pool == "RAM"
        assert plan.n_gpu_layers == 0  # CPU profile

    def test_downgrade_when_free_low(self, monkeypatch):
        # 32 GB machine but only ~10 GB free right now → must pick the 8B model.
        monkeypatch.setattr(rm, "snapshot", lambda **_: _snap(ram_available_gb=10.0))
        models = [(Path("big-32B.gguf"), 14.0), (Path("small-8B.gguf"), 5.0)]
        plan = rm.plan_llm_load(_profile(), models)
        assert plan.model_path.name == "small-8B.gguf"
        assert "downgrad" in plan.note

    def test_capacity_ceiling_picks_smallest(self, monkeypatch):
        # Total RAM smaller than every model → smallest, with a clear note.
        monkeypatch.setattr(
            rm, "snapshot", lambda **_: _snap(ram_total_gb=4.0, ram_available_gb=3.0)
        )
        models = [(Path("a-32B.gguf"), 14.0), (Path("b-8B.gguf"), 5.0)]
        plan = rm.plan_llm_load(_profile(), models)
        assert plan.model_path.name == "b-8B.gguf"
        assert "capacity" in plan.note


# ---------------------------------------------------------------------------
# plan_llm_load() — VRAM pool (CUDA offload)
# ---------------------------------------------------------------------------


class TestPlanLlmLoadVram:
    def test_uses_vram_and_full_offload(self, monkeypatch):
        monkeypatch.setattr(
            rm, "snapshot",
            lambda **_: _snap(gpu_total_gb=24.0, gpu_free_gb=20.0, gpu_used_percent=16.0),
        )
        prof = _profile(has_cuda=True, n_gpu_layers=-1)
        models = [(Path("m-8B.gguf"), 5.0)]
        plan = rm.plan_llm_load(prof, models)
        assert plan.pool == "VRAM"
        assert plan.n_gpu_layers == -1

    def test_cpu_fallback_when_vram_too_small(self, monkeypatch):
        # Model fits VRAM capacity but not the free VRAM, and no wait helps.
        monkeypatch.setattr(rm, "HEADROOM_WAIT_SECONDS", 0.0)
        monkeypatch.setattr(
            rm, "snapshot",
            lambda **_: _snap(gpu_total_gb=24.0, gpu_free_gb=2.0, gpu_used_percent=92.0),
        )
        prof = _profile(has_cuda=True, n_gpu_layers=-1)
        models = [(Path("m-8B.gguf"), 5.0)]
        plan = rm.plan_llm_load(prof, models)
        assert plan.pool == "VRAM"
        assert plan.n_gpu_layers == 0  # can't fit free VRAM → CPU


# ---------------------------------------------------------------------------
# mlock gating
# ---------------------------------------------------------------------------


class TestMlock:
    def test_off_by_default(self, monkeypatch):
        monkeypatch.setattr(rm, "snapshot", lambda **_: _snap(ram_available_gb=24.0))
        plan = rm.plan_llm_load(_profile(use_mlock=True), [(Path("m-8B.gguf"), 5.0)])
        assert plan.use_mlock is False

    def test_on_when_allowed_and_headroom(self, monkeypatch):
        monkeypatch.setattr(rm, "ALLOW_MLOCK", True)
        monkeypatch.setattr(rm, "snapshot", lambda **_: _snap(ram_available_gb=24.0))
        plan = rm.plan_llm_load(_profile(use_mlock=True), [(Path("m-8B.gguf"), 5.0)])
        assert plan.use_mlock is True

    def test_off_when_headroom_thin(self, monkeypatch):
        monkeypatch.setattr(rm, "ALLOW_MLOCK", True)
        # free 9 GB, model 5 GB → fits, but only ~4 GB left afterwards,
        # below MLOCK_MIN_FREE_AFTER_GB (6) → mlock must stay off.
        monkeypatch.setattr(rm, "snapshot", lambda **_: _snap(ram_available_gb=9.0))
        plan = rm.plan_llm_load(_profile(use_mlock=True), [(Path("m-8B.gguf"), 5.0)])
        assert plan.use_mlock is False


# ---------------------------------------------------------------------------
# fit_whisper_model()
# ---------------------------------------------------------------------------


class TestFitWhisper:
    def test_keeps_requested_when_fits(self, monkeypatch):
        monkeypatch.setattr(rm, "snapshot", lambda **_: _snap(ram_available_gb=16.0))
        name, note = rm.fit_whisper_model(_profile(), "large-v3-turbo")
        assert name == "large-v3-turbo"

    def test_downgrades_when_tight(self, monkeypatch):
        monkeypatch.setattr(rm, "snapshot", lambda **_: _snap(ram_available_gb=1.0))
        name, note = rm.fit_whisper_model(_profile(), "large-v3-turbo")
        assert name in ("small", "base", "tiny")
        assert "downgrad" in note or "smallest" in note

    def test_unknown_left_alone(self):
        name, note = rm.fit_whisper_model(_profile(), "mystery")
        assert name == "mystery"


# ---------------------------------------------------------------------------
# list_gguf_models + priority
# ---------------------------------------------------------------------------


def test_list_gguf_models_sorted(tmp_path):
    (tmp_path / "a.gguf").write_bytes(b"x" * 1024)
    (tmp_path / "b.gguf").write_bytes(b"x" * 4096)
    (tmp_path / "notamodel.txt").write_text("nope")
    models = rm.list_gguf_models(tmp_path)
    assert [p.name for p, _ in models] == ["b.gguf", "a.gguf"]


def test_apply_friendly_priority_idempotent():
    rm._priority_applied = False
    rm.apply_friendly_priority()
    rm.apply_friendly_priority()  # must not raise or lower priority twice
    assert rm._priority_applied is True
