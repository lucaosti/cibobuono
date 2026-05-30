"""Tests for pipeline_control module."""

from __future__ import annotations

import json

import pytest

from scripts import pipeline_control as pc


@pytest.fixture
def ctrl_env(tmp_path, monkeypatch):
    monkeypatch.setattr(pc, "CONTROL_PATH", tmp_path / "control.json")
    monkeypatch.setattr(pc, "PID_PATH", tmp_path / "pipeline.pid")
    monkeypatch.setattr(pc, "LOGS_DIR", tmp_path)
    for name, rel in pc.EDITABLE_FILES.items():
        if name.endswith(".txt"):
            rel.parent.mkdir(parents=True, exist_ok=True)
            rel.write_text("https://youtube.com/@test\n", encoding="utf-8")
        else:
            rel.parent.mkdir(parents=True, exist_ok=True)
            rel.write_text("[]", encoding="utf-8")
    return tmp_path


def test_write_and_read_editable_json(ctrl_env, monkeypatch):
    monkeypatch.setattr(
        pc,
        "EDITABLE_FILES",
        {"locales.json": ctrl_env / "locales.json"},
    )
    data = [{"locale_id": "x", "name": "Test"}]
    pc.write_editable("locales.json", data)
    loaded, kind = pc.read_editable("locales.json")
    assert kind == "json"
    assert loaded == data


def test_pause_resume(ctrl_env, monkeypatch):
    monkeypatch.setattr(pc, "_pid_alive", lambda _pid: True)
    pc.mark_started(pid=999, max_videos=3)
    ok, _ = pc.request_pause()
    assert ok
    assert pc.read_state()["pause_requested"] is True
    ok, _ = pc.request_resume()
    assert ok
    assert pc.read_state()["pause_requested"] is False


def test_wait_if_paused_unblocks(ctrl_env, monkeypatch):
    pc.write_state({**pc.default_state(), "pause_requested": False})
    assert pc.wait_if_paused() is True
