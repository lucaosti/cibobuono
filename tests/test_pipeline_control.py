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
    return tmp_path


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


def test_cli_status(ctrl_env, capsys):
    rc = pc.main(["status"])
    assert rc == 0
    state = json.loads(capsys.readouterr().out)
    assert state["status"] == "idle"


def test_cli_pause_without_pipeline_fails(ctrl_env, capsys):
    rc = pc.main(["pause"])
    assert rc == 1
    assert "No pipeline running" in capsys.readouterr().out
