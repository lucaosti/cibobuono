"""
Tests for asr_batch stages: the GPU stage must never touch the network.
"""

__author__ = "Luca Ostinelli"

import json
import sys

import pytest

import scripts.asr_batch as ab


@pytest.fixture
def batch(tmp_path, monkeypatch):
    monkeypatch.setattr(ab, "CACHE_DIR", tmp_path)
    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps([{"video_id": "a", "duration": 60}, {"video_id": "b", "duration": 60}]))
    calls = {"download": [], "subs": [], "asr": []}
    monkeypatch.setattr(ab, "download_audio", lambda v, url: calls["download"].append(v) or tmp_path / f"{v}.wav")
    monkeypatch.setattr(ab, "_download_youtube_manual_subs", lambda v: calls["subs"].append(v))
    monkeypatch.setattr(ab, "find_audio_file", lambda v: tmp_path / f"{v}.wav" if v == "a" else None)

    def fake_transcribe(v, model, fetch_subs=True):
        calls["asr"].append((v, fetch_subs))
        return {"segments": [{}], "source": "faster_whisper"}

    monkeypatch.setattr(ab, "transcribe_audio", fake_transcribe)
    monkeypatch.setattr(ab, "_cleanup_audio", lambda v: None)

    def run(stage):
        argv = ["asr_batch", "--meta", str(meta), "--times", str(tmp_path / "t.jsonl"), "--stage", stage]
        monkeypatch.setattr(sys, "argv", argv)
        return ab.main()

    return run, calls


def test_download_stage_fetches_audio_and_subs_only(batch):
    run, calls = batch
    assert run("download") == 0
    assert sorted(calls["download"]) == ["a", "b"]
    assert sorted(calls["subs"]) == ["a", "b"]
    assert calls["asr"] == []


def test_asr_stage_uses_cache_only(batch):
    run, calls = batch
    assert run("asr") == 1  # "b" has no cached audio
    assert calls["download"] == [] and calls["subs"] == []
    assert calls["asr"] == [("a", False)]
