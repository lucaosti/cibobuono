"""Tests for pipeline dashboard snapshot serialization."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.dashboard import (
    DASHBOARD_SNAPSHOT_PATH,
    Dashboard,
    LocaleHit,
    VideoSourcesInfo,
)


@pytest.fixture
def dash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Dashboard:
    monkeypatch.setattr("scripts.dashboard.LOGS_DIR", tmp_path)
    monkeypatch.setattr("scripts.dashboard.DASHBOARD_SNAPSHOT_PATH", tmp_path / "live.json")
    monkeypatch.setattr("scripts.dashboard.load_json", lambda _p: [])
    return Dashboard(live=False)


def test_persist_snapshot_writes_json(dash: Dashboard, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    snap_path = tmp_path / "live.json"
    monkeypatch.setattr("scripts.dashboard.DASHBOARD_SNAPSHOT_PATH", snap_path)

    dash.set_phase("Phase 2 — Processing")
    dash.set_totals(pending=5, processed=10)
    dash.persist_snapshot()

    assert snap_path.exists()
    data = json.loads(snap_path.read_text(encoding="utf-8"))
    assert data["phase"] == "Phase 2 — Processing"
    assert data["stats"]["pending"] == 5
    assert data["stats"]["processed"] == 10
    assert "updated_at" in data


def test_set_video_sources_and_extractions(dash: Dashboard):
    dash.update_video(1, "Pizza a Napoli", video_id="abc123")
    dash.set_video_sources(
        description_chars=1200,
        chapters_count=3,
        transcript_source="faster_whisper",
        transcript_chars=45000,
        uses_ner=True,
        uses_llm=True,
    )
    dash.set_extractions(
        [{"locale_name": "Da Michele", "city": "Napoli", "confidence": 0.91, "rating": "9/10"}],
        [{"locale_name": "Maybe Place", "city": "Napoli", "confidence": 0.42, "_flag_reason": "low_confidence"}],
    )

    snap = dash.to_snapshot_dict()
    src = snap["current_video"]["sources"]
    assert src["description_chars"] == 1200
    assert src["transcript_source"] == "faster_whisper"
    assert len(snap["current_video"]["extractions"]) == 2
    assert snap["stats"]["run_locales_count"] == 2

    names = {h["name"] for h in snap["run_locales"]}
    assert "Da Michele" in names
    assert "Maybe Place" in names


def test_complete_video_timing_and_recent(dash: Dashboard):
    dash.update_video(2, "Second video", video_id="vid2")
    dash.state.video_start_time -= 125.0
    dash.complete_video(outcome="processed", visits=2, flagged=1)

    snap = dash.to_snapshot_dict()
    assert snap["timing"]["avg_video_s"] == 125.0
    assert len(snap["recent_videos"]) == 1
    assert snap["recent_videos"][0]["outcome"] == "processed"
    assert snap["recent_videos"][0]["visits"] == 2


def test_snapshot_has_control_key_when_merged():
    snap = Dashboard(live=False).to_snapshot_dict()
    assert "hardware" not in snap


def test_load_snapshot_missing_returns_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    missing = tmp_path / "missing.json"
    monkeypatch.setattr("scripts.dashboard.DASHBOARD_SNAPSHOT_PATH", missing)
    assert Dashboard.load_snapshot() is None


def test_set_perception_flows_into_snapshot(dash: Dashboard):
    dash.update_video(1, "Perception video", video_id="vidp")
    dash.set_perception(
        perc_status="ok",
        speakers_count=2,
        voice_matches=["S0→voice_UCx_001 (0.83)"],
        frames_sampled=410,
        novelty_frames=57,
        captions_count=57,
        sample_captions=["Sign reading 'Trattoria da Mario'"],
    )
    src = dash.to_snapshot_dict()["current_video"]["sources"]
    assert src["perc_status"] == "ok"
    assert src["speakers_count"] == 2
    assert src["novelty_frames"] == 57
    assert src["voice_matches"] == ["S0→voice_UCx_001 (0.83)"]
    assert src["sample_captions"][0].startswith("Sign reading")


def test_reset_to_idle_clears_run_state(dash: Dashboard):
    dash.update_video(3, "Old video", video_id="old123")
    dash.set_extractions([{"locale_name": "X", "city": "Roma", "confidence": 0.9}])
    dash.reset_to_idle()
    assert dash.state.phase == "Idle"
    assert dash.state.run_locales == []
    assert dash.state.current_video_id == ""


def test_dashboard_hardware_returns_percentages():
    from scripts.resource_monitor import dashboard_hardware

    hw = dashboard_hardware()
    assert "cpu_percent" in hw
    assert "gpu_percent" in hw


def test_locale_hit_dataclass_fields():
    hit = LocaleHit(name="Test", confidence=0.8, flagged=True, flag_reason="x")
    assert hit.name == "Test"
    assert hit.flagged is True

    info = VideoSourcesInfo(video_id="v1", uses_title=True)
    assert info.video_id == "v1"
