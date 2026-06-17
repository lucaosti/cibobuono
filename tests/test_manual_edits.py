"""Tests for manual visit corrections."""

__author__ = "Luca Ostinelli"

import json
from unittest.mock import patch

import pytest

from scripts.manual_edits import remove_visit, add_manual_visit
from scripts.pipeline_executor import _publishable_extraction


def test_remove_visit(tmp_path, monkeypatch):
    visits = [{"visit_id": "visit_v1_100", "locale_id": "locale_abc", "video_id": "v1"}]
    vpath = tmp_path / "visits.json"
    vpath.write_text(json.dumps(visits), encoding="utf-8")
    cpath = tmp_path / "corrections.json"
    cpath.write_text("[]", encoding="utf-8")
    monkeypatch.setattr("scripts.manual_edits.VISITS_JSON", vpath)
    monkeypatch.setattr("scripts.manual_edits.CORRECTIONS_JSON", cpath)
    monkeypatch.setattr("scripts.manual_edits.save_json_split", lambda p, d: vpath.write_text(json.dumps(d)))
    monkeypatch.setattr("scripts.manual_edits.load_json", lambda p: json.loads(p.read_text()))

    ok, msg = remove_visit("visit_v1_100", hide_locale=True, reason="test")
    assert ok is True
    assert "hidden" in msg
    left = json.loads(vpath.read_text())
    assert left == []


def test_publishable_requires_osm():
    ext = {"locale_name": "Foo", "lat": 41.9, "lon": 12.5, "osm_verified": True, "osm_match_score": 90}
    ok, _ = _publishable_extraction(ext, set())
    assert ok is True

    weak = {**ext, "osm_match_score": 70}
    ok, why = _publishable_extraction(weak, set())
    assert ok is False
    assert "osm_weak" in why


def test_add_manual_visit_requires_geocode(tmp_path, monkeypatch):
    vpath = tmp_path / "videos.json"
    vpath.write_text(
        json.dumps([{"video_id": "v1", "channel_id": "ch1", "publish_date": "2026-05-30"}]),
        encoding="utf-8",
    )
    monkeypatch.setattr("scripts.manual_edits.VIDEOS_JSON", vpath)
    monkeypatch.setattr("scripts.manual_edits.load_json", lambda p: json.loads(p.read_text()))

    with patch("scripts.geocode_locales.geocode_locale", return_value=None):
        ok, msg, _ = add_manual_visit(locale_name="Test", video_id="v1", timestamp_start="1:00")
    assert ok is False
    assert "Geocoding" in msg
