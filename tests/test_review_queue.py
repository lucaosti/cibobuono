"""Tests for review queue and reports."""

from __future__ import annotations

import json

import pytest

from scripts.review_queue import (
    LOCALE_REPORTS_JSON,
    pending_reviews,
    submit_report,
    youtube_timestamp_url,
)


def test_youtube_timestamp_url():
    assert youtube_timestamp_url("abc123", 125) == "https://youtu.be/abc123?t=125"


def test_submit_report(tmp_path, monkeypatch):
    reports_path = tmp_path / "locale_reports.json"
    monkeypatch.setattr("scripts.review_queue.LOCALE_REPORTS_JSON", reports_path)
    # Don't mock save_json/load_json — let them operate on the tmp path so the
    # file-write assertion below actually works.
    monkeypatch.setattr("scripts.review_queue.CORRECTIONS_JSON", tmp_path / "corrections.json")

    entry = submit_report(
        locale_name="Fake Place",
        reason="Non esiste",
        video_id="vid1",
        youtube_url="https://youtu.be/vid1?t=60",
    )
    assert entry["locale_name"] == "Fake Place"
    assert entry["status"] == "open"
    assert reports_path.exists()
    assert reports_path.read_text(encoding="utf-8").count("Fake Place") == 1


def test_pending_reviews_filters_reviewed(tmp_path, monkeypatch):
    flagged = tmp_path / "flagged.json"
    flagged.write_text(
        json.dumps(
            [
                {"video_id": "v1", "timestamp_start": "1:00", "reviewed_by_human": False,
                 "locale_name": "A", "reason": "low_confidence", "llm_confidence": 0.4,
                 "youtube_url": "https://youtu.be/v1?t=60", "timestamp_end": "2:00",
                 "channel_id": "c1", "extracted_text": ""},
                {"video_id": "v2", "timestamp_start": "2:00", "reviewed_by_human": True,
                 "locale_name": "B", "reason": "low_confidence", "llm_confidence": 0.5,
                 "youtube_url": "https://youtu.be/v2?t=120", "timestamp_end": "3:00",
                 "channel_id": "c1", "extracted_text": ""},
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("scripts.review_queue.FLAGGED_SEGMENTS_JSON", flagged)
    monkeypatch.setattr("scripts.review_queue.load_json", lambda p: json.loads(p.read_text()))
    monkeypatch.setattr("scripts.review_queue._video_titles", lambda: {"v1": "Title 1"})

    items = pending_reviews()
    assert len(items) == 1
    assert items[0]["locale_name"] == "A"
    assert items[0]["video_title"] == "Title 1"
