"""Tests for holistic venue discovery."""

__author__ = "Luca Ostinelli"

import json
from unittest.mock import MagicMock

from scripts.venue_discovery import (
    _parse_discovery_json,
    build_timestamped_transcript,
    discover_venues_llm,
)


def test_parse_discovery_json():
    raw = '{"venues": [{"name": "Da Remo", "timestamp": "12:30", "on_site": true}]}'
    rows = _parse_discovery_json(raw)
    assert len(rows) == 1
    assert rows[0]["name"] == "Da Remo"


def test_build_timestamped_transcript_with_chapters():
    transcript = {
        "segments": [
            {"start": 0, "end": 5, "text": "Intro"},
            {"start": 60, "end": 70, "text": "Siamo da Peppe"},
        ]
    }
    chapters = [{"title": "Peppe Mangione", "start_time": 60.0}]
    text = build_timestamped_transcript(transcript, chapters)
    assert "CHAPTER: Peppe Mangione" in text
    assert "[1:00]" in text


def test_discover_venues_llm_parses_response():
    llm = MagicMock()
    payload = {
        "venues": [
            {
                "name": "Trattoria Rossi",
                "timestamp": "2:15",
                "on_site": True,
                "category": ["trattoria"],
                "rating": "8",
                "sentiment": "positive",
                "evidence": "siamo da Trattoria Rossi",
            }
        ]
    }
    llm.create_chat_completion.return_value = {
        "choices": [{"message": {"content": json.dumps(payload)}}]
    }
    transcript = {
        "segments": [{"start": 135, "end": 150, "text": "siamo da Trattoria Rossi"}]
    }
    rows = discover_venues_llm(
        llm,
        transcript=transcript,
        video_title="Roma criminale",
        video_description="",
        video_intel=None,
    )
    assert len(rows) == 1
    assert rows[0]["locale_name"] == "Trattoria Rossi"
    assert rows[0]["mention_time"] == 135.0
    assert rows[0]["_source"] == "discovery_llm"
