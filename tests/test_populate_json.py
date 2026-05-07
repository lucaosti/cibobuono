"""
Tests for populate_json: visit creation and flagged-segment routing.
"""

from __future__ import annotations

__author__ = "Luca Ostinelli"

import pytest
from scripts.populate_json import create_visit, create_flagged_segment
from scripts.schemas import FlagReason


_BASE_EXTRACTION = {
    "mention_time": 754,
    "mention_timestamp": "12:34",
    "chunk_end": "13:10",
    "rating": "8.5",
    "sentiment": "positive",
    "rubrica": "",
    "notes": "",
    "confidence": 0.85,
}


class TestCreateVisit:
    def test_mention_time_path(self):
        """When mention_time is present it is used as start_seconds."""
        visit = create_visit(
            extraction=_BASE_EXTRACTION,
            locale_id="locale_abc",
            video_id="vid123",
            channel_id="ch1",
            publish_date="2025-07-20",
        )
        assert visit is not None
        assert visit["visit_id"] == "visit_vid123_754"
        assert visit["timestamp_start"] == "12:34"

    def test_chunk_start_fallback(self):
        """When mention_time is absent, chunk_start/chunk_start_seconds are used."""
        extraction = {
            "chunk_start": "5:30",
            "chunk_start_seconds": 330,
            "chunk_end": "7:00",
            "rating": None,
            "sentiment": "neutral",
            "confidence": 0.7,
        }
        visit = create_visit(extraction, "locale_xyz", "vidABC", "ch2", "2025-01-01")
        assert visit is not None
        assert visit["visit_id"] == "visit_vidABC_330"
        assert visit["timestamp_start"] == "5:30"

    def test_youtube_url_format(self):
        visit = create_visit(_BASE_EXTRACTION, "locale_abc", "vid123", "ch1", "2025-07-20")
        assert visit["youtube_url"] == "https://youtu.be/vid123?t=754"

    def test_locale_id_stored(self):
        visit = create_visit(_BASE_EXTRACTION, "locale_xyz", "vid123", "ch1", "2025-07-20")
        assert visit["locale_id"] == "locale_xyz"

    def test_publish_date_stored(self):
        visit = create_visit(_BASE_EXTRACTION, "locale_abc", "vid123", "ch1", "2025-07-20")
        assert visit["date"] == "2025-07-20"

    def test_null_rating_allowed(self):
        ext = {**_BASE_EXTRACTION, "rating": None}
        visit = create_visit(ext, "locale_abc", "vid123", "ch1", "2025-07-20")
        assert visit is not None
        assert visit["rating"] is None

    def test_validation_failure_returns_none(self):
        """Invalid timestamp should fail Pydantic validation and return None."""
        bad_extraction = {
            "mention_time": 100,
            "mention_timestamp": "not_a_timestamp",
            "chunk_end": "also_bad",
            "rating": "8",
            "sentiment": "positive",
            "confidence": 0.9,
        }
        result = create_visit(bad_extraction, "locale_abc", "vid123", "ch1", "2025-07-20")
        assert result is None

    def test_chunk_start_seconds_computed_from_timestamp(self):
        """chunk_start_seconds falls back to parsing chunk_start if not present."""
        extraction = {
            "chunk_start": "12:34",
            "chunk_end": "13:10",
            "rating": None,
            "sentiment": "neutral",
            "confidence": 0.6,
        }
        visit = create_visit(extraction, "locale_abc", "vid123", "ch1", "2025-01-01")
        assert visit is not None
        assert visit["visit_id"] == "visit_vid123_754"


class TestCreateFlaggedSegment:
    def _extraction(self, **kwargs) -> dict:
        base = {
            "mention_time": 200,
            "mention_timestamp": "3:20",
            "chunk_end": "4:00",
            "locale_name": "Da Remo",
            "city": "Roma",
            "confidence": 0.3,
        }
        base.update(kwargs)
        return base

    def test_low_confidence_flag(self):
        ext = self._extraction()
        seg = create_flagged_segment(ext, "vid1", "ch1")
        assert seg is not None
        assert seg["reason"] == FlagReason.LOW_CONFIDENCE.value

    def test_geocoding_failed_flag(self):
        ext = self._extraction(_flag_reason="geocoding_failed")
        seg = create_flagged_segment(ext, "vid1", "ch1")
        assert seg["reason"] == FlagReason.GEOCODING_FAILED.value

    def test_osm_not_found_flag(self):
        ext = self._extraction(_flag_reason="osm_not_found")
        seg = create_flagged_segment(ext, "vid1", "ch1")
        assert seg["reason"] == FlagReason.OSM_NOT_FOUND.value

    def test_rating_mismatch_flag(self):
        ext = self._extraction(_flag_reason="rating_mismatch_title")
        seg = create_flagged_segment(ext, "vid1", "ch1")
        assert seg["reason"] == FlagReason.RATING_TITLE_MISMATCH.value

    def test_missing_name_flag(self):
        ext = self._extraction(locale_name="")
        seg = create_flagged_segment(ext, "vid1", "ch1")
        assert seg["reason"] == FlagReason.MISSING_NAME.value

    def test_missing_address_flag(self):
        ext = self._extraction(city="", address="")
        seg = create_flagged_segment(ext, "vid1", "ch1")
        assert seg["reason"] == FlagReason.MISSING_ADDRESS.value

    def test_explicit_reason_overrides_auto(self):
        ext = self._extraction(locale_name="")
        seg = create_flagged_segment(ext, "vid1", "ch1", reason=FlagReason.LOW_CONFIDENCE.value)
        assert seg["reason"] == FlagReason.LOW_CONFIDENCE.value

    def test_youtube_url_format(self):
        ext = self._extraction()
        seg = create_flagged_segment(ext, "vid1", "ch1")
        assert seg["youtube_url"] == "https://youtu.be/vid1?t=200"

    def test_reviewed_by_human_defaults_false(self):
        seg = create_flagged_segment(self._extraction(), "vid1", "ch1")
        assert seg["reviewed_by_human"] is False

    def test_extracted_text_truncated_at_500_chars(self):
        long_text = "x" * 600
        ext = self._extraction(text=long_text)
        seg = create_flagged_segment(ext, "vid1", "ch1")
        assert len(seg["extracted_text"]) == 500

    def test_output_has_required_fields(self):
        seg = create_flagged_segment(self._extraction(), "vid1", "ch1")
        required = {
            "video_id", "channel_id", "timestamp_start", "timestamp_end",
            "youtube_url", "reason", "llm_confidence", "reviewed_by_human",
        }
        assert required.issubset(seg.keys())

    def test_chunk_start_fallback_when_no_mention_time(self):
        """Falls back to chunk_start_seconds when mention_time is absent."""
        ext = {
            "chunk_start_seconds": 100,
            "chunk_start": "1:40",
            "chunk_end": "2:10",
            "locale_name": "Test",
            "city": "Roma",
            "confidence": 0.2,
        }
        seg = create_flagged_segment(ext, "vidX", "chX")
        assert seg["youtube_url"] == "https://youtu.be/vidX?t=100"
