"""
Tests for JSON data integrity.

Validates that all JSON files in data/ conform to schemas
and that cross-references are consistent.
"""

import pytest
from scripts.utils import DATA_DIR, load_json
from scripts.schemas import (
    validate_channels,
    validate_videos,
    validate_locales,
    validate_visits,
    validate_flagged_segments,
    validate_processed_videos,
)


@pytest.fixture
def data_dir():
    return DATA_DIR


class TestJsonFilesValid:
    """Validate that existing JSON files conform to their schemas."""

    def test_channels_schema(self, data_dir):
        data = load_json(data_dir / "channels.json")
        if data:
            result = validate_channels(data)
            assert len(result) == len(data)

    def test_videos_schema(self, data_dir):
        data = load_json(data_dir / "videos.json")
        if data:
            result = validate_videos(data)
            assert len(result) == len(data)

    def test_locales_schema(self, data_dir):
        data = load_json(data_dir / "locales.json")
        if data:
            result = validate_locales(data)
            assert len(result) == len(data)

    def test_visits_schema(self, data_dir):
        data = load_json(data_dir / "visits.json")
        if data:
            result = validate_visits(data)
            assert len(result) == len(data)

    def test_flagged_segments_schema(self, data_dir):
        data = load_json(data_dir / "flagged_segments.json")
        if data:
            result = validate_flagged_segments(data)
            assert len(result) == len(data)

    def test_processed_videos_schema(self, data_dir):
        data = load_json(data_dir / "processed_videos.json")
        if data:
            result = validate_processed_videos(data)
            assert len(result) == len(data)


class TestCrossReferences:
    """Validate referential integrity across JSON files."""

    def test_video_channel_refs(self, data_dir):
        """Every video must reference a valid channel_id."""
        videos = load_json(data_dir / "videos.json")
        channels = load_json(data_dir / "channels.json")
        channel_ids = {ch["channel_id"] for ch in channels}
        for video in videos:
            assert video["channel_id"] in channel_ids, \
                f"Video {video['video_id']} references unknown channel {video['channel_id']}"

    def test_visit_video_refs(self, data_dir):
        """Every visit must reference a valid video_id."""
        visits = load_json(data_dir / "visits.json")
        videos = load_json(data_dir / "videos.json")
        video_ids = {v["video_id"] for v in videos}
        for visit in visits:
            assert visit["video_id"] in video_ids, \
                f"Visit {visit['visit_id']} references unknown video {visit['video_id']}"

    def test_visit_locale_refs(self, data_dir):
        """Every visit must reference a valid locale_id."""
        visits = load_json(data_dir / "visits.json")
        locales = load_json(data_dir / "locales.json")
        locale_ids = {loc["locale_id"] for loc in locales}
        for visit in visits:
            assert visit["locale_id"] in locale_ids, \
                f"Visit {visit['visit_id']} references unknown locale {visit['locale_id']}"

    def test_visit_channel_refs(self, data_dir):
        """Every visit must reference a valid channel_id."""
        visits = load_json(data_dir / "visits.json")
        channels = load_json(data_dir / "channels.json")
        channel_ids = {ch["channel_id"] for ch in channels}
        for visit in visits:
            assert visit["channel_id"] in channel_ids, \
                f"Visit {visit['visit_id']} references unknown channel {visit['channel_id']}"


class TestIdUniqueness:
    """Validate that all IDs are unique within their respective files."""

    def test_unique_channel_ids(self, data_dir):
        data = load_json(data_dir / "channels.json")
        ids = [ch["channel_id"] for ch in data]
        assert len(ids) == len(set(ids)), "Duplicate channel_id found"

    def test_unique_video_ids(self, data_dir):
        data = load_json(data_dir / "videos.json")
        ids = [v["video_id"] for v in data]
        assert len(ids) == len(set(ids)), "Duplicate video_id found"

    def test_unique_locale_ids(self, data_dir):
        data = load_json(data_dir / "locales.json")
        ids = [loc["locale_id"] for loc in data]
        assert len(ids) == len(set(ids)), "Duplicate locale_id found"

    def test_unique_visit_ids(self, data_dir):
        data = load_json(data_dir / "visits.json")
        ids = [v["visit_id"] for v in data]
        assert len(ids) == len(set(ids)), "Duplicate visit_id found"
