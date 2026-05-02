"""Tests for repair_stale_state (pending videos with existing output rows)."""

import json

import pytest

from scripts.schemas import VideoStatus


@pytest.fixture
def data_paths(tmp_path, monkeypatch):
    import scripts.fetch_videos as fv
    import scripts.populate_json as pop
    import scripts.repair_stale_state as rs
    import scripts.utils as u

    monkeypatch.setattr(u, "DATA_DIR", tmp_path)
    monkeypatch.setattr(u, "VIDEOS_JSON", tmp_path / "videos.json")
    monkeypatch.setattr(u, "VISITS_JSON", tmp_path / "visits.json")
    monkeypatch.setattr(u, "FLAGGED_SEGMENTS_JSON", tmp_path / "flagged_segments.json")
    monkeypatch.setattr(u, "PROCESSED_VIDEOS_JSON", tmp_path / "processed_videos.json")
    monkeypatch.setattr(fv, "VIDEOS_JSON", tmp_path / "videos.json")
    monkeypatch.setattr(pop, "PROCESSED_VIDEOS_JSON", tmp_path / "processed_videos.json")
    monkeypatch.setattr(rs, "VIDEOS_JSON", tmp_path / "videos.json")
    monkeypatch.setattr(rs, "VISITS_JSON", tmp_path / "visits.json")
    monkeypatch.setattr(rs, "FLAGGED_SEGMENTS_JSON", tmp_path / "flagged_segments.json")
    monkeypatch.setattr(rs, "PROCESSED_VIDEOS_JSON", tmp_path / "processed_videos.json")

    return tmp_path


def test_repair_marks_pending_with_visits(data_paths, monkeypatch):
    from scripts import repair_stale_state as rs

    videos = [
        {
            "video_id": "abc123",
            "channel_id": "UC_test",
            "title": "Test",
            "status": VideoStatus.PENDING.value,
            "publish_date": "2024-01-01",
            "url": "https://youtu.be/abc123",
        }
    ]
    visits = [
        {
            "visit_id": "visit_abc123_42",
            "locale_id": "loc1",
            "video_id": "abc123",
            "channel_id": "UC_test",
            "timestamp_start": "0:42",
            "timestamp_end": "1:00",
            "youtube_url": "https://youtu.be/abc123?t=42",
            "rating": "8",
            "sentiment": "positive",
            "rubrica": "",
            "notes": "",
            "llm_confidence": 0.9,
            "extraction_date": "2024-06-01",
            "date": "2024-01-01",
        }
    ]
    (data_paths / "videos.json").write_text(json.dumps(videos), encoding="utf-8")
    (data_paths / "visits.json").write_text(json.dumps(visits), encoding="utf-8")
    (data_paths / "flagged_segments.json").write_text("[]", encoding="utf-8")
    (data_paths / "processed_videos.json").write_text("[]", encoding="utf-8")

    summary = rs.repair_stale_video_state(dry_run=False)
    assert summary["count"] == 1
    assert summary["repaired_video_ids"] == ["abc123"]

    fixed = json.loads((data_paths / "videos.json").read_text(encoding="utf-8"))
    assert fixed[0]["status"] == VideoStatus.PROCESSED.value

    proc = json.loads((data_paths / "processed_videos.json").read_text(encoding="utf-8"))
    assert len(proc) == 1
    assert proc[0]["video_id"] == "abc123"
    assert proc[0]["visits_extracted"] == 1


def test_repair_dry_run_no_write(data_paths):
    from scripts import repair_stale_state as rs

    videos = [
        {
            "video_id": "abc123",
            "channel_id": "UC_test",
            "title": "Test",
            "status": VideoStatus.PENDING.value,
            "publish_date": "2024-01-01",
            "url": "https://youtu.be/abc123",
        }
    ]
    visits = [
        {
            "visit_id": "visit_abc123_42",
            "locale_id": "loc1",
            "video_id": "abc123",
            "channel_id": "UC_test",
            "timestamp_start": "0:42",
            "timestamp_end": "1:00",
            "youtube_url": "https://youtu.be/abc123?t=42",
            "rating": "8",
            "sentiment": "positive",
            "rubrica": "",
            "notes": "",
            "llm_confidence": 0.9,
            "extraction_date": "2024-06-01",
            "date": "2024-01-01",
        }
    ]
    (data_paths / "videos.json").write_text(json.dumps(videos), encoding="utf-8")
    (data_paths / "visits.json").write_text(json.dumps(visits), encoding="utf-8")
    (data_paths / "flagged_segments.json").write_text("[]", encoding="utf-8")
    (data_paths / "processed_videos.json").write_text("[]", encoding="utf-8")

    rs.repair_stale_video_state(dry_run=True)

    fixed = json.loads((data_paths / "videos.json").read_text(encoding="utf-8"))
    assert fixed[0]["status"] == VideoStatus.PENDING.value
