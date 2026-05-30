"""Integration tests for extract_pipeline with mocks."""

__author__ = "Luca Ostinelli"

from unittest.mock import MagicMock, patch

from scripts.extract_pipeline import extract_from_video


def test_pipeline_single_chunk_protected_keeps_visit():
    chunks = [
        {
            "chunk_index": 0,
            "start_time": 0.0,
            "start_timestamp": "0:00",
            "end_timestamp": "0:45",
            "text": "Siamo da Peppe Mangione e assaggiamo il maritozzo.",
            "segment_timestamps": [
                (0.0, "Siamo da Peppe Mangione e assaggiamo il maritozzo."),
            ],
        }
    ]
    transcript = {
        "segments": [
            {"start": 0, "end": 10, "text": "Siamo da Peppe Mangione e assaggiamo il maritozzo."},
        ]
    }

    class FakeIntel:
        venue_hints = [{"name": "Peppe Mangione", "source": "title", "confidence": "very_high"}]
        city = "Roma"
        series_name = ""
        video_type = "single_venue"
        title_rating = None
        skip_reason = ""

    fake_cand = __import__(
        "scripts.ner_candidates", fromlist=["Candidate"]
    ).Candidate(
        name="Peppe Mangione",
        label="restaurant",
        start_char=0,
        end_char=12,
        start_time=0.0,
        chunk_index=0,
        ner_score=0.88,
    )

    mock_llm = MagicMock()
    detail_json = (
        '{"rating": "8", "sentiment": "positive", '
        '"notes": "maritozzo", "category": ["pasticceria"], "city": "Roma", "address": ""}'
    )
    mock_llm.create_chat_completion.return_value = {"choices": [{"message": {"content": detail_json}}]}

    with patch("scripts.extract_pipeline.extract_chunk_candidates", return_value=([fake_cand], [])):
        with patch("scripts.extract_pipeline.get_llm", return_value=mock_llm):
            with patch("scripts.extract_pipeline.classify_candidate", return_value=(True, "[rule:test]", 0.85, "rule")):
                ext, flagged = extract_from_video(
                    "vid1",
                    chunks,
                    video_title="Hit: Peppe",
                    video_intel=FakeIntel(),
                    transcript=transcript,
                )
    assert len(ext) >= 1
    assert any(e["locale_name"] == "Peppe Mangione" for e in ext)


def test_pipeline_drops_belgium_name_drop_without_signals():
    chunks = [
        {
            "chunk_index": 0,
            "start_time": 5.0,
            "start_timestamp": "0:05",
            "end_timestamp": "1:00",
            "text": "Parliamo del Belgio e della cucina europea.",
            "segment_timestamps": [(5.0, "Parliamo del Belgio e della cucina europea.")],
        }
    ]
    transcript = {"segments": [{"start": 5, "end": 15, "text": chunks[0]["text"]}]}

    fake_cand = __import__(
        "scripts.ner_candidates", fromlist=["Candidate"]
    ).Candidate(
        name="Belgio",
        label="restaurant",
        start_char=0,
        end_char=6,
        start_time=5.0,
        chunk_index=0,
        ner_score=0.4,
    )

    with patch("scripts.extract_pipeline.extract_chunk_candidates", return_value=([fake_cand], [])):
        with patch("scripts.extract_pipeline.get_llm", return_value=None):
            with patch(
                "scripts.extract_pipeline.classify_candidate",
                return_value=(False, "[rule:test]", 0.7, "rule"),
            ):
                ext, _ = extract_from_video("v2", chunks, transcript=transcript)
    assert ext == []


def test_pipeline_single_chunk_llm_keeps_visit():
    """LLM-confirmed visits in one chunk should not be flagged."""
    chunks_text = "Siamo da Trattoria Rossi e proviamo la carbonara."
    chunks = [
        {
            "chunk_index": 0,
            "start_time": 10.0,
            "start_timestamp": "0:10",
            "end_timestamp": "1:40",
            "text": chunks_text,
            "segment_timestamps": [(10.0, chunks_text)],
        }
    ]
    transcript = {"segments": [{"start": 10, "end": 30, "text": chunks_text}]}

    fake_cand = __import__(
        "scripts.ner_candidates", fromlist=["Candidate"]
    ).Candidate(
        name="Trattoria Rossi",
        label="ristorante",
        start_char=0,
        end_char=15,
        start_time=10.0,
        chunk_index=0,
        ner_score=0.55,
    )

    mock_llm = MagicMock()
    mock_llm.create_chat_completion.return_value = {
        "choices": [{"message": {"content": '{"rating": "7", "sentiment": "positive", "notes": "carbonara", "category": ["trattoria"], "city": "Roma", "address": ""}'}}]
    }

    with patch("scripts.extract_pipeline.extract_chunk_candidates", return_value=([fake_cand], [])):
        with patch("scripts.extract_pipeline.get_llm", return_value=mock_llm):
            with patch("scripts.extract_pipeline.verify_venue_name", return_value=True):
                with patch(
                    "scripts.extract_pipeline.classify_candidate",
                    return_value=(True, "evidence", 0.85, "llm"),
                ):
                    ext, flagged = extract_from_video(
                        "v3",
                        chunks,
                        video_title="Roma criminale",
                        transcript=transcript,
                    )
    assert any(e["locale_name"] == "Trattoria Rossi" for e in ext)
    assert not any(f.get("locale_name") == "Trattoria Rossi" for f in flagged)


def test_verify_venue_name_rejects_dish():
    """A dish name must be rejected even with a strong visit signal."""
    from scripts.visit_classifier import verify_venue_name, _looks_like_venue_name

    assert _looks_like_venue_name("Trattoria Rossi") is True
    assert _looks_like_venue_name("carbonara") is False
    assert _looks_like_venue_name("Carbonara") is False
    # No LLM → structural gate only
    assert verify_venue_name(None, "Da Michele", "") is True
    assert verify_venue_name(None, "pizza", "") is False
