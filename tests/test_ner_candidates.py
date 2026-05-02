"""Tests for ner_candidates (heuristic fallback; GLiNER mocked)."""

from scripts.ner_candidates import Candidate, extract_chunk_candidates


def test_heuristic_extracts_capitalized_name(monkeypatch):
    monkeypatch.setattr("scripts.ner_candidates.get_gliner", lambda: None)

    chunk = {
        "chunk_index": 0,
        "start_time": 10.0,
        "text": "Poi parliamo di Forno Roscioli che è famoso a Roma.",
        "segment_timestamps": [
            (10.0, "Poi parliamo di Forno Roscioli che è famoso a Roma."),
        ],
    }
    venues, ctx = extract_chunk_candidates(chunk)
    names = {v.name for v in venues}
    assert "Forno Roscioli" in names
    assert ctx == []


def test_char_ranges_align_with_joined_text(monkeypatch):
    monkeypatch.setattr("scripts.ner_candidates.get_gliner", lambda: None)

    chunk = {
        "chunk_index": 1,
        "start_time": 0.0,
        "text": "uno due",
        "segment_timestamps": [
            (0.0, "uno"),
            (1.0, "due"),
        ],
    }
    venues, _ = extract_chunk_candidates(chunk)
    assert isinstance(venues, list)


def test_candidate_dataclass_fields():
    c = Candidate(
        name="X",
        label="restaurant",
        start_char=0,
        end_char=2,
        start_time=1.5,
        chunk_index=3,
        ner_score=0.9,
    )
    assert c.chunk_index == 3
