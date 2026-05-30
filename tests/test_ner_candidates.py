"""Tests for ner_candidates (heuristic fallback; GLiNER mocked)."""

__author__ = "Luca Ostinelli"

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
    """Char spans returned by heuristic NER must fall within the chunk text bounds,
    and the span-to-timestamp mapping must resolve to the correct segment time."""
    monkeypatch.setattr("scripts.ner_candidates.get_gliner", lambda: None)

    # Two segments joined by a space: "Siamo al Forno Roscioli famoso a Roma."
    # _build_char_ranges must produce ranges that align with this joined text.
    seg1 = "Siamo al Forno Roscioli"
    seg2 = "famoso a Roma."
    text = seg1 + " " + seg2

    chunk = {
        "chunk_index": 1,
        "start_time": 5.0,
        "text": text,
        "segment_timestamps": [
            (5.0, seg1),
            (8.0, seg2),
        ],
    }
    venues, _ = extract_chunk_candidates(chunk)
    assert len(venues) > 0
    for v in venues:
        assert 0 <= v.start_char < len(text), f"start_char {v.start_char} out of [{0}, {len(text)})"
        assert 0 < v.end_char <= len(text), f"end_char {v.end_char} out of (0, {len(text)}]"
        assert v.start_char < v.end_char

    # "Forno Roscioli" sits in the first segment (chars 9–23), must map to t=5.0
    forno = next((v for v in venues if "Roscioli" in v.name), None)
    assert forno is not None, "Expected 'Forno Roscioli' candidate not found"
    assert forno.start_time == 5.0


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


def test_parallel_ner_all_chunks(monkeypatch):
    monkeypatch.setattr("scripts.ner_candidates.get_gliner", lambda: None)

    chunks = [
        {
            "chunk_index": 0,
            "start_time": 0.0,
            "text": "Siamo da Forno Rossi a Roma.",
            "segment_timestamps": [(0.0, "Siamo da Forno Rossi a Roma.")],
        },
        {
            "chunk_index": 1,
            "start_time": 90.0,
            "text": "Poi andiamo da Pizzeria Bianchi.",
            "segment_timestamps": [(90.0, "Poi andiamo da Pizzeria Bianchi.")],
        },
    ]
    from scripts.ner_candidates import extract_all_chunks_candidates

    out = extract_all_chunks_candidates(chunks, max_workers=2)
    assert 0 in out and 1 in out
    names = {v.name for venues, _ in out.values() for v in venues}
    assert any("Rossi" in n or "Forno" in n for n in names)
    assert any("Bianchi" in n for n in names)
