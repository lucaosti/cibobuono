"""Tests for visit_classifier deterministic rules."""

__author__ = "Luca Ostinelli"

from scripts.ner_candidates import Candidate
from scripts.visit_classifier import classify_visit_rules, get_transcript_window


def _cand(name: str) -> Candidate:
    return Candidate(
        name=name,
        label="restaurant",
        start_char=0,
        end_char=len(name),
        start_time=0.0,
        chunk_index=0,
        ner_score=0.8,
    )


def test_rule_rejects_pure_comparison():
    w = "È come da Bonci questo panino, non è mica lo stesso livello."
    d, reason = classify_visit_rules(w, _cand("Bonci"), {})
    assert d == "mention"
    assert "mention" in reason


def test_rule_accepts_visit_pattern_with_food():
    w = "Siamo da Roscioli e assaggiamo la pizza bianca, buonissima."
    d, reason = classify_visit_rules(w, _cand("Roscioli"), {})
    assert d == "visit"
    assert "visit_pattern" in reason


def test_blacklist_city_without_food():
    ctx = {"city": {"roma"}}
    w = "A Roma piove spesso in autunno."
    d, _ = classify_visit_rules(w, _cand("Roma"), ctx)
    assert d == "mention"


def test_get_transcript_window():
    tr = {
        "segments": [
            {"start": 0, "end": 5, "text": "intro"},
            {"start": 10, "end": 20, "text": "siamo da peppe"},
        ]
    }
    w = get_transcript_window(tr, 15.0, pad=10.0)
    assert "peppe" in w.lower()
