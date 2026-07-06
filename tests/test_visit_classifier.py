"""Tests for visit_classifier deterministic rules."""

__author__ = "Luca Ostinelli"

from unittest.mock import MagicMock

from scripts.ner_candidates import Candidate
from scripts.visit_classifier import (
    classify_candidate,
    classify_visit_rules,
    classify_with_llm_ensemble,
    get_transcript_window,
)


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


def _llm_response(visit: bool, evidence: str = "quote") -> dict:
    return {
        "choices": [
            {"message": {"content": f'{{"visit": {str(visit).lower()}, "evidence": "{evidence}"}}'}}
        ]
    }


class TestClassifyWithLlmEnsemble:
    def test_unanimous_visit_majority(self):
        llm = MagicMock()
        llm.create_chat_completion.side_effect = [
            _llm_response(True, "siamo qui"),
            _llm_response(True, "siamo qui"),
            _llm_response(True, "siamo qui"),
        ]
        visit, evidence, conf = classify_with_llm_ensemble(llm, "siamo qui da Roscioli", _cand("Roscioli"))
        assert visit is True
        assert evidence
        assert conf > 0.7

    def test_split_vote_confidence_lower_than_unanimous(self):
        llm = MagicMock()
        llm.create_chat_completion.side_effect = [
            _llm_response(True),
            _llm_response(False),
            _llm_response(True),
        ]
        visit, _, conf_split = classify_with_llm_ensemble(llm, "w", _cand("Roscioli"))
        assert visit is True

        llm2 = MagicMock()
        llm2.create_chat_completion.side_effect = [_llm_response(True)] * 3
        _, _, conf_unanimous = classify_with_llm_ensemble(llm2, "w", _cand("Roscioli"))
        assert conf_split < conf_unanimous

    def test_majority_mention_returns_false(self):
        llm = MagicMock()
        llm.create_chat_completion.side_effect = [
            _llm_response(False),
            _llm_response(False),
            _llm_response(True),
        ]
        visit, _, _ = classify_with_llm_ensemble(llm, "w", _cand("Roscioli"))
        assert visit is False

    def test_none_llm_returns_default(self):
        visit, evidence, conf = classify_with_llm_ensemble(None, "w", _cand("Roscioli"))
        assert visit is False
        assert conf == 0.4

    def test_three_calls_made_per_ensemble_run(self):
        llm = MagicMock()
        llm.create_chat_completion.side_effect = [_llm_response(True)] * 3
        classify_with_llm_ensemble(llm, "w", _cand("Roscioli"))
        assert llm.create_chat_completion.call_count == 3


def _cand_score(name: str, ner_score: float) -> Candidate:
    return Candidate(
        name=name, label="restaurant", start_char=0, end_char=len(name),
        start_time=0.0, chunk_index=0, ner_score=ner_score,
    )


class TestClassifyCandidateGating:
    def test_ambiguous_reason_uses_ensemble(self):
        # No mention/visit pattern, no food/price signal -> "no_clear_signal".
        w = "Che dire di Roscioli."
        llm = MagicMock()
        llm.create_chat_completion.side_effect = [_llm_response(True)] * 3
        decision, reason = classify_visit_rules(w, _cand("Roscioli"), {})
        assert decision == "unsure" and reason == "no_clear_signal"
        classify_candidate(w, _cand("Roscioli"), {}, llm)
        assert llm.create_chat_completion.call_count == 3

    def test_non_ambiguous_unsure_uses_single_call(self):
        # Visit pattern present but weak food evidence + low NER score ->
        # "visit_pattern_weak_food", resolvable with 1 LLM call.
        w = "Siamo da Roscioli oggi."
        cand = _cand_score("Roscioli", 0.3)
        decision, reason = classify_visit_rules(w, cand, {})
        assert decision == "unsure" and reason == "visit_pattern_weak_food"
        llm = MagicMock()
        llm.create_chat_completion.return_value = _llm_response(True)
        classify_candidate(w, cand, {}, llm)
        assert llm.create_chat_completion.call_count == 1
