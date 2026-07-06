"""Tests for scripts.vote_aggregator — Perceptor signal fusion."""

__author__ = "Luca Ostinelli"

import pytest

from scripts.vote_aggregator import Vote, combine_confidence, perceptor_votes


class TestPerceptorVotes:
    def test_no_perception_record_abstains(self):
        assert perceptor_votes(None, "Roscioli", 12.0) == []

    def test_no_matching_signal_abstains(self):
        record = {"video": {"captions": []}, "audio": {"segment_speakers": []}}
        assert perceptor_votes(record, "Roscioli", 12.0) == []

    def test_ocr_caption_match_votes_visit(self):
        record = {
            "video": {
                "captions": [
                    {"t": 11.0, "caption": "insegna Forno Roscioli visibile sullo sfondo"}
                ]
            },
            "audio": {"segment_speakers": []},
        }
        votes = perceptor_votes(record, "Roscioli", 12.0)
        assert any(v.source == "perceptor_ocr" and v.decision == "visit" for v in votes)

    def test_caption_far_outside_window_ignored(self):
        record = {
            "video": {"captions": [{"t": 500.0, "caption": "Roscioli insegna"}]},
            "audio": {"segment_speakers": []},
        }
        assert perceptor_votes(record, "Roscioli", 12.0) == []

    def test_host_speaker_votes_visit(self):
        record = {
            "video": {"captions": []},
            "audio": {"segment_speakers": [{"start": 10.0, "end": 15.0, "speaker": "S0"}]},
        }
        votes = perceptor_votes(record, "Roscioli", 12.0)
        assert any(v.source == "perceptor_speaker" and v.decision == "visit" for v in votes)

    def test_guest_speaker_votes_mention(self):
        record = {
            "video": {"captions": []},
            "audio": {"segment_speakers": [{"start": 10.0, "end": 15.0, "speaker": "S1"}]},
        }
        votes = perceptor_votes(record, "Roscioli", 12.0)
        assert any(v.source == "perceptor_speaker" and v.decision == "mention" for v in votes)

    def test_unknown_speaker_abstains(self):
        record = {
            "video": {"captions": []},
            "audio": {"segment_speakers": [{"start": 10.0, "end": 15.0, "speaker": "S?"}]},
        }
        assert perceptor_votes(record, "Roscioli", 12.0) == []


class TestCombineConfidence:
    def test_no_votes_returns_base_unchanged(self):
        assert combine_confidence(0.6, []) == 0.6

    def test_agreeing_visit_vote_raises_confidence(self):
        votes = [Vote(source="perceptor_ocr", decision="visit", confidence=0.9, weight=0.35)]
        assert combine_confidence(0.55, votes) > 0.55

    def test_disagreeing_mention_vote_lowers_confidence(self):
        votes = [Vote(source="perceptor_speaker", decision="mention", confidence=0.8, weight=0.30)]
        assert combine_confidence(0.7, votes) < 0.7

    def test_output_stays_in_unit_interval(self):
        votes = [Vote(source="perceptor_ocr", decision="visit", confidence=0.99, weight=0.35)]
        result = combine_confidence(0.99, votes)
        assert 0.0 <= result <= 1.0
