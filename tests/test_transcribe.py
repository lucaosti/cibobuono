"""
Tests for transcribe_video module: VTT parsing and scrolling-subtitle dedup.
"""

__author__ = "Luca Ostinelli"

import pytest
from pathlib import Path
from scripts.transcribe_video import _parse_vtt


@pytest.fixture
def tmp_vtt(tmp_path):
    """Helper to write a VTT file and return its path."""
    def _write(content: str) -> Path:
        p = tmp_path / "test_subs.it.vtt"
        p.write_text(content, encoding="utf-8")
        return p
    return _write


class TestParseVttScrollingDedup:
    """YouTube auto-subs use a scrolling format where each cue repeats
    lines from the previous cue.  Our parser should deduplicate them."""

    def test_basic_scrolling_dedup(self, tmp_vtt):
        """Each cue shows 2 lines; the first line repeats from the previous cue."""
        vtt = tmp_vtt(
            "WEBVTT\n\n"
            "00:00:00.000 --> 00:00:02.000\n"
            "ciao a tutti\n\n"
            "00:00:02.000 --> 00:00:04.000\n"
            "ciao a tutti\n"
            "oggi andiamo a mangiare\n\n"
            "00:00:04.000 --> 00:00:06.000\n"
            "oggi andiamo a mangiare\n"
            "la pizza migliore\n\n"
        )
        result = _parse_vtt(vtt)
        assert result is not None
        # Each unique line should appear exactly once in the full text
        assert result["text"].count("ciao a tutti") == 1
        assert result["text"].count("oggi andiamo a mangiare") == 1
        assert result["text"].count("la pizza migliore") == 1

    def test_exact_duplicate_cues(self, tmp_vtt):
        """Consecutive cues with identical text → merge into one segment."""
        vtt = tmp_vtt(
            "WEBVTT\n\n"
            "00:00:00.000 --> 00:00:02.000\n"
            "stessa riga\n\n"
            "00:00:02.000 --> 00:00:04.000\n"
            "stessa riga\n\n"
        )
        result = _parse_vtt(vtt)
        assert result is not None
        assert result["text"].count("stessa riga") == 1
        # Only one segment but with extended end time
        assert len(result["segments"]) == 1
        assert result["segments"][0]["end"] == 4.0

    def test_no_overlap_preserved(self, tmp_vtt):
        """When cues don't overlap, all lines are preserved."""
        vtt = tmp_vtt(
            "WEBVTT\n\n"
            "00:00:00.000 --> 00:00:02.000\n"
            "prima frase\n\n"
            "00:00:02.000 --> 00:00:04.000\n"
            "seconda frase\n\n"
            "00:00:04.000 --> 00:00:06.000\n"
            "terza frase\n\n"
        )
        result = _parse_vtt(vtt)
        assert result is not None
        assert result["text"] == "prima frase seconda frase terza frase"
        assert len(result["segments"]) == 3

    def test_formatting_tags_stripped(self, tmp_vtt):
        """VTT tags like <c> and timestamps are removed."""
        vtt = tmp_vtt(
            "WEBVTT\n\n"
            "00:00:00.000 --> 00:00:03.000\n"
            "<c>ciao</c> <00:00:01.500>mondo</c>\n\n"
        )
        result = _parse_vtt(vtt)
        assert result is not None
        assert "<" not in result["text"]
        assert "ciao" in result["text"]
        assert "mondo" in result["text"]

    def test_source_field(self, tmp_vtt):
        """Parsed transcript should be tagged as YouTube manual subs.

        Manual VTTs are the only YouTube-sub path we still trust — the
        auto-generated branch was removed because Italian proper nouns get
        mangled (e.g. "Raimond di Garibaldi" instead of "Raimondi di Garibaldi").
        """
        vtt = tmp_vtt(
            "WEBVTT\n\n"
            "00:00:00.000 --> 00:00:02.000\n"
            "test\n\n"
        )
        result = _parse_vtt(vtt)
        assert result is not None
        assert result["source"] == "youtube_subs_manual"

    def test_empty_vtt_returns_none(self, tmp_vtt):
        """VTT with no cues returns None."""
        vtt = tmp_vtt("WEBVTT\n\n")
        result = _parse_vtt(vtt)
        assert result is None
