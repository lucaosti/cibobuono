"""
Tests for video_intelligence: title analysis and chapter-based venue hints.
"""

from __future__ import annotations

__author__ = "Luca Ostinelli"

import pytest
from scripts.video_intelligence import (
    VideoIntel,
    analyze_comments,
    analyze_title,
    analyze_chapters,
    parse_description_timestamps,
)


class TestAnalyzeTitleHitDiFranchino:
    def test_single_venue_extracted(self):
        intel = analyze_title("Hit di Franchino: Da Remo Roma")
        assert intel.video_type == "single_venue"
        assert intel.series_name == "Hit di Franchino"
        assert intel.city == "Roma"
        assert len(intel.venue_hints) == 1
        assert intel.venue_hints[0]["name"] == "Da Remo"
        assert intel.venue_hints[0]["confidence"] == "high"

    def test_case_insensitive(self):
        intel = analyze_title("HIT DI FRANCHINO: Pizzeria Napoli Napoli")
        assert intel.video_type == "single_venue"
        assert intel.city == "Napoli"

    def test_non_roman_city(self):
        intel = analyze_title("Hit di Franchino: Al Portico Viterbo")
        assert intel.video_type == "single_venue"
        assert intel.city == "Viterbo"


class TestAnalyzeTitleCityCriminale:
    def test_roma_criminale(self):
        intel = analyze_title("Roma criminale")
        assert intel.video_type == "multi_venue_tour"
        assert intel.city == "Roma"

    def test_napoli_criminale(self):
        intel = analyze_title("Napoli criminale")
        assert intel.video_type == "multi_venue_tour"
        assert intel.city == "Napoli"

    def test_no_venue_hints(self):
        intel = analyze_title("Milano criminale")
        assert intel.venue_hints == []


class TestAnalyzeTitleForni:
    def test_bakery_tour(self):
        intel = analyze_title("Forni criminali Roma")
        assert intel.video_type == "multi_venue_tour"
        assert intel.series_name == "Forni criminali"
        assert intel.city == "Roma"

    def test_bakery_tour_nuova_hit_format(self):
        """Franchino titles like 'Forni criminali nuova HIT a CENTOCELLE'."""
        intel = analyze_title("Forni criminali nuova HIT a CENTOCELLE")
        assert intel.video_type == "multi_venue_tour"
        assert intel.series_name == "Forni criminali"
        assert intel.city == "CENTOCELLE"

    def test_pizza_taglio_criminale_extracts_venue_hint(self):
        """'Pizza a taglio criminale VENUE' → venue hint from title."""
        intel = analyze_title("Pizza a taglio criminale COLLE DEL SOLE")
        assert intel.video_type == "single_venue"
        assert intel.series_name == "Pizza a taglio criminale"
        assert any("COLLE DEL SOLE" in h["name"] for h in intel.venue_hints)

    def test_pizza_taglio_criminale_regina_margherita(self):
        intel = analyze_title("Pizza a taglio criminale REGINA MARGHERITA")
        assert intel.video_type == "single_venue"
        assert any("REGINA MARGHERITA" in h["name"] for h in intel.venue_hints)


class TestAnalyzeTitleCosaMangia:
    def test_eating_with_format(self):
        intel = analyze_title("Cosa mangia Franchino")
        assert intel.video_type == "multi_venue_tour"

    def test_eating_with_longer_name(self):
        intel = analyze_title("Cosa mangia il campione del mondo")
        assert intel.video_type == "multi_venue_tour"


class TestAnalyzeTitleNonReview:
    def test_cena_a_n_mani(self):
        intel = analyze_title("Cena a 4 mani da Remo")
        assert intel.video_type == "non_review"
        assert intel.skip_reason != ""

    def test_salviamo(self):
        intel = analyze_title("Salviamo questo ristorante")
        assert intel.video_type == "non_review"

    def test_sfida(self):
        intel = analyze_title("Una sfida epica di pizza")
        assert intel.video_type == "non_review"

    def test_franchino_contro(self):
        intel = analyze_title("Franchino contro il campione")
        assert intel.video_type == "non_review"

    def test_sono_stato_umiliato(self):
        intel = analyze_title("Sono stato umiliato da questo chef")
        assert intel.video_type == "non_review"

    def test_non_review_suppresses_venue_hints(self):
        """Non-review check returns early before Hit di Franchino check."""
        intel = analyze_title("Sfida Hit di Franchino: Da Remo Roma")
        assert intel.video_type == "non_review"
        assert intel.venue_hints == []


class TestAnalyzeTitleRating:
    def test_da_dieci(self):
        intel = analyze_title("Questo posto è da DIECI")
        assert intel.title_rating == "10"

    def test_da_otto(self):
        intel = analyze_title("La pizza da otto qui")
        assert intel.title_rating == "8"

    def test_numeric_rating(self):
        intel = analyze_title("Il miglior forno da 9 a Roma")
        assert intel.title_rating == "9"

    def test_numeric_with_plus(self):
        # The \b anchor in the pattern prevents capturing the trailing '+',
        # so "da 10+" correctly extracts "10".
        intel = analyze_title("Remo da 10+")
        assert intel.title_rating == "10"

    def test_no_rating(self):
        intel = analyze_title("Roma criminale")
        assert intel.title_rating is None

    def test_rating_set_even_on_non_review(self):
        """Title rating is extracted before the non-review check."""
        intel = analyze_title("Sfida da cinque stelle")
        assert intel.video_type == "non_review"
        assert intel.title_rating == "5"

    def test_unknown_title(self):
        intel = analyze_title("Un video generico qualsiasi")
        assert intel.video_type == "unknown"
        assert intel.city == ""
        assert intel.venue_hints == []
        assert intel.title_rating is None


class TestAnalyzeChapters:
    def _base_intel(self, video_type: str = "unknown") -> VideoIntel:
        return VideoIntel(video_type=video_type)

    def test_venue_added_with_start_time(self):
        chapters = [{"title": "Da Remo", "start_time": 120}]
        intel = analyze_chapters(chapters, self._base_intel())
        assert len(intel.venue_hints) == 1
        assert intel.venue_hints[0]["name"] == "Da Remo"
        assert intel.venue_hints[0]["start_time"] == 120.0
        assert intel.venue_hints[0]["confidence"] == "very_high"

    def test_confidence_is_very_high(self):
        chapters = [{"title": "Pizzeria Napoli", "start_time": 60}]
        intel = analyze_chapters(chapters, self._base_intel())
        assert intel.venue_hints[0]["confidence"] == "very_high"

    def test_nav_words_filtered(self):
        chapters = [
            {"title": "Intro", "start_time": 0},
            {"title": "Outro", "start_time": 600},
            {"title": "Da Remo", "start_time": 120},
        ]
        intel = analyze_chapters(chapters, self._base_intel())
        names = [h["name"] for h in intel.venue_hints]
        assert "Intro" not in names
        assert "Outro" not in names
        assert "Da Remo" in names

    def test_pure_number_rejected(self):
        chapters = [{"title": "1", "start_time": 0}]
        intel = analyze_chapters(chapters, self._base_intel())
        assert intel.venue_hints == []

    def test_timestamp_pattern_rejected(self):
        chapters = [{"title": "00:30", "start_time": 30}]
        intel = analyze_chapters(chapters, self._base_intel())
        assert intel.venue_hints == []

    def test_sets_video_type_to_multi_venue_tour_when_unknown(self):
        chapters = [{"title": "Forno Roscioli", "start_time": 60}]
        intel = analyze_chapters(chapters, self._base_intel("unknown"))
        assert intel.video_type == "multi_venue_tour"

    def test_preserves_existing_video_type(self):
        chapters = [{"title": "Da Remo", "start_time": 60}]
        intel = analyze_chapters(chapters, self._base_intel("single_venue"))
        assert intel.video_type == "single_venue"

    def test_duplicate_chapters_not_added_twice(self):
        chapters = [
            {"title": "Da Remo", "start_time": 60},
            {"title": "da remo", "start_time": 120},  # same name, different case
        ]
        intel = analyze_chapters(chapters, self._base_intel())
        assert len(intel.venue_hints) == 1

    def test_chapter_already_in_hints_skipped(self):
        """Chapters that duplicate existing title hints are not re-added."""
        existing_hints = [{"name": "Da Remo", "source": "title", "confidence": "high"}]
        intel = VideoIntel(venue_hints=existing_hints)
        chapters = [{"title": "Da Remo", "start_time": 60}]
        intel = analyze_chapters(chapters, intel)
        assert len(intel.venue_hints) == 1

    def test_empty_chapters_list(self):
        intel = analyze_chapters([], self._base_intel())
        assert intel.venue_hints == []
        assert intel.video_type == "unknown"

    def test_multiple_chapters(self):
        chapters = [
            {"title": "Da Remo", "start_time": 60},
            {"title": "Forno Roscioli", "start_time": 300},
            {"title": "Pizzarium", "start_time": 600},
        ]
        intel = analyze_chapters(chapters, self._base_intel())
        assert len(intel.venue_hints) == 3
        names = [h["name"] for h in intel.venue_hints]
        assert "Da Remo" in names
        assert "Forno Roscioli" in names
        assert "Pizzarium" in names

    def test_start_time_none_when_missing(self):
        chapters = [{"title": "Da Remo"}]  # no start_time key
        intel = analyze_chapters(chapters, self._base_intel())
        assert len(intel.venue_hints) == 1
        assert "start_time" not in intel.venue_hints[0]


class TestParseDescriptionTimestamps:
    def test_basic_timestamp_line(self):
        desc = "0:30 Forno Roscioli\n1:45 Da Remo"
        result = parse_description_timestamps(desc)
        assert len(result) == 2
        assert result[0] == {"timestamp": "0:30", "label": "Forno Roscioli"}
        assert result[1] == {"timestamp": "1:45", "label": "Da Remo"}

    def test_hh_mm_ss_format(self):
        desc = "1:02:30 Da Remo Roma"
        result = parse_description_timestamps(desc)
        assert len(result) == 1
        assert result[0]["timestamp"] == "1:02:30"

    def test_non_timestamp_lines_skipped(self):
        desc = "Questo video è su Roma\n0:30 Da Remo\nSubscribe!"
        result = parse_description_timestamps(desc)
        assert len(result) == 1

    def test_empty_description(self):
        assert parse_description_timestamps("") == []

    def test_label_truncated_at_200_chars(self):
        desc = "0:30 " + "x" * 250
        result = parse_description_timestamps(desc)
        assert len(result[0]["label"]) == 200


class TestAnalyzeComments:
    def _base_intel(self) -> VideoIntel:
        return VideoIntel(video_type="multi_venue_tour")

    def _comment(self, text: str, likes: int = 0) -> dict:
        return {"text": text, "like_count": likes}

    def test_empty_comments_returns_unchanged(self):
        intel = self._base_intel()
        result = analyze_comments([], intel)
        assert result.venue_hints == []

    def test_single_mention_below_threshold_ignored(self):
        # Only 1 mention — threshold requires >= 2
        comments = [self._comment("vi consiglio la pizzeria Da Remo")]
        intel = analyze_comments(comments, self._base_intel())
        assert intel.venue_hints == []

    def test_two_mentions_added(self):
        # Both comments use 'andate da' pattern so they produce identical match keys
        comments = [
            self._comment("andate da Pizzarium!"),
            self._comment("andate da Pizzarium!"),
        ]
        intel = analyze_comments(comments, self._base_intel())
        names = [h["name"].lower() for h in intel.venue_hints]
        assert any("pizzarium" in n for n in names)

    def test_high_like_comment_counts_double(self):
        # One comment with ≥5 likes → counts as 2; no second comment needed
        # Use trailing punctuation so match stops cleanly at the name
        comments = [self._comment("andate da Pizzarium!", likes=10)]
        intel = analyze_comments(comments, self._base_intel())
        names = [h["name"].lower() for h in intel.venue_hints]
        assert any("pizzarium" in n for n in names)

    def test_original_casing_preserved(self):
        # First occurrence casing should be preserved, not .title()
        comments = [
            self._comment("andate da Sora Lella!"),
            self._comment("consiglio da Sora Lella!"),
        ]
        intel = analyze_comments(comments, self._base_intel())
        names = [h["name"] for h in intel.venue_hints]
        assert any("Sora Lella" in n for n in names)

    def test_does_not_duplicate_existing_hint(self):
        intel = self._base_intel()
        intel.venue_hints = [{"name": "Da Remo", "source": "chapter", "confidence": "high"}]
        comments = [
            self._comment("pizzeria Da Remo è top"),
            self._comment("sì pizzeria Da Remo ottima"),
        ]
        result = analyze_comments(comments, intel)
        da_remo_count = sum(1 for h in result.venue_hints if "remo" in h["name"].lower())
        assert da_remo_count == 1  # not duplicated

    def test_short_comments_ignored(self):
        comments = [self._comment("ok"), self._comment("ok")]
        intel = analyze_comments(comments, self._base_intel())
        assert intel.venue_hints == []

    def test_confidence_high_for_many_mentions(self):
        comments = [self._comment(f"ristorante Sant'Isidoro visita {i}") for i in range(5)]
        intel = analyze_comments(comments, self._base_intel())
        if intel.venue_hints:
            confidences = [h["confidence"] for h in intel.venue_hints if "isidoro" in h["name"].lower()]
            assert any(c == "high" for c in confidences)
