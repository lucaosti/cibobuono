"""
Tests for extraction-related functions: non-food detection, locale name validation,
cross-chunk consensus, confidence threshold, food-relevance LLM gate, and
description-based venue hint extraction.
"""

__author__ = "Luca Ostinelli"

from unittest.mock import patch, MagicMock

from scripts.fetch_videos import detect_non_food_video, detect_recipe_video
from scripts.extract_locales import (
    _is_valid_locale_name,
    _clean_locale_name,
    check_food_video,
    extract_hints_from_description,
    is_food_review_video,
    rating_numeric_core,
)
from scripts.video_intelligence import parse_description_timestamps


class TestDetectNonFoodVideo:
    """Test that non-food videos are correctly identified and skipped."""

    def test_boxing_video(self):
        is_nf, reason = detect_non_food_video("COME SALTARE LA CORDA - BOXING TUTORIAL")
        assert is_nf is True
        assert "boxing" in reason.lower() or "saltare la corda" in reason.lower()

    def test_fitness_video(self):
        is_nf, _ = detect_non_food_video("Il mio allenamento mattutino - PALESTRA")
        assert is_nf is True

    def test_gaming_video(self):
        is_nf, _ = detect_non_food_video("GTA 6 gameplay incredibile")
        assert is_nf is True

    def test_workout_video(self):
        is_nf, _ = detect_non_food_video("Workout da casa senza attrezzi")
        assert is_nf is True

    def test_food_review_passes(self):
        is_nf, _ = detect_non_food_video("LA MIGLIOR PIZZA DI ROMA - Assaggi")
        assert is_nf is False

    def test_food_vlog_passes(self):
        is_nf, _ = detect_non_food_video("Mangiamo STREET FOOD a Napoli")
        assert is_nf is False

    def test_ambiguous_title_passes(self):
        """Ambiguous titles should pass (better to process than miss)."""
        is_nf, _ = detect_non_food_video("Una giornata a Roma")
        assert is_nf is False

    def test_prank_video(self):
        is_nf, _ = detect_non_food_video("PRANK al mio amico - scherzo epico")
        assert is_nf is True

    def test_no_substring_false_positive(self):
        """'mma' should NOT match inside 'MAMMA' or 'TOMMASO'."""
        is_nf, _ = detect_non_food_video("PROVO TUTTE LE PASTE PRONTE VIVA LA MAMMA")
        assert is_nf is False

    def test_no_substring_tommaso(self):
        is_nf, _ = detect_non_food_video("TOMMASO PARADISO ha APERTO un RISTORANTE??")
        assert is_nf is False

    def test_case_insensitive(self):
        is_nf, _ = detect_non_food_video("BOXING TRAINING INTENSO")
        assert is_nf is True

    def test_description_catches_non_food(self):
        """Title may be vague, but description reveals non-food content."""
        is_nf, reason = detect_non_food_video(
            "COLPI SEGRETI PER CHI VUOLE GUAI",
            description="Scuola di Botte, ovvero Simone Cicalone",
        )
        assert is_nf is True
        assert "description" in reason

    def test_description_only_no_false_positive(self):
        """Food-related description should NOT trigger non-food filter."""
        is_nf, _ = detect_non_food_video(
            "Una giornata speciale",
            description="Sono andato a mangiare la pizza più buona di Roma",
        )
        assert is_nf is False


class TestDetectRecipeVideo:
    """Test recipe detection still works correctly."""

    def test_recipe_detected(self):
        is_r, _ = detect_recipe_video("La ricetta della carbonara perfetta")
        assert is_r is True

    def test_food_review_not_recipe(self):
        is_r, _ = detect_recipe_video("LA MIGLIOR PIZZA DI ROMA")
        assert is_r is False


class TestIsValidLocaleName:
    """Test locale name validation rejects generic words and keeps proper names."""

    def test_proper_name(self):
        assert _is_valid_locale_name("Antico Forno Roscioli") is True

    def test_da_prefix(self):
        assert _is_valid_locale_name("Da Marione") is True

    def test_single_generic_word(self):
        assert _is_valid_locale_name("pizzeria") is False

    def test_generic_with_article(self):
        assert _is_valid_locale_name("la pizzeria") is False

    def test_generic_combo(self):
        assert _is_valid_locale_name("forni e panifici") is False

    def test_proper_with_category(self):
        assert _is_valid_locale_name("Pizzeria Napoli") is True

    def test_too_short(self):
        assert _is_valid_locale_name("ab") is False

    def test_empty(self):
        assert _is_valid_locale_name("") is False


class TestCleanLocaleName:
    """Test locale name cleaning."""

    def test_strips_whitespace(self):
        assert _clean_locale_name("  Bonci  ") == "Bonci"

    def test_strips_punctuation(self):
        assert _clean_locale_name("'Da Trapani'") == "Da Trapani"

    def test_collapses_spaces(self):
        assert _clean_locale_name("Antico   Forno") == "Antico Forno"

    def test_removes_repeated_chars(self):
        result = _clean_locale_name("Naaaaapola")
        assert "aaaa" not in result


class TestIsFoodReviewVideo:
    """Test the LLM-based food-relevance gate (mocked LLM)."""

    def _mock_llm_response(self, answer: str):
        """Create a mock LLM that returns the given answer."""
        mock_llm = MagicMock()
        mock_llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": answer}}]
        }
        return mock_llm

    @patch("scripts.extract_locales.get_llm")
    def test_food_llm_no_overridden_by_title_intel(self, mock_get_llm):
        mock_get_llm.return_value = self._mock_llm_response("NO")
        from scripts.video_intelligence import VideoIntel

        intel = VideoIntel(video_type="multi_venue_tour", city="Roma")
        is_food, reason = check_food_video(
            "Roma criminale",
            "parliamo di boxe e allenamento",
            video_intel=intel,
        )
        assert is_food is True
        assert reason.startswith("Rules:")

    @patch("scripts.extract_locales.get_llm")
    def test_food_llm_no_overridden_by_loose_title(self, mock_get_llm):
        mock_get_llm.return_value = self._mock_llm_response("NO")
        from scripts.video_intelligence import VideoIntel

        is_food, reason = check_food_video(
            "Girls trip gastro a Roma",
            "parliamo di boxe",
            video_intel=VideoIntel(),
        )
        assert is_food is True
        assert "Override" in reason

    @patch("scripts.extract_locales.get_llm")
    def test_food_rules_skip_llm_for_criminale_title(self, mock_get_llm):
        from scripts.video_intelligence import VideoIntel

        is_food, reason = check_food_video(
            "Napoli criminale",
            "",
            video_intel=VideoIntel(),
        )
        assert is_food is True
        assert reason.startswith("Rules:")
        mock_get_llm.assert_not_called()

    @patch("scripts.extract_locales.get_llm")
    def test_food_video_passes(self, mock_get_llm):
        mock_get_llm.return_value = self._mock_llm_response("SI")
        is_food, reason = is_food_review_video(
            "LA MIGLIOR PIZZA DI ROMA",
            "andiamo a provare questa pizzeria fantastica che si chiama Bonci"
        )
        assert is_food is True
        assert "food video" in reason or reason.startswith("Rules:")

    @patch("scripts.extract_locales.get_llm")
    def test_non_food_video_blocked(self, mock_get_llm):
        mock_get_llm.return_value = self._mock_llm_response("NO")
        is_food, reason = is_food_review_video(
            "COLPI SEGRETI PER CHI VUOLE GUAI",
            "oggi facciamo un allenamento di boxe con Cicalone"
        )
        assert is_food is False
        assert "NOT" in reason

    @patch("scripts.extract_locales.get_llm")
    def test_ambiguous_defaults_to_false(self, mock_get_llm):
        """FORSE or ambiguous answers are treated as NO (strict mode)."""
        mock_get_llm.return_value = self._mock_llm_response("FORSE")
        is_food, reason = is_food_review_video(
            "Una giornata a Roma",
            "siamo usciti questa mattina per fare delle cose"
        )
        assert is_food is False
        assert "NOT" in reason

    @patch("scripts.extract_locales.get_llm")
    def test_no_substring_false_positive(self, mock_get_llm):
        """'SI' inside words like 'viSIte' or 'deSIdererebbe' must NOT match."""
        mock_get_llm.return_value = self._mock_llm_response(
            "NO. Il video non mostra visite a locali, non si desidererebbe"
        )
        is_food, reason = is_food_review_video(
            "Un video generico", "un testo qualsiasi"
        )
        assert is_food is False

    @patch("scripts.extract_locales.get_llm")
    def test_si_with_trailing_text(self, mock_get_llm):
        """'SI' followed by an explanation should still match."""
        mock_get_llm.return_value = self._mock_llm_response(
            "SI. The video clearly shows the blogger eating at a pizzeria."
        )
        is_food, reason = is_food_review_video(
            "Pizza review", "andiamo da Bonci"
        )
        assert is_food is True

    @patch("scripts.extract_locales.get_llm")
    def test_llm_unavailable_defaults_to_true(self, mock_get_llm):
        mock_get_llm.return_value = None
        is_food, reason = is_food_review_video(
            "Qualsiasi video", "qualsiasi testo"
        )
        assert is_food is True

    def test_empty_transcript_defaults_to_true(self):
        """No transcript → proceed (pre-transcript gate)."""
        is_food, reason = is_food_review_video("Un video", "")
        assert is_food is True
        assert "pre-transcript" in reason or "Rules" in reason

    @patch("scripts.extract_locales.get_llm")
    def test_llm_exception_defaults_to_true(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.create_chat_completion.side_effect = RuntimeError("LLM crashed")
        mock_get_llm.return_value = mock_llm
        is_food, reason = is_food_review_video(
            "Un video", "un testo qualsiasi per il test"
        )
        assert is_food is True
        assert "Error" in reason or "error" in reason.lower()


class TestExtractHintsFromDescription:
    """Test venue-name extraction from video descriptions."""

    def test_da_pattern(self):
        hints = extract_hints_from_description(
            "fare un salto da Mastrodonato in via Marmorata a Roma"
        )
        assert any("Mastrodonato" in h for h in hints)

    def test_fritti_di_pattern(self):
        hints = extract_hints_from_description(
            "voglio tutti i fritti di Mastrodonato"
        )
        assert any("Mastrodonato" in h for h in hints)

    def test_via_pattern(self):
        hints = extract_hints_from_description(
            "in via Marmorata 3 a Roma"
        )
        assert any("Marmorata" in h for h in hints)

    def test_empty_description(self):
        assert extract_hints_from_description("") == []

    def test_no_venues(self):
        hints = extract_hints_from_description("solo un video normale senza locali")
        assert hints == []

    def test_mangio_pattern(self):
        hints = extract_hints_from_description(
            "mangio tutti i fritti di Sant'Isidoro locale e pizzeria"
        )
        assert len(hints) >= 1


class TestParseDescriptionTimestamps:
    def test_mm_ss_and_labels(self):
        desc = "0:00 Intro\n1:30 Da Peppe\n12:05 Outro"
        rows = parse_description_timestamps(desc)
        assert len(rows) == 3
        assert rows[0]["timestamp"] == "0:00"
        assert "Intro" in rows[0]["label"]
        assert rows[1]["timestamp"] == "1:30"

    def test_hh_mm_ss(self):
        desc = "1:02:03 Long video chapter"
        rows = parse_description_timestamps(desc)
        assert len(rows) == 1
        assert rows[0]["timestamp"] == "1:02:03"

    def test_empty(self):
        assert parse_description_timestamps("") == []


class TestRatingNumericCore:
    def test_modifiers(self):
        assert rating_numeric_core("8--") == 8.0
        assert rating_numeric_core("6++") == 6.0
        assert rating_numeric_core("10") == 10.0


class TestGetLlmCaching:
    """Verify get_llm() caches the result in _llm_instance on direct load path."""

    def test_result_is_cached_after_first_call(self):
        import scripts.extract_locales as el

        original_instance = el._llm_instance
        original_future = el._llm_load_future
        try:
            el._llm_instance = None
            el._llm_load_future = None
            fake_model = object()
            call_count = {"n": 0}

            def _fake_load():
                call_count["n"] += 1
                return fake_model

            with patch.object(el, "_load_llm_impl", side_effect=_fake_load):
                result1 = el.get_llm()
                result2 = el.get_llm()

            assert result1 is fake_model
            assert result2 is fake_model
            # _load_llm_impl must only have been called ONCE (second call uses cache)
            assert call_count["n"] == 1
        finally:
            el._llm_instance = original_instance
            el._llm_load_future = original_future

    def test_none_result_not_cached(self):
        """When hardware disables LLM (returns None), we should retry on next call."""
        import scripts.extract_locales as el

        original_instance = el._llm_instance
        original_future = el._llm_load_future
        try:
            el._llm_instance = None
            el._llm_load_future = None
            call_count = {"n": 0}

            def _fake_load_none():
                call_count["n"] += 1
                return None

            with patch.object(el, "_load_llm_impl", side_effect=_fake_load_none):
                result1 = el.get_llm()
                result2 = el.get_llm()

            assert result1 is None
            assert result2 is None
            # Both calls should have tried to load (None is not cached)
            assert call_count["n"] == 2
        finally:
            el._llm_instance = original_instance
            el._llm_load_future = original_future
