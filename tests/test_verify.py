"""
Tests for verify_locales module: name normalization, fuzzy matching, verification flow.
"""

__author__ = "Luca Ostinelli"

from scripts.verify_locales import _normalize_name, _find_best_match


class TestNormalizeName:
    """Test locale name normalization for comparison."""

    def test_basic_lowercase(self):
        assert _normalize_name("Da Marione") == "da marione"

    def test_strip_punctuation(self):
        assert _normalize_name("L'Antico Forno") == "l antico forno"

    def test_collapse_spaces(self):
        assert _normalize_name("  Pizzeria   Napoli  ") == "pizzeria napoli"

    def test_strip_special_chars(self):
        assert _normalize_name("Café & Bar - Roma") == "café bar roma"

    def test_empty(self):
        assert _normalize_name("") == ""


class TestFindBestMatch:
    """Test fuzzy matching of locale names against OSM places."""

    def _make_place(self, name):
        return {"name": name, "lat": 41.9, "lon": 12.5, "tags": {"amenity": "restaurant"}}

    def test_exact_match(self):
        places = [self._make_place("Da Marione"), self._make_place("Pizza Roma")]
        match = _find_best_match("Da Marione", places)
        assert match is not None
        assert match["name"] == "Da Marione"
        assert match["_match_score"] == 100

    def test_case_insensitive_match(self):
        places = [self._make_place("ROSCIOLI")]
        match = _find_best_match("roscioli", places)
        assert match is not None
        assert match["_match_score"] >= 80

    def test_partial_match(self):
        places = [self._make_place("Antico Forno Roscioli")]
        match = _find_best_match("Roscioli", places)
        assert match is not None
        assert match["_match_score"] >= 80

    def test_no_match(self):
        places = [self._make_place("Pizzeria Napoli"), self._make_place("Bar San Marco")]
        match = _find_best_match("Scuola del botto", places)
        assert match is None or match["_match_score"] < 80

    def test_empty_places(self):
        match = _find_best_match("Da Marione", [])
        assert match is None

    def test_fuzzy_transcription_error(self):
        """Whisper might garble names; fuzzy should still match."""
        places = [self._make_place("Vittoria Spaziale")]
        match = _find_best_match("Vittoria Spaziale", places)
        assert match is not None
        assert match["_match_score"] >= 90

    def test_token_order_invariant(self):
        """token_sort_ratio handles word reordering."""
        places = [self._make_place("Forno Antico Roscioli")]
        match = _find_best_match("Roscioli Forno Antico", places)
        assert match is not None
        assert match["_match_score"] >= 80

    def test_short_osm_name_rejected(self):
        """Very short OSM names like 'aT' must NOT match long locale names.

        This was the original bug: partial_ratio('scuola del latte', 'at') → 100
        because 'at' appears inside 'latte'.
        """
        places = [self._make_place("aT"), self._make_place("Ba")]
        match = _find_best_match("Scuola del Latte", places)
        assert match is None

    def test_extreme_length_ratio_rejected(self):
        """OSM name much shorter than locale name should not match."""
        places = [self._make_place("Rio")]
        match = _find_best_match("Ristorante Rio Grande di Mario", places)
        assert match is None

    def test_empty_target_name(self):
        """Empty locale name should return None."""
        places = [self._make_place("Da Marione")]
        match = _find_best_match("", places)
        assert match is None

    def test_montecitorio_false_match_rejected(self):
        """'Montecitorio' must NOT match 'Pizzeria La Montecarlo' (score ~73)."""
        places = [self._make_place("Pizzeria La Montecarlo")]
        match = _find_best_match("Montecitorio", places)
        assert match is None
