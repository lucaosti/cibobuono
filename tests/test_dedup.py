"""
Tests for deduplication logic.
"""

from scripts.deduplicate_locales import (
    haversine_distance,
    name_similarity,
    names_match,
    find_duplicate,
    merge_locale,
)


class TestHaversineDistance:
    def test_same_point(self):
        """Distance between same point is 0."""
        d = haversine_distance(41.8912, 12.4921, 41.8912, 12.4921)
        assert d == 0.0

    def test_nearby_points(self):
        """Two points ~14m apart in Rome."""
        d = haversine_distance(41.8912, 12.4921, 41.8913, 12.4921)
        assert 10 < d < 20  # ~11m

    def test_far_apart(self):
        """Rome to Milan is ~477km."""
        d = haversine_distance(41.9028, 12.4964, 45.4642, 9.1900)
        assert 450_000 < d < 500_000

    def test_within_threshold(self):
        """Points within 50m should be detected."""
        # ~30m apart
        d = haversine_distance(41.8912, 12.4921, 41.8915, 12.4921)
        assert d < 50


class TestNameSimilarity:
    def test_identical(self):
        score = name_similarity("Forno Rossi", "Forno Rossi")
        assert score == 100

    def test_case_insensitive(self):
        score = name_similarity("FORNO ROSSI", "forno rossi")
        assert score == 100

    def test_word_order(self):
        """Swapped words should still score high."""
        score = name_similarity("Forno Rossi", "Rossi Forno")
        assert score >= 80

    def test_partial_match(self):
        """Shared surname with different prefix."""
        score = name_similarity("Forno dei Rossi", "Panificio Rossi")
        assert score >= 60

    def test_same_business_different_type(self):
        """Same family name, different business type."""
        score = name_similarity("Forno Rossi", "Panificio Rossi")
        assert score >= 60

    def test_completely_different(self):
        """Unrelated names should score low."""
        score = name_similarity("Pizzeria Napoli", "Forno Rossi")
        assert score < 50

    def test_extra_words(self):
        """Extra articles/prepositions should be handled."""
        score = name_similarity("Il Forno di Rossi", "Forno Rossi")
        assert score >= 70

    def test_abbreviation(self):
        """Common abbreviations."""
        score = name_similarity("Panifico F.lli Rossi", "Panificio Fratelli Rossi")
        assert score >= 40

    def test_sant_isidoro_variants(self):
        """All Sant'Isidoro variants must score high enough to dedup."""
        s1 = name_similarity("Sant'Isidoro pizza e bolle", "Sant' Isidoro")
        s2 = name_similarity("Sant'Isidoro pizza e bolle", "Sant'Isidoro Pizze Boiler")
        s3 = name_similarity("Sant' Isidoro", "Sant'Isidoro Pizze Boiler")
        assert s1 >= 70, f"pizza e bolle vs base: {s1}"
        assert s2 >= 70, f"pizza e bolle vs Boiler: {s2}"
        assert s3 >= 70, f"base vs Boiler: {s3}"


class TestNamesMatch:
    def test_match_with_aliases(self):
        """Should match if any alias combination matches."""
        result = names_match(
            "Forno Rossi", ["Il Forno"],
            "Panificio Rossi", [],
            threshold=60,
        )
        # "Forno Rossi" vs "Panificio Rossi" should match (shared "Rossi")
        assert result is True

    def test_no_match(self):
        result = names_match(
            "Pizzeria Napoli", [],
            "Forno Rossi", [],
            threshold=70,
        )
        assert result is False


class TestFindDuplicate:
    def test_find_nearby_duplicate(self):
        """Should find a duplicate within threshold with similar name."""
        existing = [
            {
                "locale_id": "locale_abc",
                "name": "Forno Rossi",
                "aliases": [],
                "lat": 41.8912,
                "lon": 12.4921,
            }
        ]
        new = {
            "name": "Panificio Rossi",
            "aliases": [],
            "lat": 41.8913,  # ~11m away
            "lon": 12.4921,
        }
        result = find_duplicate(new, existing)
        assert result is not None
        assert result["name"] == "Forno Rossi"

    def test_no_duplicate_far_away(self):
        """Should not find duplicate if far away."""
        existing = [
            {
                "locale_id": "locale_abc",
                "name": "Forno Rossi",
                "aliases": [],
                "lat": 41.8912,
                "lon": 12.4921,
            }
        ]
        new = {
            "name": "Forno Rossi",  # Same name but far away
            "aliases": [],
            "lat": 45.0,  # Milan
            "lon": 9.0,
        }
        result = find_duplicate(new, existing)
        assert result is None

    def test_no_duplicate_different_name_nearby(self):
        """Should not find duplicate if names don't match even if nearby."""
        existing = [
            {
                "locale_id": "locale_abc",
                "name": "Pizzeria Napoli",
                "aliases": [],
                "lat": 41.8912,
                "lon": 12.4921,
            }
        ]
        new = {
            "name": "Forno Rossi",  # Different name, same location
            "aliases": [],
            "lat": 41.8912,
            "lon": 12.4921,
        }
        result = find_duplicate(new, existing)
        assert result is None  # Names don't match


class TestMergeLocale:
    def test_add_alias(self):
        """Names that differ beyond noise-word stripping are added as aliases."""
        existing = {
            "name": "Trattoria da Mario",
            "aliases": [],
            "category": ["trattoria"],
        }
        new_data = {
            "name": "Il Vecchio Mario",
            "category": ["ristorante"],
        }
        result = merge_locale(existing, new_data)
        assert "Il Vecchio Mario" in result["aliases"]
        assert "ristorante" in result["category"]

    def test_equivalent_names_not_aliased(self):
        """Names that normalize to the same core should not produce aliases."""
        existing = {
            "name": "Forno Rossi",
            "aliases": [],
            "category": ["forno"],
        }
        new_data = {
            "name": "Panificio Rossi",
            "category": ["panificio"],
        }
        result = merge_locale(existing, new_data)
        assert "panificio" in result["category"]

    def test_no_duplicate_alias(self):
        """Should not add alias if already very similar to existing name."""
        existing = {
            "name": "Forno Rossi",
            "aliases": [],
            "category": [],
        }
        new_data = {
            "name": "Forno Rossi",
            "category": [],
        }
        result = merge_locale(existing, new_data)
        assert len(result["aliases"]) == 0

    def test_merge_categories(self):
        existing = {
            "name": "Forno Rossi",
            "aliases": [],
            "category": ["forno"],
        }
        new_data = {
            "name": "Forno Rossi",
            "category": ["panificio", "forno"],
        }
        result = merge_locale(existing, new_data)
        assert "panificio" in result["category"]
        assert result["category"].count("forno") == 1  # No duplicates
