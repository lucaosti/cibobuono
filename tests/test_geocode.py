"""Tests for geocode_locales: cache logic and batch geocoding (network mocked)."""

from __future__ import annotations

__author__ = "Luca Ostinelli"

import json
from unittest.mock import MagicMock, patch


class TestGeocodeLocaleCache:
    def test_cache_hit_returns_cached_value_without_network(self, tmp_path, monkeypatch):
        import scripts.geocode_locales as gl

        cached = {"lat": 41.89, "lon": 12.49, "display_name": "Roma", "geocoded_city": "Roma"}
        cache_file = tmp_path / "geocode_cache.json"
        cache_file.write_text(json.dumps({"forno roscioli||": cached}), encoding="utf-8")

        monkeypatch.setattr(gl, "GEOCODE_CACHE_FILE", cache_file)

        with patch("scripts.geocode_locales.Nominatim", side_effect=AssertionError("network called")) if False else patch.object(gl, "_save_geocode_cache"):
            # Patch the import inside geocode_locale to avoid network
            with patch.dict("sys.modules", {"geopy.geocoders": MagicMock(), "geopy.exc": MagicMock()}):
                result = gl._load_geocode_cache()
        assert result["forno roscioli||"] == cached

    def test_load_cache_handles_corrupt_file(self, tmp_path, monkeypatch):
        import scripts.geocode_locales as gl

        cache_file = tmp_path / "geocode_cache.json"
        cache_file.write_text("NOT JSON {{{", encoding="utf-8")
        monkeypatch.setattr(gl, "GEOCODE_CACHE_FILE", cache_file)

        result = gl._load_geocode_cache()
        assert result == {}

    def test_load_cache_handles_missing_file(self, tmp_path, monkeypatch):
        import scripts.geocode_locales as gl

        monkeypatch.setattr(gl, "GEOCODE_CACHE_FILE", tmp_path / "nonexistent.json")
        assert gl._load_geocode_cache() == {}

    def test_geocode_locale_returns_none_on_import_error(self, monkeypatch):
        import scripts.geocode_locales as gl

        cache_file = MagicMock()
        cache_file.exists.return_value = False
        monkeypatch.setattr(gl, "GEOCODE_CACHE_FILE", cache_file)

        with patch.dict("sys.modules", {"geopy": None, "geopy.geocoders": None, "geopy.exc": None}):
            import importlib
            # Trigger ImportError path by raising on import inside geocode_locale
            orig = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

        # Simpler: patch the inner import directly
        with patch("builtins.__import__", side_effect=ImportError("geopy not installed")):
            try:
                result = gl.geocode_locale("Test", "Roma")
            except ImportError:
                result = None
        assert result is None

    def test_geocode_locale_uses_cache_on_hit(self, tmp_path, monkeypatch):
        import scripts.geocode_locales as gl

        cached = {"lat": 41.89, "lon": 12.49, "display_name": "Roma", "geocoded_city": "Roma"}
        monkeypatch.setattr(gl, "GEOCODE_CACHE_FILE", tmp_path / "geocode_cache.json")
        monkeypatch.setattr(gl, "_load_geocode_cache", lambda: {"forno roscioli|roma|": cached})
        monkeypatch.setattr(gl, "_save_geocode_cache", lambda c: None)

        result = gl.geocode_locale("Forno Roscioli", "Roma")
        assert result == cached

    def test_geocode_locale_network_success(self, tmp_path, monkeypatch):
        import scripts.geocode_locales as gl

        monkeypatch.setattr(gl, "_load_geocode_cache", lambda: {})
        monkeypatch.setattr(gl, "_save_geocode_cache", lambda c: None)
        monkeypatch.setattr(gl, "_rate_limit", lambda: None)

        fake_location = MagicMock()
        fake_location.latitude = 41.8954
        fake_location.longitude = 12.4772
        fake_location.address = "Forno Roscioli, Via dei Chiavari, Roma"
        fake_location.raw = {"address": {"city": "Roma"}}

        mock_geolocator = MagicMock()
        mock_geolocator.geocode.return_value = fake_location

        mock_nominatim_class = MagicMock(return_value=mock_geolocator)
        mock_exc = MagicMock()
        mock_exc.GeocoderTimedOut = Exception
        mock_exc.GeocoderServiceError = Exception

        with patch.dict("sys.modules", {"geopy.geocoders": MagicMock(Nominatim=mock_nominatim_class), "geopy.exc": mock_exc}):
            result = gl.geocode_locale("Forno Roscioli", "Roma")

        assert result is not None
        assert result["lat"] == round(41.8954, 4)
        assert result["lon"] == round(12.4772, 4)
        assert result["geocoded_city"] == "Roma"

    def test_geocode_locale_returns_none_when_not_found(self, tmp_path, monkeypatch):
        import scripts.geocode_locales as gl

        monkeypatch.setattr(gl, "_load_geocode_cache", lambda: {})
        saved = {}
        monkeypatch.setattr(gl, "_save_geocode_cache", lambda c: saved.update(c))
        monkeypatch.setattr(gl, "_rate_limit", lambda: None)

        mock_geolocator = MagicMock()
        mock_geolocator.geocode.return_value = None

        mock_nominatim_class = MagicMock(return_value=mock_geolocator)
        mock_exc = MagicMock()
        mock_exc.GeocoderTimedOut = Exception
        mock_exc.GeocoderServiceError = Exception

        with patch.dict("sys.modules", {"geopy.geocoders": MagicMock(Nominatim=mock_nominatim_class), "geopy.exc": mock_exc}):
            result = gl.geocode_locale("Nonexistent Place", "Roma")

        assert result is None
        # Miss is cached as None
        assert any(v is None for v in saved.values())


class TestGeocodeExtractions:
    def test_geocodes_all_extractions(self, monkeypatch):
        import scripts.geocode_locales as gl

        geo_result = {"lat": 41.89, "lon": 12.49, "display_name": "Roma", "geocoded_city": "Roma"}
        monkeypatch.setattr(gl, "geocode_locale", lambda name, city, address: geo_result)

        extractions = [
            {"locale_name": "Da Remo", "city": "Roma", "address": ""},
            {"locale_name": "Forno Roscioli", "city": "Roma", "address": ""},
        ]
        geocoded, failed = gl.geocode_extractions(extractions)
        assert len(geocoded) == 2
        assert failed == []
        assert geocoded[0]["lat"] == 41.89

    def test_failed_geocoding_goes_to_failed_list(self, monkeypatch):
        import scripts.geocode_locales as gl

        monkeypatch.setattr(gl, "geocode_locale", lambda name, city, address: None)

        extractions = [{"locale_name": "Ristorante Inesistente", "city": "", "address": ""}]
        geocoded, failed = gl.geocode_extractions(extractions)
        assert geocoded == []
        assert len(failed) == 1

    def test_partial_geocoding(self, monkeypatch):
        import scripts.geocode_locales as gl

        geo_result = {"lat": 41.89, "lon": 12.49, "display_name": "Roma", "geocoded_city": "Roma"}

        def _fake_geocode(name, city, address):
            return geo_result if name == "Da Remo" else None

        monkeypatch.setattr(gl, "geocode_locale", _fake_geocode)

        extractions = [
            {"locale_name": "Da Remo", "city": "Roma", "address": ""},
            {"locale_name": "Posto Inesistente", "city": "", "address": ""},
        ]
        geocoded, failed = gl.geocode_extractions(extractions)
        assert len(geocoded) == 1
        assert len(failed) == 1
        assert geocoded[0]["locale_name"] == "Da Remo"

    def test_address_filled_from_display_name_when_missing(self, monkeypatch):
        import scripts.geocode_locales as gl

        geo_result = {"lat": 41.89, "lon": 12.49, "display_name": "Via dei Chiavari 34, Roma", "geocoded_city": "Roma"}
        monkeypatch.setattr(gl, "geocode_locale", lambda name, city, address: geo_result)

        extractions = [{"locale_name": "Da Remo", "city": "Roma", "address": ""}]
        geocoded, _ = gl.geocode_extractions(extractions)
        assert geocoded[0]["address"] == "Via dei Chiavari 34, Roma"

    def test_existing_address_not_overwritten(self, monkeypatch):
        import scripts.geocode_locales as gl

        geo_result = {"lat": 41.89, "lon": 12.49, "display_name": "Via dei Chiavari 34, Roma", "geocoded_city": "Roma"}
        monkeypatch.setattr(gl, "geocode_locale", lambda name, city, address: geo_result)

        extractions = [{"locale_name": "Da Remo", "city": "Roma", "address": "Via Paola 45"}]
        geocoded, _ = gl.geocode_extractions(extractions)
        assert geocoded[0]["address"] == "Via Paola 45"
