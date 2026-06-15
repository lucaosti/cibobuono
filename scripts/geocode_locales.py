"""
geocode_locales.py — Geocode locale names using Nominatim (OpenStreetMap).

Free, open-source geocoding with no API key required.
Rate limited to 1 request/second per Nominatim policy.
Caches results to avoid redundant requests.
"""

from __future__ import annotations

__author__ = "Luca Ostinelli"

import json
import os
import re
import tempfile
import time

from scripts.utils import CACHE_DIR, CONTENT_LANGUAGE, ensure_dirs, setup_logging

logger = setup_logging("geocode")

# Italian cities and Rome neighborhoods that appear in YouTube food video titles.
# Used by _clean_city() to extract a usable city name from noisy strings like
# "nuova HIT a CENTOCELLE" → "Centocelle".
_KNOWN_CITIES: frozenset[str] = frozenset({
    "Roma", "Milano", "Napoli", "Torino", "Firenze", "Bologna", "Venezia",
    "Genova", "Palermo", "Catania", "Bari", "Verona", "Padova", "Brescia",
    "Bergamo", "Parma", "Modena", "Viterbo", "Frosinone", "Latina", "Rieti",
    "Caserta", "Salerno", "Messina", "Lecce", "Reggio Calabria", "Perugia",
    "Ancona", "Trieste", "Cagliari", "Bolzano", "Trento", "Udine", "Ravenna",
    "Ferrara", "Piacenza", "Rimini", "Pesaro", "Foggia", "Taranto", "Matera",
    "Cosenza", "Catanzaro", "Reggio Emilia", "Prato", "Livorno", "Pisa",
    "Arezzo", "Siena", "Grosseto", "Lucca", "Pistoia", "Massa", "Carrara",
    "Berlino", "Parigi", "Londra", "Madrid", "New York", "Tokyo", "Osaka",
    "Napoli", "Berlino",
})
_ROMAN_NEIGHBORHOODS: frozenset[str] = frozenset({
    "Centocelle", "Pigneto", "Testaccio", "Trastevere", "Ostiense", "Prati",
    "Nomentano", "Tiburtino", "Prenestino", "Tuscolano", "Esquilino", "Monti",
    "Parioli", "Trionfale", "Garbatella", "Torpignattara", "Quadraro",
    "Don Bosco", "Casilino", "Marconi", "Portuense", "Gianicolense",
    "Magliana", "Acilia", "Ostia", "Fiumicino", "Ciampino",
})
# Neighborhoods that map to their parent city for geocoding purposes.
_NEIGHBORHOOD_TO_CITY: dict[str, str] = {n.lower(): "Roma" for n in _ROMAN_NEIGHBORHOODS}


def _clean_city(city: str) -> str:
    """Extract a usable city name from a potentially noisy city string.

    Handles titles like "nuova HIT a CENTOCELLE" → "Centocelle" and
    bare neighborhood names like "Centocelle" → "Roma".
    """
    if not city:
        return ""
    city = city.strip()

    # Direct match against known cities / neighborhoods (case-insensitive)
    city_lower = city.lower()
    for known in _KNOWN_CITIES:
        if city_lower == known.lower():
            return known
    for nbh in _ROMAN_NEIGHBORHOODS:
        if city_lower == nbh.lower():
            return "Roma"

    # Substring search: find a known city/neighborhood within the noisy string
    for known in sorted(_KNOWN_CITIES, key=len, reverse=True):
        if re.search(r'\b' + re.escape(known) + r'\b', city, re.IGNORECASE):
            return known
    for nbh in sorted(_ROMAN_NEIGHBORHOODS, key=len, reverse=True):
        if re.search(r'\b' + re.escape(nbh) + r'\b', city, re.IGNORECASE):
            logger.debug(f"City cleaned: '{city}' → 'Roma' via neighborhood '{nbh}'")
            return "Roma"

    # Last resort: extract the last word(s) that start with an uppercase letter
    words = city.split()
    tail: list[str] = []
    for w in reversed(words):
        w_clean = re.sub(r"[^\w]", "", w)
        if w_clean and w_clean[0].isupper() and len(w_clean) >= 3:
            tail.insert(0, w_clean)
        elif tail:
            break
    if tail:
        candidate = " ".join(tail)
        if candidate.lower() != city.lower():
            logger.debug(f"City cleaned: '{city}' → '{candidate}'")
        return candidate

    return city

# Cache file for geocoding results
GEOCODE_CACHE_FILE = CACHE_DIR / "geocode_cache.json"

# Rate limiting
_last_request_time = 0.0
RATE_LIMIT_SECONDS = 1.1  # Nominatim requires >= 1 second between requests


def _load_geocode_cache() -> dict:
    if GEOCODE_CACHE_FILE.exists():
        try:
            with open(GEOCODE_CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
        except OSError:
            return {}
    return {}


def _save_geocode_cache(cache: dict) -> None:
    ensure_dirs()
    fd, tmp = tempfile.mkstemp(
        suffix=".json.tmp", prefix=".geocode_cache.", dir=str(GEOCODE_CACHE_FILE.parent), text=True
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
        os.replace(tmp, GEOCODE_CACHE_FILE)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _rate_limit():
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < RATE_LIMIT_SECONDS:
        time.sleep(RATE_LIMIT_SECONDS - elapsed)
    _last_request_time = time.time()


def geocode_locale(
    name: str,
    city: str = "",
    address: str = "",
    country: str = "Italy",
) -> dict | None:
    """
    Geocode a locale using Nominatim.
    
    Args:
        name: Locale name
        city: City name (helps accuracy)
        address: Street address if known
        country: Country (default: Italy)
    
    Returns:
        dict with lat, lon, display_name, address or None if not found
    """
    try:
        from geopy.geocoders import Nominatim
        from geopy.exc import GeocoderTimedOut, GeocoderServiceError
    except ImportError:
        logger.error("geopy not installed. Install with: pip install geopy")
        return None

    city = _clean_city(city)
    cache_key = f"{name}|{city}|{address}".lower().strip()
    cache = _load_geocode_cache()
    if cache_key in cache:
        logger.debug(f"Geocode cache hit: {cache_key}")
        return cache[cache_key]

    geolocator = Nominatim(
        user_agent="cibobuono_open_dataset/1.0",
        timeout=10,
    )

    queries = []
    if address and city:
        queries.append(f"{name}, {address}, {city}, {country}")
        queries.append(f"{address}, {city}, {country}")
    if city:
        queries.append(f"{name}, {city}, {country}")
    queries.append(f"{name}, {country}")
    if city:
        queries.append(f"{city}, {country}")

    for query in queries:
        try:
            _rate_limit()
            location = geolocator.geocode(
                query, exactly_one=True, language=CONTENT_LANGUAGE
            )
            if location:
                raw_addr = (location.raw or {}).get("address", {})
                geocoded_city = (
                    raw_addr.get("city")
                    or raw_addr.get("town")
                    or raw_addr.get("village")
                    or ""
                )
                result = {
                    "lat": round(location.latitude, 4),
                    "lon": round(location.longitude, 4),
                    "display_name": location.address,
                    "geocoded_city": geocoded_city,
                }
                cache[cache_key] = result
                _save_geocode_cache(cache)
                logger.info(f"Geocoded '{query}' -> ({result['lat']}, {result['lon']}), city={geocoded_city!r}")
                return result

        except (GeocoderTimedOut, GeocoderServiceError) as e:
            logger.warning(f"Geocoding error for '{query}': {e}")
            time.sleep(2)
            continue
        except Exception as e:
            logger.warning(f"Unexpected geocoding error for '{query}': {e}")
            continue

    logger.warning(f"Could not geocode: {name} ({city})")
    # Cache the miss to avoid retrying
    cache[cache_key] = None
    _save_geocode_cache(cache)
    return None


def geocode_extractions(extractions: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Geocode a list of extracted locales.
    Adds lat, lon to each extraction.
    
    Returns:
        (geocoded, failed) — successfully geocoded and failed extraction lists.
    """
    geocoded = []
    failed = []

    for ext in extractions:
        name = ext.get("locale_name", "")
        city = ext.get("city", "")
        address = ext.get("address", "")

        result = geocode_locale(name, city, address)
        if result:
            ext["lat"] = result["lat"]
            ext["lon"] = result["lon"]
            ext["geocoded_city"] = result.get("geocoded_city", "")
            if not ext.get("address") and result.get("display_name"):
                ext["address"] = result["display_name"]
            geocoded.append(ext)
        else:
            failed.append(ext)

    logger.info(f"Geocoded: {len(geocoded)} success, {len(failed)} failed")
    return geocoded, failed


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m scripts.geocode_locales 'Locale Name' ['City']")
        sys.exit(1)

    name = sys.argv[1]
    city = sys.argv[2] if len(sys.argv) > 2 else ""
    result = geocode_locale(name, city)
    if result:
        print(f"Found: {result}")
    else:
        print("Not found")
