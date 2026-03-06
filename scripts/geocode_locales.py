"""
geocode_locales.py — Geocode locale names using Nominatim (OpenStreetMap).

Free, open-source geocoding with no API key required.
Rate limited to 1 request/second per Nominatim policy.
Caches results to avoid redundant requests.
"""

import json
import time

from scripts.utils import CACHE_DIR, ensure_dirs, setup_logging

logger = setup_logging("geocode")

# Cache file for geocoding results
GEOCODE_CACHE_FILE = CACHE_DIR / "geocode_cache.json"

# Rate limiting
_last_request_time = 0.0
RATE_LIMIT_SECONDS = 1.1  # Nominatim requires >= 1 second between requests


def _load_geocode_cache() -> dict:
    """Load geocoding cache from disk."""
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
    """Save geocoding cache to disk."""
    ensure_dirs()
    with open(GEOCODE_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def _rate_limit():
    """Enforce rate limiting for Nominatim."""
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

    # Build cache key
    cache_key = f"{name}|{city}|{address}".lower().strip()
    cache = _load_geocode_cache()
    if cache_key in cache:
        logger.debug(f"Geocode cache hit: {cache_key}")
        return cache[cache_key]

    geolocator = Nominatim(
        user_agent="cibobuono_open_dataset/1.0",
        timeout=10,
    )

    # Try multiple query strategies
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
            location = geolocator.geocode(query, exactly_one=True, language="it")
            if location:
                result = {
                    "lat": round(location.latitude, 4),
                    "lon": round(location.longitude, 4),
                    "display_name": location.address,
                }
                # Cache the result
                cache[cache_key] = result
                _save_geocode_cache(cache)
                logger.info(f"Geocoded '{query}' -> ({result['lat']}, {result['lon']})")
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
