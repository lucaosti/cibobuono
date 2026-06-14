"""
verify_locales.py — Verify extracted locales actually exist as real businesses.

Uses the Overpass API (free OpenStreetMap query API) to search for food
establishments near the geocoded coordinates matching the locale name.

Safeguards against API instability:
- Multiple Overpass mirror endpoints with automatic failover
- Rate limiting between every request (not just first attempt)
- Exponential backoff on failure
- Distinction between "API failed" vs "no matching place found"
- Cache only caches definitive results (found or genuinely not found),
  never caches API failures
"""

from __future__ import annotations

__author__ = "Luca Ostinelli"


import json
import os
import re
import tempfile
import time
import urllib.parse
import urllib.request

from thefuzz import fuzz

from scripts.utils import CACHE_DIR, ensure_dirs, setup_logging

logger = setup_logging("verify")

VERIFY_CACHE_FILE = CACHE_DIR / "verify_cache.json"

# Multiple Overpass mirrors for failover.
# Rotated on each retry so a single overloaded server doesn't block us.
OVERPASS_URLS = [
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

SEARCH_RADIUS_METERS = 500
FALLBACK_RADIUS_METERS = 1000
MIN_NAME_MATCH_SCORE = 80

_last_overpass_time = 0.0
OVERPASS_MIN_INTERVAL = 4.0  # seconds between ANY two Overpass requests


def _load_verify_cache() -> dict:
    if VERIFY_CACHE_FILE.exists():
        try:
            with open(VERIFY_CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_verify_cache(cache: dict) -> None:
    ensure_dirs()
    fd, tmp = tempfile.mkstemp(
        suffix=".json.tmp", prefix=".verify_cache.", dir=str(VERIFY_CACHE_FILE.parent), text=True
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
        os.replace(tmp, VERIFY_CACHE_FILE)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _rate_limit():
    """Wait until OVERPASS_MIN_INTERVAL has passed since last request."""
    global _last_overpass_time
    elapsed = time.time() - _last_overpass_time
    if elapsed < OVERPASS_MIN_INTERVAL:
        time.sleep(OVERPASS_MIN_INTERVAL - elapsed)
    _last_overpass_time = time.time()


def _normalize_name(name: str) -> str:
    name = name.lower().strip()
    name = re.sub(r"['\"\-–—.,;:!?()&/\\@#]", " ", name)
    name = re.sub(r"\s+", " ", name)
    return name.strip()


def _query_overpass_once(url: str, query: str, timeout: int = 25) -> list[dict] | None:
    """Single Overpass request. Returns list of places or None on failure."""
    _rate_limit()
    try:
        data = urllib.parse.urlencode({"data": query}).encode("utf-8")
        req = urllib.request.Request(
            url, data=data,
            headers={"User-Agent": "cibobuono_open_dataset/1.0"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))

        places = []
        for el in result.get("elements", []):
            tags = el.get("tags", {})
            name = tags.get("name", "")
            if not name:
                continue
            place_lat = el.get("center", {}).get("lat") or el.get("lat")
            place_lon = el.get("center", {}).get("lon") or el.get("lon")
            places.append({"name": name, "lat": place_lat, "lon": place_lon, "tags": tags})

        return places

    except Exception as e:
        logger.warning(f"Overpass [{url.split('/')[2]}] failed: {e}")
        return None


def _build_overpass_query(lat: float, lon: float, radius: int) -> str:
    return f"""
    [out:json][timeout:20];
    (
      nwr["amenity"~"restaurant|cafe|fast_food|bar|pub|ice_cream|food_court|biergarten"](around:{radius},{lat},{lon});
      nwr["shop"~"bakery|butcher|deli|pastry|confectionery|cheese|seafood|wine"](around:{radius},{lat},{lon});
      nwr["cuisine"](around:{radius},{lat},{lon});
    );
    out center tags;
    """


def query_overpass_with_failover(
    lat: float, lon: float, radius: int = SEARCH_RADIUS_METERS,
) -> tuple[list[dict] | None, bool]:
    """Query Overpass with automatic failover across mirrors.

    Returns:
        (places, api_ok) where:
        - places: list of found places, or empty list, or None if ALL mirrors failed
        - api_ok: True if at least one mirror returned a valid response (even if 0 places)
    """
    query = _build_overpass_query(lat, lon, radius)

    for i, url in enumerate(OVERPASS_URLS):
        places = _query_overpass_once(url, query)
        if places is not None:
            logger.debug(f"Overpass found {len(places)} food places near ({lat}, {lon})")
            return places, True
        # Backoff before trying next mirror
        if i < len(OVERPASS_URLS) - 1:
            wait = (i + 1) * 3
            logger.info(f"  Trying next Overpass mirror in {wait}s...")
            time.sleep(wait)

    logger.warning(f"All Overpass mirrors failed for ({lat}, {lon})")
    return None, False


def _find_best_match(locale_name: str, places: list[dict]) -> dict | None:
    """Find the best fuzzy match for a locale name in a list of OSM places."""
    norm_target = _normalize_name(locale_name)
    if not norm_target:
        return None

    best_match = None
    best_score = 0

    for place in places:
        norm_osm = _normalize_name(place.get("name", ""))
        if len(norm_osm) < 3:
            continue

        shorter = min(len(norm_target), len(norm_osm))
        longer = max(len(norm_target), len(norm_osm))
        if shorter / longer < 0.35:
            continue

        scores = [
            fuzz.ratio(norm_target, norm_osm),
            fuzz.token_sort_ratio(norm_target, norm_osm),
            fuzz.token_set_ratio(norm_target, norm_osm),
        ]
        if shorter / longer >= 0.5:
            scores.append(fuzz.partial_ratio(norm_target, norm_osm))

        score = max(scores)
        if score > best_score:
            best_score = score
            best_match = {**place, "_match_score": score}

    if best_match and best_score >= MIN_NAME_MATCH_SCORE:
        return best_match
    return None


def verify_locale_exists(
    locale_name: str, lat: float, lon: float, city: str = "",
) -> dict | None:
    """Verify that a named food locale actually exists near the given coordinates.

    Returns dict with verification info or None.
    """
    cache_key = f"{locale_name}|{lat}|{lon}".lower()
    cache = _load_verify_cache()
    if cache_key in cache:
        logger.debug(f"Verify cache hit: {cache_key}")
        cached = cache[cache_key]
        return cached if cached else None

    # Primary search (500m)
    places, api_ok = query_overpass_with_failover(lat, lon, SEARCH_RADIUS_METERS)

    # Fallback to wider radius ONLY if the API responded successfully
    # but found 0 places (i.e., the area is sparse, not an API error)
    if api_ok and places is not None and len(places) == 0:
        logger.info(f"  Widening search to {FALLBACK_RADIUS_METERS}m for '{locale_name}'")
        places, api_ok = query_overpass_with_failover(lat, lon, FALLBACK_RADIUS_METERS)

    # If ALL API calls failed, don't cache — we'll retry next pipeline run
    if not api_ok or places is None:
        logger.warning(
            f"✗ Could not verify '{locale_name}' ({city}) — "
            f"Overpass API unavailable. Will retry on next run."
        )
        return None

    match = _find_best_match(locale_name, places)

    if match:
        result = {
            "osm_name": match["name"],
            "match_score": match["_match_score"],
            "verified_lat": match.get("lat"),
            "verified_lon": match.get("lon"),
            "osm_amenity": match.get("tags", {}).get("amenity", ""),
            "osm_cuisine": match.get("tags", {}).get("cuisine", ""),
        }
        logger.info(
            f"✓ Verified '{locale_name}' ({city}) → "
            f"OSM: '{match['name']}' (score={match['_match_score']})"
        )
        cache[cache_key] = result
        _save_verify_cache(cache)
        return result

    logger.warning(
        f"✗ Could not verify '{locale_name}' ({city}) — "
        f"no matching food place near ({lat}, {lon}). "
        f"Checked {len(places)} OSM places."
    )
    cache[cache_key] = None
    _save_verify_cache(cache)
    return None


def verify_extractions(
    extractions: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Verify a list of geocoded extractions against OpenStreetMap.

    Returns (verified, unverified) extraction lists.
    """
    verified = []
    unverified = []

    for ext in extractions:
        name = ext.get("locale_name", "")
        lat = ext.get("lat")
        lon = ext.get("lon")
        city = ext.get("city", "")

        if lat is None or lon is None:
            unverified.append(ext)
            continue

        result = verify_locale_exists(name, lat, lon, city)
        if result:
            ext["osm_verified"] = True
            ext["osm_name"] = result["osm_name"]
            ext["osm_match_score"] = result["match_score"]
            if result.get("verified_lat") and result.get("verified_lon"):
                ext["lat"] = round(result["verified_lat"], 4)
                ext["lon"] = round(result["verified_lon"], 4)
            verified.append(ext)
        else:
            ext["osm_verified"] = False
            ext["_flag_reason"] = "osm_not_found"
            unverified.append(ext)

    logger.info(
        f"Verification: {len(verified)} verified, {len(unverified)} not found on OSM"
    )
    return verified, unverified


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print("Usage: python -m scripts.verify_locales 'Locale Name' lat lon ['city']")
        sys.exit(1)

    name = sys.argv[1]
    lat = float(sys.argv[2])
    lon = float(sys.argv[3])
    city = sys.argv[4] if len(sys.argv) > 4 else ""

    result = verify_locale_exists(name, lat, lon, city)
    if result:
        print(f"✓ Verified: {result}")
    else:
        print("✗ Not found on OpenStreetMap")
