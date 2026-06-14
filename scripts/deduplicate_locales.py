"""
deduplicate_locales.py — Deduplicate locales by distance and name similarity.

Uses haversine distance (<200m) AND fuzzy name matching (thefuzz) to identify
duplicate locales. Merges aliases when duplicates are found.

The deduplication requires BOTH conditions to match:
- Geographic proximity: within 200 meters (wider because geocoding can be
  imprecise — Nominatim often defaults to a city center for vague addresses)
- Name similarity: ≥ 70 after noise-word normalization (generic category words
  like "pizzeria", "forno", and articles are stripped before comparison)
"""

from __future__ import annotations

__author__ = "Luca Ostinelli"


import math
from urllib.parse import quote_plus

from scripts.utils import (
    DEDUP_DISTANCE_METERS,
    DEDUP_NAME_SIMILARITY_THRESHOLD,
    LOCALES_JSON,
    load_json,
    save_json_split,
    setup_logging,
)
from scripts.schemas import Locale, generate_locale_id

logger = setup_logging("dedup")


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate distance in meters between two points using the Haversine formula.
    """
    R = 6371000  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


_NOISE_WORDS = {
    "pizza", "pizze", "pizzeria", "forno", "bar", "ristorante", "trattoria",
    "osteria", "gelateria", "pub", "birreria", "enoteca", "bottega",
    "e", "di", "da", "del", "della", "dei", "il", "la", "le", "lo",
    "bolle", "boiler", "station", "house", "casa",
}


def _normalize_for_dedup(name: str) -> str:
    """Strip noise words and punctuation so core tokens are compared."""
    import re as _re
    name = name.lower().strip()
    name = _re.sub(r"[''`]", "'", name)
    name = _re.sub(r"[^\w\s']", " ", name)
    tokens = [t for t in name.split() if t not in _NOISE_WORDS]
    return " ".join(tokens) if tokens else name.lower().strip()


def name_similarity(name1: str, name2: str) -> int:
    """
    Calculate fuzzy similarity between two locale names.

    First compares core tokens (stripped of generic category words and
    articles) so that "Sant'Isidoro pizza e bolle" and "Sant'Isidoro Pizze
    Boiler" get a high score.  Falls back to raw token_set_ratio when the
    normalized forms are too short to be meaningful.

    Returns score 0-100.
    """
    try:
        from thefuzz import fuzz
    except ImportError:
        logger.error("thefuzz not installed. Install with: pip install 'thefuzz[speedup]'")
        return 100 if name1.lower().strip() == name2.lower().strip() else 0

    n1 = _normalize_for_dedup(name1)
    n2 = _normalize_for_dedup(name2)

    scores = [
        fuzz.token_set_ratio(name1.lower(), name2.lower()),
        fuzz.token_set_ratio(n1, n2),
        fuzz.ratio(n1, n2),
    ]
    return max(scores)


def names_match(name1: str, aliases1: list[str], name2: str, aliases2: list[str],
                threshold: int = DEDUP_NAME_SIMILARITY_THRESHOLD) -> bool:
    """
    Check if two locales have similar names, considering all aliases.
    Returns True if ANY pair of names/aliases exceeds the threshold.
    """
    all_names1 = [name1] + aliases1
    all_names2 = [name2] + aliases2

    for n1 in all_names1:
        for n2 in all_names2:
            if name_similarity(n1, n2) >= threshold:
                return True
    return False


def find_duplicate(
    locale: dict,
    existing_locales: list[dict],
    distance_threshold: float = DEDUP_DISTANCE_METERS,
    name_threshold: int = DEDUP_NAME_SIMILARITY_THRESHOLD,
) -> dict | None:
    """
    Find a duplicate of the given locale in the existing list.
    Both distance AND name similarity must match.
    
    Returns the matching existing locale, or None.
    """
    lat = locale.get("lat", 0)
    lon = locale.get("lon", 0)
    name = locale.get("name", locale.get("locale_name", ""))
    aliases = locale.get("aliases", [])

    for existing in existing_locales:
        e_lat = existing.get("lat", 0)
        e_lon = existing.get("lon", 0)
        e_name = existing.get("name", "")
        e_aliases = existing.get("aliases", [])

        # Check distance first (cheaper)
        dist = haversine_distance(lat, lon, e_lat, e_lon)
        if dist > distance_threshold:
            continue

        # Check name similarity
        if names_match(name, aliases, e_name, e_aliases, name_threshold):
            logger.info(
                f"Duplicate found: '{name}' matches '{e_name}' "
                f"(distance: {dist:.0f}m, name similarity match)"
            )
            return existing

    return None


def merge_locale(existing: dict, new_data: dict) -> dict:
    """
    Merge new locale data into an existing locale.
    Adds new name as alias if it differs from existing names.
    """
    new_name = new_data.get("name", new_data.get("locale_name", ""))
    existing_name = existing.get("name", "")
    existing_aliases = existing.get("aliases", [])

    if new_name and new_name != existing_name:
        all_names = [existing_name] + existing_aliases
        # Use simple case-insensitive + whitespace-collapsed comparison for
        # alias detection.  The heavy noise-word-stripping similarity used for
        # dedup is too aggressive here ("Panificio Rossi" ≠ "Forno Rossi" as
        # display names even though they represent the same business).
        from thefuzz import fuzz
        is_known = any(
            fuzz.ratio(new_name.lower().strip(), n.lower().strip()) >= 90
            for n in all_names
        )
        if not is_known:
            existing_aliases.append(new_name)
            existing["aliases"] = existing_aliases
            logger.info(f"Added alias '{new_name}' to locale '{existing_name}'")

    # Merge categories
    new_cats = new_data.get("category", [])
    existing_cats = existing.get("category", [])
    for cat in new_cats:
        if cat not in existing_cats:
            existing_cats.append(cat)
    existing["category"] = existing_cats

    return existing


def deduplicate_locales(new_locales: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Deduplicate new locales against existing locales.json.
    
    Returns:
        (updated_all_locales, mapping) where mapping maps new locale data to locale_ids
    """
    existing = load_json(LOCALES_JSON)
    locale_mapping = []  # tracks which locale_id each new extraction maps to

    for new_locale in new_locales:
        name = new_locale.get("locale_name", new_locale.get("name", ""))
        lat = new_locale.get("lat", 0)
        lon = new_locale.get("lon", 0)

        duplicate = find_duplicate(new_locale, existing)

        if duplicate:
            merge_locale(duplicate, new_locale)
            locale_mapping.append({
                "extraction": new_locale,
                "locale_id": duplicate["locale_id"],
                "action": "merged",
            })
        else:
            locale_id = generate_locale_id(name, lat, lon)
            city = new_locale.get("city", "")
            query = f"{name}, {city}" if city else name
            google_maps_url = f"https://www.google.com/maps/search/?api=1&query={quote_plus(query)}"
            locale_entry = {
                "locale_id": locale_id,
                "name": name,
                "aliases": [],
                "address": new_locale.get("address", ""),
                "city": city,
                "lat": lat,
                "lon": lon,
                "category": new_locale.get("category", []),
                "google_maps_url": google_maps_url,
            }
            try:
                Locale(**locale_entry)
            except Exception as e:
                logger.error(f"Locale validation failed for '{name}': {e}")
                continue
            existing.append(locale_entry)
            locale_mapping.append({
                "extraction": new_locale,
                "locale_id": locale_id,
                "action": "created",
            })
            logger.info(f"New locale: '{name}' ({locale_id})")

    save_json_split(LOCALES_JSON, existing)
    logger.info(f"Locales after dedup: {len(existing)} total")

    return existing, locale_mapping


if __name__ == "__main__":
    # Test deduplication with sample data
    sample = [
        {"locale_name": "Forno Rossi", "city": "Roma", "lat": 41.8912, "lon": 12.4921, "category": ["forno"]},
        {"locale_name": "Panificio Rossi", "city": "Roma", "lat": 41.8913, "lon": 12.4922, "category": ["panificio"]},
    ]

    print(f"Name similarity: {name_similarity('Forno Rossi', 'Panificio Rossi')}")
    print(f"Name similarity: {name_similarity('Forno dei Rossi', 'Forno Rossi')}")
    print(f"Name similarity: {name_similarity('Pizzeria Napoli', 'Forno Rossi')}")

    dist = haversine_distance(41.8912, 12.4921, 41.8913, 12.4922)
    print(f"Distance between samples: {dist:.1f}m")
