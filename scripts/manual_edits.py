"""
manual_edits.py — Human corrections: remove false visits, add verified venues.

Used by the dashboard to fix extraction mistakes without editing JSON by hand.
"""

from __future__ import annotations

__author__ = "Luca Ostinelli"

from scripts.schemas import VideoStatus, timestamp_to_seconds
from scripts.utils import (
    CORRECTIONS_JSON,
    FLAGGED_SEGMENTS_JSON,
    LOCALES_JSON,
    VIDEOS_JSON,
    VISITS_JSON,
    load_json,
    save_json,
    save_json_split,
    today_str,
    setup_logging,
)

logger = setup_logging("manual_edits")

MIN_PUBLISH_OSM_SCORE = 85


def list_visits(*, limit: int = 100, video_id: str = "") -> list[dict]:
    """All visits enriched for the corrections UI."""
    locales = {l["locale_id"]: l for l in load_json(LOCALES_JSON)}
    titles = {v["video_id"]: v.get("title", "") for v in load_json(VIDEOS_JSON)}
    out: list[dict] = []
    for v in reversed(load_json(VISITS_JSON)):
        if video_id and v.get("video_id") != video_id:
            continue
        loc = locales.get(v.get("locale_id", ""), {})
        out.append(
            {
                "visit_id": v.get("visit_id"),
                "locale_id": v.get("locale_id"),
                "locale_name": loc.get("name", "?"),
                "city": loc.get("city", ""),
                "address": loc.get("address", ""),
                "lat": loc.get("lat"),
                "lon": loc.get("lon"),
                "video_id": v.get("video_id"),
                "video_title": titles.get(v.get("video_id", ""), "")[:100],
                "timestamp_start": v.get("timestamp_start"),
                "timestamp_end": v.get("timestamp_end"),
                "youtube_url": v.get("youtube_url", ""),
                "rating": v.get("rating"),
                "confidence": v.get("llm_confidence"),
                "notes": (v.get("notes") or "")[:200],
                "extraction_date": v.get("extraction_date"),
            }
        )
        if len(out) >= limit:
            break
    return out


def remove_visit(
    visit_id: str,
    *,
    hide_locale: bool = False,
    reason: str = "",
) -> tuple[bool, str]:
    """Delete a visit; optionally hide the locale site-wide via corrections.json."""
    visits = load_json(VISITS_JSON)
    target = next((v for v in visits if v.get("visit_id") == visit_id), None)
    if not target:
        return False, "Visita non trovata"

    locale_id = target.get("locale_id", "")
    visits = [v for v in visits if v.get("visit_id") != visit_id]
    save_json_split(VISITS_JSON, visits)

    msg = f"Visita {visit_id} rimossa"
    if hide_locale and locale_id:
        corrections = load_json(CORRECTIONS_JSON)
        if not any(c.get("locale_id") == locale_id and c.get("type") == "hide" for c in corrections):
            corrections.append(
                {
                    "locale_id": locale_id,
                    "type": "hide",
                    "reason": (reason or "Rimosso manualmente dalla dashboard")[:300],
                }
            )
            save_json(CORRECTIONS_JSON, corrections)
        msg += f"; locale {locale_id} nascosto"
    logger.info(msg)
    return True, msg


def _video_meta(video_id: str) -> tuple[str, str]:
    for v in load_json(VIDEOS_JSON):
        if v.get("video_id") == video_id:
            return v.get("channel_id", ""), v.get("publish_date", today_str())
    return "", today_str()


def _extraction_from_fields(
    locale_name: str,
    *,
    city: str = "",
    address: str = "",
    lat: float | None = None,
    lon: float | None = None,
    timestamp_start: str = "0:00",
    timestamp_end: str = "1:30",
    rating: str | None = None,
    notes: str = "",
    category: list[str] | None = None,
    confidence: float = 0.95,
    osm_verified: bool = False,
    osm_match_score: int | None = None,
) -> dict:
    try:
        mention_time = float(timestamp_to_seconds(timestamp_start))
    except (ValueError, TypeError):
        mention_time = 0.0
    return {
        "locale_name": locale_name.strip(),
        "city": city.strip(),
        "address": address.strip(),
        "lat": lat,
        "lon": lon,
        "category": category or ["ristorante"],
        "rating": rating,
        "sentiment": "neutral",
        "notes": notes.strip() or "Aggiunto manualmente",
        "rubrica": "",
        "confidence": confidence,
        "chunk_start": timestamp_start,
        "chunk_end": timestamp_end,
        "chunk_start_seconds": mention_time,
        "mention_time": mention_time,
        "mention_timestamp": timestamp_start,
        "verified": True,
        "osm_verified": osm_verified,
        "osm_match_score": osm_match_score,
        "_source": "manual",
    }


def add_manual_visit(
    *,
    locale_name: str,
    video_id: str,
    timestamp_start: str,
    timestamp_end: str = "1:30",
    city: str = "",
    address: str = "",
    rating: str | None = None,
    notes: str = "",
    require_osm: bool = True,
) -> tuple[bool, str, dict | None]:
    """
    Geocode + optional OSM verify, then create locale + visit.

    Returns (ok, message, visit_dict|None).
    """
    locale_name = locale_name.strip()
    if len(locale_name) < 2:
        return False, "Nome locale troppo corto", None
    if not video_id:
        return False, "video_id obbligatorio", None

    channel_id, publish_date = _video_meta(video_id)
    if not channel_id:
        return False, f"Video {video_id} non trovato in videos.json", None

    from scripts.geocode_locales import geocode_locale
    from scripts.verify_locales import verify_locale_exists
    from scripts.deduplicate_locales import deduplicate_locales
    from scripts.populate_json import populate_visits

    ext = _extraction_from_fields(
        locale_name,
        city=city,
        address=address,
        timestamp_start=timestamp_start,
        timestamp_end=timestamp_end,
        rating=rating,
        notes=notes,
        confidence=0.95,
    )

    geo = geocode_locale(locale_name, city, address)
    if not geo:
        return False, "Geocoding fallito — controlla nome e città", None

    ext["lat"] = geo["lat"]
    ext["lon"] = geo["lon"]
    if not ext.get("address"):
        ext["address"] = geo.get("display_name", "")

    if require_osm:
        osm = verify_locale_exists(locale_name, geo["lat"], geo["lon"], city)
        if not osm:
            return (
                False,
                "Non trovato su OpenStreetMap vicino alle coordinate — "
                "verifica nome/città o disattiva require_osm",
                None,
            )
        score = int(osm.get("match_score", 0))
        if score < MIN_PUBLISH_OSM_SCORE:
            return (
                False,
                f"Match OSM debole ({score}% < {MIN_PUBLISH_OSM_SCORE}%) — "
                f"nome OSM: '{osm.get('osm_name')}'",
                None,
            )
        ext["osm_verified"] = True
        ext["osm_match_score"] = score
        ext["osm_name"] = osm.get("osm_name")
        if osm.get("verified_lat") and osm.get("verified_lon"):
            ext["lat"] = round(float(osm["verified_lat"]), 4)
            ext["lon"] = round(float(osm["verified_lon"]), 4)

    _, mapping = deduplicate_locales([ext])
    if not mapping:
        return False, "Deduplicazione fallita", None

    new_visits = populate_visits(mapping, video_id, channel_id, publish_date)
    if not new_visits:
        return False, "Visita già presente o validazione fallita", None

    visit = new_visits[0]
    logger.info("Manual visit added: %s @ %s", locale_name, video_id)
    return True, f"Visita creata: {visit.get('visit_id')}", visit


def promote_flagged_to_visit(
    *,
    video_id: str,
    timestamp_start: str,
    locale_name: str | None = None,
    city: str = "",
    rating: str | None = None,
    notes: str = "",
) -> tuple[bool, str]:
    """Approve a flagged segment → geocode, OSM verify, create visit."""
    flagged = load_json(FLAGGED_SEGMENTS_JSON)
    seg = None
    for s in flagged:
        if s.get("video_id") == video_id and s.get("timestamp_start") == timestamp_start:
            if locale_name and s.get("locale_name") and s.get("locale_name") != locale_name:
                continue
            seg = s
            break
    if not seg:
        return False, "Segmento flaggato non trovato"

    name = (locale_name or seg.get("locale_name") or "").strip()
    if not name:
        return False, "Nome locale mancante"

    ok, msg, _ = add_manual_visit(
        locale_name=name,
        video_id=video_id,
        timestamp_start=seg.get("timestamp_start", timestamp_start),
        timestamp_end=seg.get("timestamp_end", "1:30"),
        city=city or (seg.get("city") or ""),
        rating=rating or seg.get("rating"),
        notes=notes or (seg.get("extracted_text") or ""),
        require_osm=True,
    )
    if not ok:
        return False, msg

    for s in flagged:
        if s.get("video_id") == video_id and s.get("timestamp_start") == timestamp_start:
            s["reviewed_by_human"] = True
            s["reviewed_date"] = today_str()
            s["review_action"] = "approve"
            if notes:
                s["review_notes"] = notes[:500]
            break
    save_json(FLAGGED_SEGMENTS_JSON, flagged)
    return True, msg
