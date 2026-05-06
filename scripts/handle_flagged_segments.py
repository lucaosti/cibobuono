"""
handle_flagged_segments.py — Process manually reviewed flagged segments.

After a human reviews flagged_segments.json and fills in missing fields,
this script imports the corrected data into locales.json and visits.json.
"""

__author__ = "Luca Ostinelli"

from scripts.utils import (
    FLAGGED_SEGMENTS_JSON,
    LOCALES_JSON,
    VIDEOS_JSON,
    VISITS_JSON,
    load_json,
    save_json,
    setup_logging,
    today_str,
)
from scripts.schemas import Locale, Visit, generate_locale_id, generate_visit_id, timestamp_to_seconds
from scripts.geocode_locales import geocode_locale
from scripts.deduplicate_locales import find_duplicate

logger = setup_logging("flagged_review")


def process_reviewed_segments() -> tuple[int, int]:
    """
    Process flagged segments that have been reviewed by a human.
    
    For each segment where reviewed_by_human is True and required fields
    are filled in (locale_name, city), creates the corresponding
    locale and visit entries.
    
    Returns:
        (locales_created, visits_created) counts
    """
    flagged = load_json(FLAGGED_SEGMENTS_JSON)
    locales = load_json(LOCALES_JSON)
    visits = load_json(VISITS_JSON)
    videos = load_json(VIDEOS_JSON)

    existing_visit_ids = {v["visit_id"] for v in visits}

    locales_created = 0
    visits_created = 0
    updated_segments = []

    for segment in flagged:
        if not segment.get("reviewed_by_human", False):
            updated_segments.append(segment)
            continue

        locale_name = segment.get("locale_name")
        if not locale_name:
            logger.warning(f"Reviewed segment missing locale_name, skipping")
            updated_segments.append(segment)
            continue

        city = segment.get("city", "")
        video_id = segment.get("video_id", "")
        channel_id = segment.get("channel_id", "")

        geo = geocode_locale(locale_name, city)
        if not geo:
            logger.warning(f"Could not geocode reviewed locale: {locale_name} ({city})")
            updated_segments.append(segment)
            continue

        lat = geo["lat"]
        lon = geo["lon"]

        new_locale_data = {
            "name": locale_name,
            "locale_name": locale_name,
            "lat": lat,
            "lon": lon,
            "city": city,
            "aliases": [],
            "category": [],
        }

        duplicate = find_duplicate(new_locale_data, locales)
        if duplicate:
            locale_id = duplicate["locale_id"]
            logger.info(f"Reviewed locale matches existing: {locale_name} -> {locale_id}")
        else:
            locale_id = generate_locale_id(locale_name, lat, lon)
            locale_entry = {
                "locale_id": locale_id,
                "name": locale_name,
                "aliases": [],
                "address": geo.get("display_name", ""),
                "city": city,
                "lat": lat,
                "lon": lon,
                "category": [],
            }
            try:
                Locale(**locale_entry)
            except Exception as e:
                logger.warning(f"Locale validation failed for '{locale_name}': {e}")
                updated_segments.append(segment)
                continue
            locales.append(locale_entry)
            locales_created += 1
            logger.info(f"Created locale from review: {locale_name} ({locale_id})")

        start_ts = segment.get("timestamp_start", "0:00")
        start_seconds = timestamp_to_seconds(start_ts)
        visit_id = generate_visit_id(video_id, start_seconds)

        if visit_id not in existing_visit_ids:
            # Find video publish date
            publish_date = today_str()
            for v in videos:
                if v["video_id"] == video_id:
                    publish_date = v.get("publish_date", today_str())
                    break

            visit = {
                "visit_id": visit_id,
                "locale_id": locale_id,
                "video_id": video_id,
                "channel_id": channel_id,
                "timestamp_start": segment.get("timestamp_start", "0:00"),
                "timestamp_end": segment.get("timestamp_end", "0:00"),
                "youtube_url": f"https://youtu.be/{video_id}?t={start_seconds}",
                "rating": segment.get("rating"),
                "sentiment": "neutral",
                "rubrica": "",
                "llm_confidence": segment.get("llm_confidence", 0.0),
                "extraction_date": today_str(),
                "date": publish_date,
            }
            try:
                Visit(**visit)
            except Exception as e:
                logger.warning(f"Visit validation failed for {video_id}: {e}")
                updated_segments.append(segment)
                continue
            visits.append(visit)
            existing_visit_ids.add(visit_id)
            visits_created += 1

        # Mark segment with review date (keep in flagged for audit trail)
        segment["reviewed_date"] = today_str()
        updated_segments.append(segment)

    save_json(FLAGGED_SEGMENTS_JSON, updated_segments)
    save_json(LOCALES_JSON, locales)
    save_json(VISITS_JSON, visits)

    logger.info(f"Review complete: {locales_created} locales created, {visits_created} visits created")
    return locales_created, visits_created


if __name__ == "__main__":
    locales, visits = process_reviewed_segments()
    print(f"Processed review: {locales} new locales, {visits} new visits")
