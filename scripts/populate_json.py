"""
populate_json.py — Populate visits.json and flagged_segments.json from extractions.

Creates visit entries with deterministic IDs.
Updates processed_videos.json for incrementality tracking.
Routes low-confidence extractions to flagged_segments.json for manual review.
"""

from scripts.utils import (
    FLAGGED_SEGMENTS_JSON,
    PROCESSED_VIDEOS_JSON,
    VISITS_JSON,
    load_json,
    save_json,
    setup_logging,
    today_str,
)
from scripts.schemas import (
    FlagReason,
    Visit,
    FlaggedSegment,
    ProcessedVideo,
    VideoStatus,
    generate_visit_id,
    timestamp_to_seconds,
)

logger = setup_logging("populate")


def create_visit(
    extraction: dict,
    locale_id: str,
    video_id: str,
    channel_id: str,
    publish_date: str,
) -> dict | None:
    """
    Create a visit entry from an LLM extraction.
    Returns validated visit dict or None on failure.
    """
    try:
        # Use precise mention_time (segment-level) when available,
        # falling back to chunk_start_seconds for older cached data.
        mention_seconds = extraction.get("mention_time")
        if mention_seconds is not None:
            start_seconds = int(mention_seconds)
            start_ts = extraction.get("mention_timestamp", "0:00")
        else:
            start_ts = extraction.get("chunk_start", "0:00")
            start_seconds = int(extraction.get("chunk_start_seconds", timestamp_to_seconds(start_ts)))

        visit_id = generate_visit_id(video_id, start_seconds)
        youtube_url = f"https://youtu.be/{video_id}?t={start_seconds}"

        visit_data = {
            "visit_id": visit_id,
            "locale_id": locale_id,
            "video_id": video_id,
            "channel_id": channel_id,
            "timestamp_start": start_ts,
            "timestamp_end": extraction.get("chunk_end", "0:00"),
            "youtube_url": youtube_url,
            "rating": extraction.get("rating"),
            "sentiment": extraction.get("sentiment", "neutral"),
            "rubrica": extraction.get("rubrica", ""),
            "notes": extraction.get("notes", ""),
            "llm_confidence": extraction.get("confidence", 0.5),
            "extraction_date": today_str(),
            "date": publish_date,
        }

        # Validate
        Visit(**visit_data)
        return visit_data

    except Exception as e:
        logger.error(f"Failed to create visit for {video_id}: {e}")
        return None


def create_flagged_segment(
    extraction: dict,
    video_id: str,
    channel_id: str,
    reason: str | None = None,
) -> dict | None:
    """
    Create a flagged segment entry from a low-confidence extraction.
    """
    mention_seconds = extraction.get("mention_time")
    start_seconds = int(mention_seconds) if mention_seconds is not None else int(extraction.get("chunk_start_seconds", 0))

    if reason is None:
        flag = extraction.get("_flag_reason", "")
        if flag == "geocoding_failed":
            reason = FlagReason.GEOCODING_FAILED.value
        elif flag == "osm_not_found":
            reason = FlagReason.OSM_NOT_FOUND.value
        elif not extraction.get("locale_name"):
            reason = FlagReason.MISSING_NAME.value
        elif not extraction.get("city") and not extraction.get("address"):
            reason = FlagReason.MISSING_ADDRESS.value
        else:
            reason = FlagReason.LOW_CONFIDENCE.value

    start_ts = extraction.get("mention_timestamp") or extraction.get("chunk_start", "0:00")

    segment = {
        "video_id": video_id,
        "channel_id": channel_id,
        "timestamp_start": start_ts,
        "timestamp_end": extraction.get("chunk_end", "0:00"),
        "youtube_url": f"https://youtu.be/{video_id}?t={start_seconds}",
        "reason": reason,
        "extracted_text": (extraction.get("text") or extraction.get("locale_name") or "")[:500],
        "llm_confidence": extraction.get("confidence", 0.0),
        "reviewed_by_human": False,
        "reviewed_date": None,
        "locale_name": extraction.get("locale_name"),
        "rating": extraction.get("rating"),
        "city": extraction.get("city"),
    }

    # Validate with pydantic model
    try:
        FlaggedSegment(**segment)
    except Exception as e:
        logger.error(f"FlaggedSegment validation failed for {video_id}: {e}")
        return None

    return segment


def populate_visits(
    locale_mapping: list[dict],
    video_id: str,
    channel_id: str,
    publish_date: str,
) -> list[dict]:
    """
    Create visit entries from locale mapping (output of deduplication).
    
    Returns:
        List of newly created visit dicts.
    """
    existing_visits = load_json(VISITS_JSON)
    existing_visit_ids = {v["visit_id"] for v in existing_visits}

    new_visits = []

    for mapping in locale_mapping:
        extraction = mapping["extraction"]
        locale_id = mapping["locale_id"]

        visit = create_visit(extraction, locale_id, video_id, channel_id, publish_date)
        if visit and visit["visit_id"] not in existing_visit_ids:
            new_visits.append(visit)
            existing_visit_ids.add(visit["visit_id"])

    # Save visits
    if new_visits:
        all_visits = existing_visits + new_visits
        save_json(VISITS_JSON, all_visits)
        logger.info(f"Added {len(new_visits)} visits ({len(all_visits)} total)")

    return new_visits


def populate_flagged(
    flagged_extractions: list[dict],
    video_id: str,
    channel_id: str,
) -> list[dict]:
    """
    Add flagged segments from low-confidence extractions.
    """
    existing_flagged = load_json(FLAGGED_SEGMENTS_JSON)

    new_flagged = []
    for extraction in flagged_extractions:
        segment = create_flagged_segment(extraction, video_id, channel_id)
        if segment is not None:
            new_flagged.append(segment)

    if new_flagged:
        all_flagged = existing_flagged + new_flagged
        save_json(FLAGGED_SEGMENTS_JSON, all_flagged)
        logger.info(f"Added {len(new_flagged)} flagged segments ({len(all_flagged)} total)")

    return new_flagged


def update_processed_videos(
    video_id: str,
    channel_id: str,
    status: VideoStatus,
    visits_count: int = 0,
    flagged_count: int = 0,
) -> None:
    """Mark a video as processed."""
    processed = load_json(PROCESSED_VIDEOS_JSON)

    # Check if already processed (idempotent)
    for p in processed:
        if p["video_id"] == video_id:
            logger.info(f"Video {video_id} already in processed list")
            return

    entry = {
        "video_id": video_id,
        "channel_id": channel_id,
        "processed_date": today_str(),
        "status": status.value,
        "visits_extracted": visits_count,
        "flagged_segments": flagged_count,
    }

    try:
        ProcessedVideo(**entry)
    except Exception as e:
        logger.error(f"Validation failed for processed video {video_id}: {e}")
        return

    processed.append(entry)
    save_json(PROCESSED_VIDEOS_JSON, processed)
    logger.info(f"Marked video {video_id} as {status.value}")


if __name__ == "__main__":
    print("This module is used by run_pipeline.py")
