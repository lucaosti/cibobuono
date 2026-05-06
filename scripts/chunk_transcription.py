"""
chunk_transcription.py — Split transcription into temporal chunks for LLM processing.

Groups Whisper segments into chunks of ~1-2 minutes for optimal LLM context.
Each chunk includes overlap to avoid missing locale mentions at boundaries.
"""

__author__ = "Luca Ostinelli"

from scripts.schemas import timestamp_to_seconds as _strict_ts_to_seconds
from scripts.utils import setup_logging

logger = setup_logging("chunker")

# Chunk settings
CHUNK_DURATION_SECONDS = 90   # Target chunk length: 1.5 minutes
OVERLAP_SECONDS = 15          # Overlap between chunks to avoid boundary issues


def seconds_to_timestamp(seconds: float) -> str:
    """Convert seconds to MM:SS or HH:MM:SS format."""
    total = int(seconds)
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def timestamp_to_seconds(timestamp: str) -> int:
    """Convert MM:SS or HH:MM:SS to total seconds.

    Returns 0 for malformed timestamps (graceful fallback for ASR output).
    """
    try:
        return _strict_ts_to_seconds(timestamp)
    except (ValueError, TypeError):
        return 0


def chunk_transcription(
    transcript: dict,
    chunk_duration: float = CHUNK_DURATION_SECONDS,
    overlap: float = OVERLAP_SECONDS,
) -> list[dict]:
    """
    Split a Whisper transcription into temporal chunks.
    
    Args:
        transcript: dict with 'video_id', 'segments' (from Whisper)
        chunk_duration: target chunk length in seconds
        overlap: overlap between consecutive chunks in seconds
    
    Returns:
        list of chunk dicts with:
        - video_id
        - chunk_index
        - start_time (seconds)
        - end_time (seconds)
        - start_timestamp (MM:SS or HH:MM:SS)
        - end_timestamp (MM:SS or HH:MM:SS)
        - text (concatenated segment texts)
    """
    video_id = transcript.get("video_id", "unknown")
    segments = transcript.get("segments", [])

    if not segments:
        logger.warning(f"No segments found for video {video_id}")
        return []

    total_duration = segments[-1].get("end", 0)
    if total_duration == 0:
        logger.warning(f"Video {video_id} has zero duration")
        return []

    chunks = []
    chunk_start = 0.0
    chunk_index = 0

    while chunk_start < total_duration:
        chunk_end = min(chunk_start + chunk_duration, total_duration)

        chunk_segments = []
        for seg in segments:
            seg_start = seg.get("start", 0)
            seg_end = seg.get("end", 0)
            if seg_end > chunk_start and seg_start < chunk_end:
                chunk_segments.append(seg)

        if chunk_segments:
            actual_start = chunk_segments[0].get("start", chunk_start)
            actual_end = chunk_segments[-1].get("end", chunk_end)
            text = " ".join(seg.get("text", "") for seg in chunk_segments).strip()

            # Preserve per-segment timestamps for precise venue-mention matching
            seg_timestamps = [
                (seg.get("start", 0.0), seg.get("text", "").strip())
                for seg in chunk_segments
                if seg.get("text", "").strip()
            ]

            chunks.append({
                "video_id": video_id,
                "chunk_index": chunk_index,
                "start_time": actual_start,
                "end_time": actual_end,
                "start_timestamp": seconds_to_timestamp(actual_start),
                "end_timestamp": seconds_to_timestamp(actual_end),
                "text": text,
                "segment_timestamps": seg_timestamps,
            })

            chunk_index += 1

        chunk_start = chunk_end - overlap
        if chunk_start >= total_duration:
            break
        # Avoid infinite loop on very short remaining segments
        if chunk_end >= total_duration:
            break

    logger.info(f"Video {video_id}: {len(chunks)} chunks from {len(segments)} segments ({total_duration:.0f}s)")
    return chunks


if __name__ == "__main__":
    import json
    import sys
    from scripts.utils import CACHE_DIR

    if len(sys.argv) < 2:
        print("Usage: python -m scripts.chunk_transcription <video_id>")
        sys.exit(1)

    vid = sys.argv[1]
    transcript_path = CACHE_DIR / f"{vid}_transcript.json"
    if not transcript_path.exists():
        print(f"Transcript not found: {transcript_path}")
        sys.exit(1)

    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = json.load(f)

    chunks = chunk_transcription(transcript)
    for c in chunks:
        print(f"[{c['start_timestamp']} - {c['end_timestamp']}] {c['text'][:80]}...")
