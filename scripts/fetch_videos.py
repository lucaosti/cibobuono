"""
fetch_videos.py — Fetch and catalog videos from channels using yt-dlp.

Two-phase flow:
  Phase 1 (catalog): Fetch ALL video metadata from a channel. Insert them
      into videos.json with status="pending". No audio download.
      Recipe videos are detected by title keywords and moved to
      skipped_videos.json with status="skipped".
  Phase 2 (download): Download audio for a single video (called by pipeline).
      Videos are processed newest-first for better subtitle quality.

This ensures videos.json is always a complete inventory of every video in every
channel, with a clear status field showing pipeline progress.
"""

from __future__ import annotations

__author__ = "Luca Ostinelli"


import json
import re
import subprocess
from pathlib import Path

from scripts.utils import (
    CACHE_DIR,
    CHANNELS_JSON,
    SKIPPED_VIDEOS_JSON,
    VIDEOS_JSON,
    ensure_dirs,
    load_json,
    save_json,
    setup_logging,
    today_str,
    YOUTUBE_EXTRACTOR_ARGS,
    yt_dlp_command,
)
from scripts.schemas import SkippedVideo, Video, VideoStatus

logger = setup_logging("fetch_videos")

# ---------------------------------------------------------------------------
# Retry helpers for yt-dlp subprocess calls
# ---------------------------------------------------------------------------

import time as _time

_YTDLP_RETRIES = 2
_YTDLP_RETRY_BASE_DELAY = 5.0  # seconds; doubles on each retry


def _run_ytdlp_with_retry(cmd: list, *, timeout: int) -> subprocess.CompletedProcess:
    """Run a yt-dlp command with simple exponential-backoff retry on transient failures.

    Retries on returncode != 0 (rate-limit / transient network errors) up to
    ``_YTDLP_RETRIES`` times.  ``TimeoutExpired`` is NOT retried — the caller
    handles it.
    """
    delay = _YTDLP_RETRY_BASE_DELAY
    for attempt in range(_YTDLP_RETRIES + 1):
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode == 0:
            return result
        if attempt < _YTDLP_RETRIES:
            logger.warning(
                "yt-dlp attempt %d/%d failed (rc=%d); retrying in %.0fs",
                attempt + 1, _YTDLP_RETRIES + 1, result.returncode, delay,
            )
            _time.sleep(delay)
            delay *= 2
    return result  # return last failure for the caller to handle


# ---------------------------------------------------------------------------
# Recipe detection
# ---------------------------------------------------------------------------

# Fast keyword pre-filter: recipe/cooking signals (short, precise list).
RECIPE_KEYWORDS = [
    "ricetta", "ricette", "recipe", "recipes",
    "cucino", "cuciniamo", "preparo", "prepariamo", "preparazione",
    "come fare", "come si fa", "come si prepara",
    "faccio a casa", "a casa mia", "cucina casalinga",
    "tutorial cucina", "cooking tutorial",
    "impasto", "lievitazione",
]

# Keyword list for non-food video detection.  Add new terms here to extend coverage.
NON_FOOD_KEYWORDS = [
    # Sport / fitness / combat
    "boxe", "boxing", "allenamento", "palestra", "workout", "fitness",
    "saltare la corda", "jump rope", "esercizi", "addominali", "pesi",
    "running", "calcio", "football", "basket", "tennis",
    "sparring", "kickboxing", "mma", "arti marziali",
    "pugile", "pugili", "pugilato", "bendaggi", "sacco da boxe",
    "guardia del pugilato", "resistenze",
    "botte", "pugni", "rissa", "aggressione",
    "difesa personale", "self defense", "self-defense",
    # Gaming
    "gameplay", "gaming", "playstation", "xbox", "fortnite", "minecraft",
    "gta", "fifa", "videogame", "videogioco",
    # Tech / tutorials (non-food)
    "tutorial photoshop", "tutorial premiere", "montaggio video",
    "come editare", "setup tour", "unboxing",
    # Lifestyle / generic vlogs without food
    "haul", "ask me anything", "tag challenge",
    "morning routine", "night routine",
    # Music
    "freestyle", "dissing", "rap battle", "nuovo singolo", "feat.",
    # Other clearly non-food
    "prank", "scherzo", "challenge estrema",
]


def detect_recipe_video(title: str) -> tuple[bool, str]:
    """Return (True, reason) if the title suggests a recipe/cooking video."""
    title_lower = title.lower().strip()
    for kw in RECIPE_KEYWORDS:
        if kw in title_lower:
            return True, f'Recipe keyword in title: "{kw}"'
    return False, ""


def detect_non_food_video(title: str, description: str = "") -> tuple[bool, str]:
    """Return (True, reason) if the title/description clearly indicates a non-food video.

    Uses word-boundary keyword matching to avoid false positives like 'mma'
    matching inside 'MAMMA' or 'TOMMASO'.  Conservative by design: ambiguous
    titles stay pending rather than being skipped.
    """
    for text_label, text in [("title", title), ("description", description)]:
        if not text:
            continue
        text_lower = text.lower().strip()
        for kw in NON_FOOD_KEYWORDS:
            if re.search(r"\b" + re.escape(kw) + r"\b", text_lower):
                return True, f'Non-food video ({text_label}): "{kw}"'
    return False, ""


SHORTS_MAX_DURATION = 60  # seconds


def detect_short_video(url: str, duration: float | int | None) -> tuple[bool, str]:
    """Return (True, reason) if the video is a YouTube Short."""
    if "/shorts/" in url:
        return True, "URL contains /shorts/"
    if duration is not None and 0 < duration <= SHORTS_MAX_DURATION:
        return True, f"Duration {int(duration)}s (≤{SHORTS_MAX_DURATION}s)"
    return False, ""


# ---------------------------------------------------------------------------
# yt-dlp helpers
# ---------------------------------------------------------------------------

def fetch_video_list(channel_url: str) -> list[dict]:
    """
    Fetch the COMPLETE list of videos from a channel via yt-dlp --flat-playlist.
    Returns a list of dicts (video_id, title, url, upload_date) in
    YouTube's native order (newest-first).
    """
    try:
        result = _run_ytdlp_with_retry(
            [
                *yt_dlp_command(),
                "--flat-playlist",
                "--dump-json",
                *YOUTUBE_EXTRACTOR_ARGS,
                channel_url,
            ],
            timeout=300,
        )

        if result.returncode != 0:
            logger.error(f"yt-dlp failed for {channel_url}: {result.stderr[:500]}")
            return []

        videos = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                video_id = data.get("id", "")
                title = data.get("title", "")
                url = data.get("url", data.get("webpage_url", f"https://youtu.be/{video_id}"))
                upload_date = data.get("upload_date", "")

                if upload_date and len(upload_date) == 8:
                    upload_date = f"{upload_date[:4]}-{upload_date[4:6]}-{upload_date[6:8]}"

                if video_id and title:
                    duration = data.get("duration")  # seconds (float or int)
                    videos.append({
                        "video_id": video_id,
                        "title": title,
                        "url": url if url.startswith("http") else f"https://youtu.be/{video_id}",
                        "upload_date": upload_date,
                        "duration": duration,
                    })
            except json.JSONDecodeError as exc:
                logger.debug("Skipping malformed yt-dlp JSON line for %s: %s", channel_url, exc)
                continue

        return videos

    except subprocess.TimeoutExpired:
        logger.error(f"Timeout fetching video list for {channel_url}")
        return []
    except Exception as e:
        logger.error(f"Error fetching video list for {channel_url}: {e}")
        return []


def fetch_video_upload_date(video_id: str) -> str:
    """
    Fetch the real upload_date for a single video via yt-dlp --dump-json.
    Returns YYYY-MM-DD string, or empty string on failure.
    """
    try:
        result = subprocess.run(
            [
                *yt_dlp_command(),
                "--dump-json",
                "--no-download",
                "--no-playlist",
                *YOUTUBE_EXTRACTOR_ARGS,
                f"https://youtu.be/{video_id}",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0 and result.stdout.strip():
            data = json.loads(result.stdout.strip().split("\n")[0])
            upload_date = data.get("upload_date", "")
            if upload_date and len(upload_date) == 8:
                return f"{upload_date[:4]}-{upload_date[4:6]}-{upload_date[6:8]}"
    except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception) as e:
        logger.warning(f"Could not fetch upload_date for {video_id}: {e}")
    return ""


def fetch_video_metadata(video_id: str) -> dict:
    """
    Fetch structured metadata (title, description, chapters) via yt-dlp JSON dump.
    Cached as {video_id}_metadata.json. Also refreshes description.txt when present.
    Returns a dict with keys: title, description, chapters (list of {start_time, title}).
    """
    ensure_dirs()
    meta_path = CACHE_DIR / f"{video_id}_metadata.json"
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass

    slim: dict = {"title": "", "description": "", "chapters": []}
    try:
        result = subprocess.run(
            [
                *yt_dlp_command(),
                "--dump-json",
                "--no-download",
                "--no-playlist",
                *YOUTUBE_EXTRACTOR_ARGS,
                f"https://youtu.be/{video_id}",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0 or not result.stdout.strip():
            logger.warning(
                f"yt-dlp metadata failed for {video_id}: "
                f"{(result.stderr or '')[:300]}"
            )
            return slim

        raw = json.loads(result.stdout.strip().split("\n", 1)[0])
        slim["title"] = (raw.get("title") or "").strip()
        slim["description"] = (raw.get("description") or "").strip()
        for ch in raw.get("chapters") or []:
            if not isinstance(ch, dict):
                continue
            slim["chapters"].append(
                {
                    "start_time": ch.get("start_time"),
                    "title": (ch.get("title") or "").strip(),
                }
            )
        meta_path.write_text(
            json.dumps(slim, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        desc_path = CACHE_DIR / f"{video_id}_description.txt"
        if slim["description"]:
            desc_path.write_text(slim["description"], encoding="utf-8")
        logger.info(
            f"Fetched metadata for {video_id}: "
            f"{len(slim['description'])} chars, {len(slim['chapters'])} chapters"
        )
    except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception) as e:
        logger.warning(f"Could not fetch metadata for {video_id}: {e}")

    return slim


def fetch_video_description(video_id: str) -> str:
    """
    Fetch the full video description from YouTube via yt-dlp.

    Returns the longest description found across cache files; re-fetches via
    dump-json when cached copies look truncated (common cause of missing venues).
    """
    meta_path = CACHE_DIR / f"{video_id}_metadata.json"
    desc_path = CACHE_DIR / f"{video_id}_description.txt"

    candidates: list[str] = []

    if desc_path.exists():
        try:
            candidates.append(desc_path.read_text(encoding="utf-8").strip())
        except OSError:
            pass

    if meta_path.exists():
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
            candidates.append((data.get("description") or "").strip())
        except json.JSONDecodeError:
            pass

    best = max(candidates, key=len, default="")
    # Descriptions under ~80 chars are almost always truncated cache artifacts.
    if len(best) >= 80:
        return best

    # Force fresh metadata download when cache looks truncated.
    if meta_path.exists():
        try:
            meta_path.unlink()
        except OSError:
            pass
    meta = fetch_video_metadata(video_id)
    desc = (meta.get("description") or "").strip()
    if desc:
        desc_path.write_text(desc, encoding="utf-8")
        return desc

    return best


def fetch_video_comments(video_id: str, max_comments: int = 40) -> list[dict]:
    """Fetch top YouTube comments via yt-dlp (cached). Used for venue hints."""
    cache_path = CACHE_DIR / f"{video_id}_comments.json"
    if cache_path.exists():
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            return data.get("comments") or []
        except json.JSONDecodeError:
            pass

    comments: list[dict] = []
    try:
        result = subprocess.run(
            [
                *yt_dlp_command(),
                "--skip-download",
                "--dump-single-json",
                "--quiet",
                "--no-warnings",
                "--extractor-args",
                f"youtube:max-comments={max_comments}",
                *YOUTUBE_EXTRACTOR_ARGS,
                f"https://youtu.be/{video_id}",
            ],
            capture_output=True,
            text=True,
            timeout=90,
        )
        if result.returncode != 0 or not result.stdout.strip():
            logger.debug(f"No comments for {video_id}")
            cache_path.write_text(json.dumps({"comments": []}, indent=2), encoding="utf-8")
            return []

        raw = json.loads(result.stdout.strip().split("\n", 1)[0])
        for c in raw.get("comments") or []:
            if not isinstance(c, dict):
                continue
            text = (c.get("text") or "").strip()
            if len(text) < 4:
                continue
            comments.append(
                {
                    "text": text,
                    "author": (c.get("author") or "").strip(),
                    "like_count": int(c.get("like_count") or 0),
                }
            )
        cache_path.write_text(
            json.dumps({"comments": comments}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(f"Fetched {len(comments)} comments for {video_id}")
    except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception) as e:
        logger.warning(f"Could not fetch comments for {video_id}: {e}")

    return comments


def download_audio(video_id: str, video_url: str) -> Path | None:
    """Download audio as WAV for Whisper. Returns path or None on failure."""
    ensure_dirs()
    output_path = CACHE_DIR / f"{video_id}.wav"

    if output_path.exists():
        logger.info(f"Audio already cached: {video_id}")
        return output_path

    try:
        url = video_url if video_url.startswith("http") else f"https://youtu.be/{video_id}"
        result = _run_ytdlp_with_retry(
            [
                *yt_dlp_command(),
                "-x",
                "--audio-format", "wav",
                "--audio-quality", "0",
                "-o", str(output_path),
                "--no-playlist",
                *YOUTUBE_EXTRACTOR_ARGS,
                url,
            ],
            timeout=600,
        )

        if result.returncode != 0:
            logger.error(f"Failed to download audio for {video_id}: {result.stderr[:500]}")
            return None

        if not output_path.exists():
            for f in CACHE_DIR.glob(f"{video_id}.*"):
                if f.suffix in {".wav", ".mp3", ".m4a", ".webm"}:
                    return f

        return output_path if output_path.exists() else None

    except subprocess.TimeoutExpired:
        logger.error(f"Timeout downloading audio for {video_id}")
        return None
    except Exception as e:
        logger.error(f"Error downloading audio for {video_id}: {e}")
        return None


def download_video(video_id: str, video_url: str, *, max_height: int = 480) -> Path | None:
    """Download a low-resolution video for Perceptor frame sampling.

    Capped at *max_height* to bound bandwidth and disk (~0.5 GB/h at 480p);
    the caller deletes the file right after frame extraction. Returns the
    cached mp4 path or None on failure.
    """
    ensure_dirs()
    output_path = CACHE_DIR / f"{video_id}_video.mp4"

    if output_path.exists():
        logger.info(f"Video already cached: {video_id}")
        return output_path

    try:
        url = video_url if video_url.startswith("http") else f"https://youtu.be/{video_id}"
        result = _run_ytdlp_with_retry(
            [
                *yt_dlp_command(),
                "-f", f"bv*[height<={max_height}]+ba/b[height<={max_height}]/b",
                "--merge-output-format", "mp4",
                "-o", str(output_path),
                "--no-playlist",
                *YOUTUBE_EXTRACTOR_ARGS,
                url,
            ],
            timeout=900,
        )

        if result.returncode != 0:
            logger.error(f"Failed to download video for {video_id}: {result.stderr[:500]}")
            return None
        return output_path if output_path.exists() else None

    except subprocess.TimeoutExpired:
        logger.error(f"Timeout downloading video for {video_id}")
        return None
    except Exception as e:
        logger.error(f"Error downloading video for {video_id}: {e}")
        return None


# ---------------------------------------------------------------------------
# Phase 1: Catalog all videos (no download)
# ---------------------------------------------------------------------------

def catalog_channel_videos() -> int:
    """
    For every channel in channels.json, fetch the complete video list and
    insert ALL unknown videos into videos.json with status="pending".
    Recipe videos go to skipped_videos.json with status="skipped".

    Returns the number of newly cataloged videos (pending + skipped).
    """
    channels = load_json(CHANNELS_JSON)
    if not channels:
        logger.info("No channels found in channels.json")
        return 0

    existing_videos = load_json(VIDEOS_JSON)
    existing_ids = {v["video_id"] for v in existing_videos}

    skipped_videos = load_json(SKIPPED_VIDEOS_JSON)
    skipped_ids = {v["video_id"] for v in skipped_videos}

    new_pending: list[dict] = []
    new_skipped: list[dict] = []

    for channel in channels:
        channel_id = channel["channel_id"]
        channel_url = channel["url"]
        logger.info(f"Cataloging videos for: {channel['name']} ({channel_id})")

        all_vids = fetch_video_list(channel_url)
        logger.info(f"  Total videos on channel: {len(all_vids)}")

        for v in all_vids:
            vid = v["video_id"]
            if vid in existing_ids or vid in skipped_ids:
                continue

            # ---- Shorts check ----
            is_short, short_reason = detect_short_video(v["url"], v.get("duration"))
            if is_short:
                entry = {
                    "video_id": vid,
                    "channel_id": channel_id,
                    "title": v["title"],
                    "url": v["url"],
                    "reason": short_reason,
                    "skipped_date": today_str(),
                }
                try:
                    SkippedVideo(**entry)
                except Exception as e:
                    logger.error(f"  Validation failed for skipped video {vid}: {e}")
                    continue
                new_skipped.append(entry)
                skipped_ids.add(vid)
                logger.info(f"  Skipped (short): {v['title'][:60]} — {short_reason}")
                continue

            # ---- Recipe check ----
            is_recipe, reason = detect_recipe_video(v["title"])
            if is_recipe:
                entry = {
                    "video_id": vid,
                    "channel_id": channel_id,
                    "title": v["title"],
                    "url": v["url"],
                    "reason": reason,
                    "skipped_date": today_str(),
                }
                try:
                    SkippedVideo(**entry)
                except Exception as e:
                    logger.error(f"  Validation failed for skipped video {vid}: {e}")
                    continue
                new_skipped.append(entry)
                skipped_ids.add(vid)
                logger.info(f"  Skipped (recipe): {v['title'][:60]} — {reason}")
                continue

            # ---- Non-food content check ----
            is_non_food, nf_reason = detect_non_food_video(v["title"])
            if is_non_food:
                entry = {
                    "video_id": vid,
                    "channel_id": channel_id,
                    "title": v["title"],
                    "url": v["url"],
                    "reason": nf_reason,
                    "skipped_date": today_str(),
                }
                try:
                    SkippedVideo(**entry)
                except Exception as e:
                    logger.error(f"  Validation failed for skipped video {vid}: {e}")
                    continue
                new_skipped.append(entry)
                skipped_ids.add(vid)
                logger.info(f"  Skipped (non-food): {v['title'][:60]} — {nf_reason}")
                continue

            # ---- Use upload date from flat-playlist (fast), resolve later if empty ----
            publish_date = v.get("upload_date", "")

            entry = {
                "video_id": vid,
                "channel_id": channel_id,
                "title": v["title"],
                "url": v["url"],
                "publish_date": publish_date,
                "processed_date": "",
                "status": VideoStatus.PENDING.value,
            }
            try:
                Video(**entry)
            except Exception as e:
                logger.error(f"  Validation failed for video {vid}: {e}")
                continue

            new_pending.append(entry)
            existing_ids.add(vid)

    # Persist
    if new_pending:
        all_videos = existing_videos + new_pending
        save_json(VIDEOS_JSON, all_videos)
        logger.info(f"Cataloged {len(new_pending)} new pending videos (total {len(all_videos)})")

    if new_skipped:
        all_skipped = skipped_videos + new_skipped
        save_json(SKIPPED_VIDEOS_JSON, all_skipped)
        logger.info(f"Skipped {len(new_skipped)} recipe videos (total {len(all_skipped)})")

    return len(new_pending) + len(new_skipped)


# ---------------------------------------------------------------------------
# Helpers used by the pipeline
# ---------------------------------------------------------------------------

def get_pending_videos() -> list[dict]:
    """Return videos with status='pending', ordered newest first (by publish_date).

    Newer videos have better YouTube ASR subtitles, so processing them first
    yields higher-quality extractions.  Videos with missing dates sort last.
    """
    all_vids = load_json(VIDEOS_JSON)
    pending = [v for v in all_vids if v.get("status") == VideoStatus.PENDING.value]
    pending.sort(key=lambda v: v.get("publish_date") or "", reverse=True)
    return pending


def update_video_status(
    video_id: str,
    status: VideoStatus,
    publish_date: str = "",
    *,
    _videos_cache: dict[str, dict] | None = None,
) -> None:
    """Update a video's status (and publish_date if provided) in videos.json.

    Pass ``_videos_cache`` (a ``{video_id: video_dict}`` mapping already loaded
    from ``VIDEOS_JSON``) to avoid the full read-modify-write cycle when updating
    many videos in a loop.  The caller is responsible for flushing the cache to
    disk via :func:`flush_videos_cache` when done.
    """
    if _videos_cache is not None:
        v = _videos_cache.get(video_id)
        if v is not None:
            v["status"] = status.value
            if status == VideoStatus.PROCESSED:
                v["processed_date"] = today_str()
            if publish_date and not v.get("publish_date"):
                v["publish_date"] = publish_date
        return

    all_vids = load_json(VIDEOS_JSON)
    for v in all_vids:
        if v["video_id"] == video_id:
            v["status"] = status.value
            if status == VideoStatus.PROCESSED:
                v["processed_date"] = today_str()
            if publish_date and not v.get("publish_date"):
                v["publish_date"] = publish_date
            break
    save_json(VIDEOS_JSON, all_vids)


def load_videos_cache() -> dict[str, dict]:
    """Load videos.json into an in-memory ``{video_id: video_dict}`` mapping.

    Use together with :func:`update_video_status` (passing ``_videos_cache``) and
    :func:`flush_videos_cache` to batch-update many videos with a single read and
    a single write instead of one read-modify-write per video.
    """
    return {v["video_id"]: v for v in load_json(VIDEOS_JSON)}


def flush_videos_cache(cache: dict[str, dict]) -> None:
    """Write an in-memory videos cache back to ``VIDEOS_JSON``."""
    save_json(VIDEOS_JSON, list(cache.values()))


if __name__ == "__main__":
    catalog_channel_videos()
