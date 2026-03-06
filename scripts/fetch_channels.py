"""
fetch_channels.py — Extract channel info from channels_input.txt using yt-dlp.

Reads YouTube channel URLs, extracts channel_id, name, and description
without any paid API. Uses yt-dlp to scrape channel metadata.
Infers rubriche from video titles using pattern matching.
"""

import re
import subprocess
import json
from collections import Counter

from scripts.utils import (
    CHANNELS_INPUT,
    CHANNELS_JSON,
    load_json,
    save_json,
    setup_logging,
)
from scripts.schemas import Channel

logger = setup_logging("fetch_channels")


def read_channel_urls(path=CHANNELS_INPUT) -> list[str]:
    """Read channel URLs from input file, skipping comments and blanks."""
    if not path.exists():
        logger.error(f"channels_input.txt not found at {path}")
        return []
    urls = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                urls.append(line)
    return urls


def extract_channel_id_from_url(url: str) -> str:
    """
    Derive a deterministic channel_id from the URL.
    Handles formats like:
    - https://www.youtube.com/@handle
    - https://www.youtube.com/channel/UC...
    - https://www.youtube.com/c/ChannelName
    """
    # @handle format
    match = re.search(r"youtube\.com/@([^/?&]+)", url)
    if match:
        return match.group(1).lower()

    # /channel/UCxxx format
    match = re.search(r"youtube\.com/channel/([^/?&]+)", url)
    if match:
        return match.group(1)

    # /c/name format
    match = re.search(r"youtube\.com/c/([^/?&]+)", url)
    if match:
        return match.group(1).lower()

    # Fallback: use last path segment
    match = re.search(r"youtube\.com/([^/?&]+)", url)
    if match:
        return match.group(1).lower()

    return url.strip("/").split("/")[-1].lower()


def fetch_channel_metadata(url: str) -> dict | None:
    """
    Use yt-dlp to fetch channel metadata (name, description)
    by extracting info from the channel's main page.
    """
    try:
        result = subprocess.run(
            [
                "yt-dlp",
                "--dump-json",
                "--playlist-items", "0",
                "--flat-playlist",
                "--extractor-args", "youtube:lang=it",
                url,
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode != 0:
            # Try alternative: get first video's channel info
            result = subprocess.run(
                [
                    "yt-dlp",
                    "--dump-json",
                    "--playlist-items", "1",
                    "--extractor-args", "youtube:lang=it",
                    url,
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )

        if result.returncode == 0 and result.stdout.strip():
            data = json.loads(result.stdout.strip().split('\n')[0])
            return {
                "name": data.get("channel", data.get("uploader", "")),
                "description": data.get("description", ""),
            }
    except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception) as e:
        logger.warning(f"Failed to fetch metadata for {url}: {e}")

    return None


def infer_rubriche_from_titles(url: str, max_videos: int = 50) -> list[str]:
    """
    Infer rubrica names from video titles by looking for repeated patterns.
    Many food YouTubers use patterns like "RUBRICA NAME - ..." or "[RUBRICA] ...".
    """
    try:
        result = subprocess.run(
            [
                "yt-dlp",
                "--flat-playlist",
                "--dump-json",
                "--playlist-items", f"1:{max_videos}",
                "--extractor-args", "youtube:lang=it",
                url,
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode != 0:
            return []

        titles = []
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                try:
                    data = json.loads(line)
                    title = data.get("title", "")
                    if title:
                        titles.append(title)
                except json.JSONDecodeError:
                    continue

        # Extract patterns: "PREFIX - rest" or "[PREFIX] rest" or "PREFIX: rest"
        prefixes = []
        for title in titles:
            # "RUBRICA - ..."
            match = re.match(r"^([A-Z\s]{3,30})\s*[-–—|]\s+", title)
            if match:
                prefixes.append(match.group(1).strip())
                continue
            # "[RUBRICA] ..."
            match = re.match(r"^\[([^\]]+)\]", title)
            if match:
                prefixes.append(match.group(1).strip())
                continue
            # "RUBRICA: ..."
            match = re.match(r"^([A-Za-z\s]{3,30}):\s+", title)
            if match:
                prefixes.append(match.group(1).strip())

        # Keep prefixes that appear at least 2 times
        counter = Counter(prefixes)
        rubriche = [name for name, count in counter.most_common() if count >= 2]
        return rubriche

    except (subprocess.TimeoutExpired, Exception) as e:
        logger.warning(f"Failed to infer rubriche for {url}: {e}")
        return []


def fetch_channels() -> list[dict]:
    """
    Main function: read URLs, fetch metadata, build channels.json.
    Incremental: only adds new channels, doesn't overwrite existing ones.
    """
    urls = read_channel_urls()
    if not urls:
        logger.info("No channel URLs found in channels_input.txt")
        return []

    existing = load_json(CHANNELS_JSON)
    existing_ids = {ch["channel_id"] for ch in existing}
    new_channels = []

    for url in urls:
        channel_id = extract_channel_id_from_url(url)
        if channel_id in existing_ids:
            logger.info(f"Channel already exists: {channel_id}")
            continue

        logger.info(f"Fetching metadata for: {url}")
        metadata = fetch_channel_metadata(url)
        name = metadata["name"] if metadata else channel_id
        description = metadata.get("description", "") if metadata else ""

        logger.info(f"Inferring rubriche for: {channel_id}")
        rubriche = infer_rubriche_from_titles(url)

        channel = {
            "channel_id": channel_id,
            "name": name,
            "url": url,
            "description": description,
            "rubriche": rubriche,
        }

        # Validate
        try:
            Channel(**channel)
        except Exception as e:
            logger.error(f"Validation failed for channel {channel_id}: {e}")
            continue

        new_channels.append(channel)
        existing_ids.add(channel_id)
        logger.info(f"Added channel: {name} ({channel_id}) with rubriche: {rubriche}")

    if new_channels:
        all_channels = existing + new_channels
        save_json(CHANNELS_JSON, all_channels)
        logger.info(f"Saved {len(new_channels)} new channels ({len(all_channels)} total)")

    return new_channels


if __name__ == "__main__":
    fetch_channels()
