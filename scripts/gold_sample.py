"""
gold_sample.py — Pick a stratified sample of long-form videos and fetch their metadata.

Builds the input file for :mod:`scripts.teacher_label` (same shape as the
16-video ``sample_meta.json``): a JSON list of
``{video_id, channel, title, duration, description, chapters, upload_date}``.

Steps:

1. Pool: the catalogue (``data/videos.json`` shape) intersected with each
   channel's *Videos* tab listed by yt-dlp. The Videos tab excludes Shorts,
   so the intersection is the long-form part of the catalogue; a duration
   floor guards against mislabelled entries.
2. Publish year per video without one request per video: the Videos tab is
   sorted newest first, so the position of every year boundary is found by
   binary search over the list (about ten metadata requests per boundary).
   YouTube rate-limits per-video metadata requests (HTTP 429 after a few
   hundred in a few minutes), which rules out fetching the whole catalogue.
3. Selection: ``--per-channel`` videos per channel, allocated across publish
   years proportionally to the pool (largest-remainder rounding, at least one
   per year while the budget allows), sampled with a fixed seed. Full
   metadata (title, description, chapters, duration, exact upload_date) is
   fetched only for the picks; a pick whose metadata cannot be fetched
   (members-only video, transient error) is replaced by the next candidate of
   the same stratum. Ids in ``--exclude`` (e.g. the 16-video sample) are never
   picked.

All per-video metadata goes through one JSONL cache, so the script resumes
after interruptions and never asks YouTube twice for the same video.

CLI:
    python -m scripts.gold_sample --catalog videos.json --channels channels_input.txt \
        --work runs/gold200 --out runs/gold200/gold_meta.json --per-channel 50
"""

from __future__ import annotations

__author__ = "Luca Ostinelli"

import argparse
import json
import random
import re
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

from scripts.utils import YOUTUBE_EXTRACTOR_ARGS, setup_logging, yt_dlp_command

logger = setup_logging("gold_sample")

MIN_LONGFORM_SECONDS = 90
META_FIELDS = "%(.{id,title,description,chapters,duration,upload_date,uploader_id})j"
FLAT_FIELDS = "%(.{id,title,duration})j"
REQUEST_PAUSE_SECONDS = 1.5
RATE_LIMIT_BACKOFF_SECONDS = 90


def _channel_handle(url: str) -> str:
    """'https://www.youtube.com/@Name' or '/c/Name' -> 'Name'."""
    m = re.search(r"youtube\.com/(?:@|c/|user/)?([^/?#]+)", url.strip())
    return m.group(1) if m else url.strip()


def _channel_key(url: str) -> str:
    """Catalogue channel_id convention: lower-case handle."""
    return _channel_handle(url).lower()


def _videos_tab_url(url: str) -> str:
    handle = _channel_handle(url)
    if "/c/" in url or "/user/" in url:
        base = url.strip().rstrip("/")
    else:
        base = f"https://www.youtube.com/@{handle}"
    return f"{base}/videos"


def list_videos_tab(channel_url: str, cache_dir: Path) -> list[dict]:
    """Flat-list the channel's Videos tab (no Shorts), newest first. Cached per channel."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"flat_{_channel_key(channel_url)}.jsonl"
    if not cache.exists() or cache.stat().st_size == 0:
        logger.info("Listing Videos tab for %s", channel_url)
        result = subprocess.run(
            [*yt_dlp_command(), "--flat-playlist", *YOUTUBE_EXTRACTOR_ARGS,
             "--print", FLAT_FIELDS, _videos_tab_url(channel_url)],
            capture_output=True, text=True, timeout=900,
        )
        if result.returncode != 0:
            raise RuntimeError(f"yt-dlp flat listing failed for {channel_url}: {result.stderr[-500:]}")
        cache.write_text(result.stdout, encoding="utf-8")
    rows = [json.loads(l) for l in cache.read_text(encoding="utf-8").splitlines() if l.strip()]
    seen: set[str] = set()
    ordered = []
    for row in rows:
        if row["id"] not in seen:
            seen.add(row["id"])
            ordered.append(row)
    return ordered


class MetaStore:
    """Per-video yt-dlp metadata with a JSONL cache and polite pacing.

    Failures are cached too (``{"id": ..., "error": ...}``) so members-only
    videos are not retried; rate-limit errors are not cached and trigger a
    backoff instead.
    """

    def __init__(self, cache_path: Path):
        self.cache_path = cache_path
        self.rows: dict[str, dict] = {}
        self.requests = 0
        if cache_path.exists():
            for line in cache_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    self.rows[row["id"]] = row
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(cache_path, "a", encoding="utf-8")

    def _store(self, row: dict) -> None:
        self.rows[row["id"]] = row
        self._fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._fh.flush()

    def get(self, video_id: str) -> dict | None:
        """Metadata dict, or None when the video cannot be fetched."""
        if video_id in self.rows:
            row = self.rows[video_id]
            return None if row.get("error") else row
        for attempt in range(3):
            time.sleep(REQUEST_PAUSE_SECONDS)
            self.requests += 1
            result = subprocess.run(
                [*yt_dlp_command(), "--skip-download", "--no-playlist", *YOUTUBE_EXTRACTOR_ARGS,
                 "--print", META_FIELDS, f"https://youtu.be/{video_id}"],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode == 0 and result.stdout.strip():
                try:
                    row = json.loads(result.stdout.strip().splitlines()[-1])
                except json.JSONDecodeError:
                    row = None
                if row and row.get("upload_date"):
                    self._store(row)
                    return row
            err = result.stderr.strip()
            if "429" in err or "not a bot" in err or "non essere un bot" in err:
                logger.warning("%s: rate limited, backing off %ds (attempt %d)",
                               video_id, RATE_LIMIT_BACKOFF_SECONDS, attempt + 1)
                time.sleep(RATE_LIMIT_BACKOFF_SECONDS)
                continue
            logger.warning("%s: metadata unavailable: %s", video_id, err[-200:])
            self._store({"id": video_id, "error": err[-200:]})
            return None
        logger.error("%s: giving up after repeated rate limits", video_id)
        return None

    def year_at(self, ordered: list[dict], index: int) -> int | None:
        """Publish year of the video at *index*, probing neighbours when it is unavailable."""
        n = len(ordered)
        for delta in (0, 1, -1, 2, -2, 3, -3):
            j = index + delta
            if 0 <= j < n:
                row = self.get(ordered[j]["id"])
                if row:
                    return int(row["upload_date"][:4])
        return None


def years_by_position(ordered: list[dict], store: MetaStore) -> dict[str, int]:
    """Assign a publish year to every video of a newest-first list via boundary search."""
    n = len(ordered)
    if n == 0:
        return {}
    # Both ends of a channel list can be members-only: scan inward to the
    # first readable video. Unreadable videos past the last readable one take
    # the oldest year, and the boundary search stays inside the readable span.
    first = next((i for i in range(n) if store.get(ordered[i]["id"])), None)
    if first is None:
        raise RuntimeError("no readable video in the channel list")
    last = next(i for i in range(n - 1, -1, -1) if store.get(ordered[i]["id"]))
    newest = store.year_at(ordered, first)
    oldest = store.year_at(ordered, last)
    # first_before[y] = first index whose year < y (list is descending in time)
    first_before: dict[int, int] = {}
    for year in range(newest, oldest, -1):
        lo, hi = 0, last + 1  # invariant: year_at(lo-1) >= year, year_at(hi) < year
        while lo < hi:
            mid = (lo + hi) // 2
            y_mid = store.year_at(ordered, mid)
            if y_mid is None or y_mid >= year:
                lo = mid + 1
            else:
                hi = mid
        first_before[year] = lo
    years: dict[str, int] = {}
    for idx, row in enumerate(ordered):
        year = newest
        for y in range(newest, oldest, -1):
            if idx >= first_before[y]:
                year = y - 1
            else:
                break
        years[row["id"]] = year
    return years


def allocate(counts: dict[str, int], budget: int) -> dict[str, int]:
    """Proportional allocation with largest remainder; >= 1 per stratum when possible."""
    total = sum(counts.values())
    if total == 0:
        return {}
    budget = min(budget, total)
    quotas = {k: budget * c / total for k, c in counts.items()}
    alloc = {k: min(counts[k], int(q)) for k, q in quotas.items()}
    for k in sorted(counts, key=lambda k: counts[k]):
        if alloc[k] == 0 and sum(alloc.values()) < budget:
            alloc[k] = 1
    remainders = sorted(counts, key=lambda k: quotas[k] - int(quotas[k]), reverse=True)
    i = 0
    while sum(alloc.values()) < budget and remainders:
        k = remainders[i % len(remainders)]
        if alloc[k] < counts[k]:
            alloc[k] += 1
        i += 1
        if i > 10 * len(remainders) * budget:
            break
    while sum(alloc.values()) > budget:
        k = max(alloc, key=lambda k: alloc[k])
        alloc[k] -= 1
    return alloc


def _meta_record(video_id: str, channel: str, m: dict) -> dict:
    return {
        "video_id": video_id,
        "channel": channel,
        "title": m.get("title") or "",
        "duration": m.get("duration"),
        "description": m.get("description") or "",
        "chapters": m.get("chapters") or None,
        "upload_date": m["upload_date"],
    }


def select_channel(channel: str, pool: list[tuple[str, int]], per_channel: int, seed: int,
                   store: MetaStore, min_seconds: int) -> list[dict]:
    """Stratified picks for one channel; *pool* is [(video_id, year)]."""
    rng = random.Random(f"{seed}:{channel}")
    by_year: dict[int, list[str]] = defaultdict(list)
    for vid, year in pool:
        by_year[year].append(vid)
    alloc = allocate({y: len(v) for y, v in by_year.items()}, per_channel)
    logger.info("%s: pool %d, allocation %s", channel, len(pool), dict(sorted(alloc.items())))
    chosen: list[dict] = []
    short = 0
    for year in sorted(alloc):
        ids = sorted(by_year[year])
        rng.shuffle(ids)
        got = 0
        for vid in ids:
            if got >= alloc[year]:
                break
            m = store.get(vid)
            if not m or (m.get("duration") or 0) < min_seconds:
                continue
            chosen.append(_meta_record(vid, channel, m))
            got += 1
        short += alloc[year] - got
    if short:
        # Fill from any year, newest first, so the channel still gets its quota.
        spare = [vid for y in sorted(by_year, reverse=True) for vid in by_year[y]]
        picked = {r["video_id"] for r in chosen}
        rng.shuffle(spare)
        for vid in spare:
            if short <= 0:
                break
            if vid in picked:
                continue
            m = store.get(vid)
            if m and (m.get("duration") or 0) >= min_seconds:
                chosen.append(_meta_record(vid, channel, m))
                short -= 1
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser(description="Stratified long-form video sample with metadata")
    parser.add_argument("--catalog", required=True, type=Path, help="videos.json (list of {video_id, channel_id, ...})")
    parser.add_argument("--channels", required=True, type=Path, help="channels_input.txt")
    parser.add_argument("--work", required=True, type=Path, help="directory for flat listings and the metadata cache")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--per-channel", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--exclude", type=Path, default=None, help="text file with video ids to skip, one per line")
    parser.add_argument("--min-seconds", type=int, default=MIN_LONGFORM_SECONDS)
    args = parser.parse_args()

    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    channel_of = {r["video_id"]: r["channel_id"] for r in catalog}
    channel_urls = [l.strip() for l in args.channels.read_text(encoding="utf-8").splitlines() if l.strip()]
    exclude: set[str] = set()
    if args.exclude and args.exclude.exists():
        exclude = {l.strip() for l in args.exclude.read_text(encoding="utf-8").splitlines() if l.strip()}

    store = MetaStore(args.work / "pool_meta.jsonl")
    t0 = time.time()
    chosen: list[dict] = []
    year_table: dict[str, dict] = {}
    for url in channel_urls:
        key = _channel_key(url)
        ordered = list_videos_tab(url, args.work)
        years = years_by_position(ordered, store)
        pool = [
            (row["id"], years[row["id"]]) for row in ordered
            if channel_of.get(row["id"]) == key and row["id"] not in exclude
            and (row.get("duration") or 0) >= args.min_seconds
        ]
        year_table[key] = dict(sorted(Counter(y for _, y in pool).items()))
        logger.info("%s: Videos tab %d, catalogue long-form pool %d, years %s (%d requests so far)",
                    key, len(ordered), len(pool), year_table[key], store.requests)
        chosen.extend(select_channel(key, pool, args.per_channel, args.seed, store, args.min_seconds))

    chosen.sort(key=lambda r: (r["channel"], r["upload_date"], r["video_id"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(chosen, ensure_ascii=False, indent=1), encoding="utf-8")
    (args.work / "pool_years.json").write_text(json.dumps(year_table, indent=1), encoding="utf-8")
    logger.info("Wrote %d videos to %s (total %.1f h audio, %d metadata requests, %.1f min)",
                len(chosen), args.out, sum(r["duration"] or 0 for r in chosen) / 3600,
                store.requests, (time.time() - t0) / 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
