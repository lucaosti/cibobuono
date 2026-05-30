"""
venue_discovery.py — Holistic LLM venue discovery from timestamped transcript.

Complements chunk-level GLiNER+rules: one structured pass over the full video
(or chapter segments) finds visits that NER misses (ASR garbling, no trigger words).
"""

from __future__ import annotations

__author__ = "Luca Ostinelli"

import json
import re
from typing import TYPE_CHECKING

from scripts.chunk_transcription import seconds_to_timestamp
from scripts.extract_locales import (
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    _clean_locale_name,
    _is_valid_locale_name,
    _normalize_rating,
    _normalize_sentiment,
)
from scripts.utils import setup_logging

if TYPE_CHECKING:
    from scripts.video_intelligence import VideoIntel

logger = setup_logging("venue_discovery")

# ~12k chars ≈ safe slice for 32B with prompt overhead
_MAX_TRANSCRIPT_CHARS = 14_000
_WINDOW_SECONDS = 600  # 10 min fallback windows when no chapters


_DISCOVERY_SYSTEM = (
    "You extract food-venue VISITS from Italian food-vlog transcripts. "
    "Return ONLY valid JSON."
)

_DISCOVERY_USER = (
    'VIDEO TITLE: "{title}"\n'
    'CITY (if known): "{city}"\n'
    'SERIES / FORMAT: "{series}"\n'
    'DESCRIPTION EXCERPT:\n"{description}"\n\n'
    "STRUCTURED HINTS (chapters / description timestamps — treat as anchors):\n"
    "{hints}\n\n"
    "TRANSCRIPT (timestamped):\n"
    "{transcript}\n\n"
    "Return one JSON object:\n"
    '{{"venues": [\n'
    '  {{"name": "proper business name", "timestamp": "MM:SS or H:MM:SS", '
    '"on_site": true, "category": ["pizzeria"], "rating": null, '
    '"sentiment": "positive|neutral|negative", "evidence": "short quote"}}\n'
    "]}}\n\n"
    "Rules:\n"
    "- Include ONLY specific food businesses the host visits ON CAMERA (eat/order/enter).\n"
    "- name = restaurant/pizzeria/forno/bar/osteria/trattoria/market stall proper name.\n"
    "- EXCLUDE: dishes, ingredients, cities, neighborhoods, people, brands, cattle breeds, "
    "generic words ('questo posto' without a name).\n"
    "- Use HINTS timestamps when the transcript is unclear; align each venue to when they arrive.\n"
    "- If a hint names a venue and the host clearly visits it, include it.\n"
    "- Empty venues list if none found.\n"
    "- Italian business names as spoken/written by creator."
)


def _parse_discovery_json(text: str) -> list[dict]:
    text = (text or "").strip()
    payload: dict | None = None
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            payload = data
    except json.JSONDecodeError:
        pass
    if payload is None:
        m = re.search(r"\{[\s\S]*\"venues\"[\s\S]*\}", text)
        if m:
            try:
                data = json.loads(m.group(0))
                if isinstance(data, dict):
                    payload = data
            except json.JSONDecodeError:
                pass
    if not payload:
        return []

    raw = payload.get("venues")
    if not isinstance(raw, list):
        return []
    out: list[dict] = []
    for item in raw:
        if isinstance(item, dict) and item.get("name"):
            out.append(item)
    return out


def _timestamp_to_seconds(ts: str) -> float | None:
    ts = (ts or "").strip()
    if not ts:
        return None
    parts = ts.split(":")
    try:
        nums = [int(p) for p in parts]
    except ValueError:
        return None
    if len(nums) == 2:
        return float(nums[0] * 60 + nums[1])
    if len(nums) == 3:
        return float(nums[0] * 3600 + nums[1] * 60 + nums[2])
    return None


def _hint_lines(video_intel: VideoIntel | None) -> str:
    if not video_intel or not video_intel.venue_hints:
        return "(none)"
    lines: list[str] = []
    for h in video_intel.venue_hints:
        name = (h.get("name") or "").strip()
        if not name:
            continue
        src = h.get("source", "?")
        st = h.get("start_time")
        ts = seconds_to_timestamp(float(st)) if st is not None else "?"
        lines.append(f"- {name} @ {ts} ({src})")
    return "\n".join(lines) if lines else "(none)"


def build_timestamped_transcript(
    transcript: dict,
    chapters: list[dict] | None = None,
    *,
    max_chars: int = _MAX_TRANSCRIPT_CHARS,
) -> str:
    """Format transcript as [MM:SS] lines, optionally grouped by chapter headers."""
    segments = transcript.get("segments") or []
    if not segments:
        return (transcript.get("text") or "")[:max_chars]

    if chapters:
        sorted_ch = sorted(
            [c for c in chapters if isinstance(c, dict)],
            key=lambda c: float(c.get("start_time") or 0),
        )
        if sorted_ch:
            parts: list[str] = []
            for i, ch in enumerate(sorted_ch):
                start = float(ch.get("start_time") or 0)
                end = (
                    float(sorted_ch[i + 1].get("start_time"))
                    if i + 1 < len(sorted_ch)
                    else float(segments[-1].get("end", start + 1))
                )
                title = (ch.get("title") or "").strip()
                header = f"=== CHAPTER: {title} ({seconds_to_timestamp(start)}) ==="
                chunk_lines: list[str] = []
                for seg in segments:
                    s = float(seg.get("start", 0))
                    if start <= s < end:
                        t = (seg.get("text") or "").strip()
                        if t:
                            chunk_lines.append(f"[{seconds_to_timestamp(s)}] {t}")
                if chunk_lines:
                    parts.append(header)
                    parts.extend(chunk_lines)
            text = "\n".join(parts)
            if text.strip():
                return text[:max_chars]

    lines: list[str] = []
    for seg in segments:
        s = float(seg.get("start", 0))
        t = (seg.get("text") or "").strip()
        if t:
            lines.append(f"[{seconds_to_timestamp(s)}] {t}")
    return "\n".join(lines)[:max_chars]


def discover_venues_llm(
    llm,
    *,
    transcript: dict,
    video_title: str = "",
    video_description: str = "",
    video_intel: VideoIntel | None = None,
    chapters: list[dict] | None = None,
) -> list[dict]:
    """
    Single holistic LLM pass: list all on-site food-venue visits.

    Returns extraction-shaped dicts (locale_name, mention_time, confidence, …).
    """
    if llm is None:
        return []

    formatted = build_timestamped_transcript(transcript, chapters)
    if len(formatted.strip()) < 20:
        return []

    city = (video_intel.city if video_intel else "") or ""
    series = (video_intel.series_name if video_intel else "") or ""
    if video_intel and video_intel.video_type:
        series = series or video_intel.video_type

    user_msg = _DISCOVERY_USER.format(
        title=(video_title or "")[:200].replace('"', "'"),
        city=city.replace('"', "'"),
        series=series.replace('"', "'"),
        description=(video_description or "")[:2500].replace('"', "'"),
        hints=_hint_lines(video_intel).replace('"', "'"),
        transcript=formatted.replace('"', "'"),
    )

    try:
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": _DISCOVERY_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=min(1200, LLM_MAX_TOKENS),
            temperature=LLM_TEMPERATURE,
            stop=["```", "\n\n\n"],
        )
        out = response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.warning(f"Holistic venue discovery failed: {e}")
        return []

    rows: list[dict] = []
    for item in _parse_discovery_json(out):
        name = _clean_locale_name(str(item.get("name") or ""))
        if not _is_valid_locale_name(name):
            continue
        if not item.get("on_site", True):
            continue

        ts_raw = str(item.get("timestamp") or "")
        mention_time = _timestamp_to_seconds(ts_raw)
        if mention_time is None and video_intel:
            for h in video_intel.venue_hints or []:
                hn = (h.get("name") or "").lower()
                if hn and (hn in name.lower() or name.lower() in hn):
                    st = h.get("start_time")
                    if st is not None:
                        mention_time = float(st)
                        break
        if mention_time is None:
            mention_time = 0.0

        conf = 0.82
        if video_intel:
            for h in video_intel.venue_hints or []:
                hn = (h.get("name") or "").lower()
                if hn and (hn in name.lower() or name.lower() in hn):
                    src = h.get("source", "")
                    if src == "chapter":
                        conf = 0.88
                    elif src in ("title", "description_timestamp"):
                        conf = 0.85
                    break

        cat = item.get("category")
        if not isinstance(cat, list) or not cat:
            cat = ["ristorante"]

        rows.append(
            {
                "locale_name": name,
                "address": "",
                "city": city,
                "category": cat,
                "rating": _normalize_rating(item.get("rating")),
                "sentiment": _normalize_sentiment(item.get("sentiment")),
                "notes": str(item.get("evidence") or "")[:500],
                "rubrica": series,
                "confidence": conf,
                "chunk_start": seconds_to_timestamp(mention_time),
                "chunk_end": seconds_to_timestamp(mention_time + 90),
                "chunk_start_seconds": mention_time,
                "mention_time": mention_time,
                "mention_timestamp": seconds_to_timestamp(mention_time),
                "verified": True,
                "_source": "discovery_llm",
            }
        )

    logger.info(f"Holistic discovery: {len(rows)} venues from LLM")
    return rows
