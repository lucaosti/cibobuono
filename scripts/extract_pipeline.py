"""
extract_pipeline.py — Neuro-symbolic venue extraction: GLiNER candidates + rules + LLM.

Replaces monolithic chunk JSON extraction; keeps extract_from_video(...) signature
(plus optional transcript= for accurate time windows).
"""

from __future__ import annotations

__author__ = "Luca Ostinelli"


import json
import re
from collections import defaultdict
from typing import TYPE_CHECKING

from thefuzz import fuzz

from scripts.chunk_transcription import seconds_to_timestamp
from scripts.extract_locales import (
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    _clean_locale_name,
    _is_valid_locale_name,
    _normalize_rating,
    _normalize_sentiment,
    get_llm,
)
from scripts.utils import CONFIDENCE_THRESHOLD
from scripts.ner_candidates import Candidate, extract_chunk_candidates
from scripts.utils import setup_logging
from scripts.visit_classifier import classify_candidate, get_transcript_window

if TYPE_CHECKING:
    from scripts.video_intelligence import VideoIntel

logger = setup_logging("extract_pipeline")

_DETAIL_SYSTEM = (
    "You extract structured fields for ONE food venue from an Italian vlog excerpt. "
    "Return ONLY JSON."
)

_DETAIL_USER_TEMPLATE = (
    'VIDEO TITLE: "{title}"\n'
    'PLACE: "{place}"\n\n'
    'EXCERPT:\n"{window}"\n\n'
    "Return one JSON object:\n"
    '{{"rating": null or blogger overall grade as string (e.g. "8", "8--", "6++"),\n'
    ' "sentiment": "positive" | "neutral" | "negative",\n'
    ' "notes": "foods tried, prices, short factual observations",\n'
    ' "category": ["pizzeria"],\n'
    ' "city": "",\n'
    ' "address": ""}}\n\n'
    "Use null/empty if unknown. Italian only in notes."
)


def _normalize_name(name: str) -> str:
    return _clean_locale_name(name).lower().strip()


def _label_to_category(label: str) -> list[str]:
    label = label.lower().strip()
    if label == "bakery":
        return ["panificio", "forno"]
    if label == "bar or cafe":
        return ["bar", "caffe"]
    if label == "street food stall":
        return ["street_food"]
    if label == "food market":
        return ["mercato"]
    return ["ristorante"]


def _is_protected_name(name: str, video_intel: VideoIntel | None) -> bool:
    if not video_intel or not video_intel.venue_hints:
        return False

    nl = name.lower().strip()
    for h in video_intel.venue_hints:
        hn = (h.get("name") or "").lower().strip()
        if not hn:
            continue
        if fuzz.ratio(nl, hn) >= 75 or hn in nl or nl in hn:
            return True
    return False


def _hint_start_time(name: str, video_intel: VideoIntel | None) -> float | None:
    """Return start_time from chapter or description-timestamp hint."""
    if not video_intel or not video_intel.venue_hints:
        return None

    nl = name.lower().strip()
    best: float | None = None
    best_score = 0
    for h in video_intel.venue_hints:
        if h.get("source") not in ("chapter", "description_timestamp"):
            continue
        hn = (h.get("name") or "").lower().strip()
        if not hn:
            continue
        score = fuzz.ratio(nl, hn)
        if score > best_score and (score >= 75 or hn in nl or nl in hn):
            best_score = score
            st = h.get("start_time")
            if st is not None:
                best = float(st)
    return best


def _parse_detail_json(text: str) -> dict | None:
    text = text.strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(0))
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            pass
    return None


def _detail_llm(
    llm,
    window: str,
    place: str,
    video_title: str,
) -> dict:
    user_msg = _DETAIL_USER_TEMPLATE.format(
        title=(video_title or "")[:200].replace('"', "'"),
        place=place.replace('"', "'")[:120],
        window=(window[:2800]).replace('"', "'"),
    )
    try:
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": _DETAIL_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=min(400, LLM_MAX_TOKENS),
            temperature=LLM_TEMPERATURE,
            stop=["```", "\n\n\n"],
        )
        out = response["choices"][0]["message"]["content"].strip()
        data = _parse_detail_json(out)
        return data or {}
    except Exception as e:
        logger.warning(f"Detail LLM failed for '{place}': {e}")
        return {}


def _visit_confidence(conf: float, ner_score: float, src: str) -> float:
    bonus = 0.08 if src == "rule" else 0.05 if src == "llm" else 0.0
    return _norm_confidence(min(1.0, conf * 0.65 + ner_score * 0.35 + bonus))


def _promote_venue_hints(
    video_intel: VideoIntel | None,
    channel_rubriche: list[str],
    extractions: list[dict],
    flagged: list[dict],
) -> None:
    """Turn high-confidence title/chapter/description hints into extractions."""
    if not video_intel or not video_intel.venue_hints:
        return

    seen = {_normalize_name(e["locale_name"]) for e in extractions}
    conf_map = {"very_high": 0.88, "high": 0.82, "medium": 0.72}

    rubrica = channel_rubriche[0] if channel_rubriche else ""
    if video_intel.series_name:
        rubrica = video_intel.series_name

    for h in video_intel.venue_hints:
        name = _clean_locale_name(str(h.get("name") or ""))
        if not _is_valid_locale_name(name):
            continue
        norm = _normalize_name(name)
        if norm in seen:
            continue

        conf = conf_map.get(str(h.get("confidence") or ""), 0.68)
        start = h.get("start_time")
        row = {
            "locale_name": name,
            "address": str(h.get("address") or "").strip(),
            "city": video_intel.city or "",
            "category": ["ristorante"],
            "rating": None,
            "sentiment": "neutral",
            "notes": f"Promosso da hint {h.get('source', '?')}",
            "rubrica": rubrica,
            "confidence": conf,
            "chunk_start": seconds_to_timestamp(float(start or 0)),
            "chunk_end": seconds_to_timestamp(float(start or 0) + 90),
            "chunk_start_seconds": float(start or 0),
            "mention_time": float(start or 0),
            "mention_timestamp": seconds_to_timestamp(float(start or 0)),
            "verified": False,
        }

        if conf >= CONFIDENCE_THRESHOLD:
            extractions.append(row)
        else:
            flagged.append({**row, "_flag_reason": "hint_low_confidence"})
        seen.add(norm)


def _norm_confidence(x: object) -> float:
    try:
        v = float(x)
        return max(0.0, min(1.0, round(v, 2)))
    except (TypeError, ValueError):
        return 0.55


def extract_from_video(
    video_id: str,
    chunks: list[dict],
    channel_rubriche: list[str] | None = None,
    video_description: str = "",
    video_title: str = "",
    video_intel: VideoIntel | None = None,
    youtube_extra: dict | None = None,
    transcript: dict | None = None,
) -> tuple[list[dict], list[dict]]:
    """
    NER → visit/mention classification → optional detail LLM → cross-chunk filter.

    Args:
        transcript: full Whisper transcript (segments); improves time windows. Optional.
        youtube_extra: retained for API compatibility (chapters may inform future work).
    """
    # youtube_extra carries description_timestamps and chapters already processed
    # by run_pipeline into video_intel.venue_hints — nothing more to do here.
    channel_rubriche = channel_rubriche or []
    llm = get_llm()

    # Accumulate positive visit hits per normalized name: list of dict records
    visit_hits: list[dict] = []
    flagged: list[dict] = []

    for chunk in chunks:
        venues, ctx_ents = extract_chunk_candidates(chunk)
        ctx_by_label: dict[str, set[str]] = defaultdict(set)
        for c in ctx_ents:
            ctx_by_label[c.label].add(c.name.lower().strip())

        for cand in venues:
            name_clean = _clean_locale_name(cand.name)
            if not _is_valid_locale_name(name_clean):
                continue

            window = get_transcript_window(transcript, cand.start_time, 18.0)
            if not window:
                window = chunk.get("text", "")[:2000]

            is_visit, evidence, conf, src = classify_candidate(
                window, cand, ctx_by_label, llm
            )

            if not is_visit:
                if cand.ner_score >= 0.42:
                    flagged.append(
                        {
                            "locale_name": name_clean,
                            "city": "",
                            "address": "",
                            "category": _label_to_category(cand.label),
                            "rating": None,
                            "sentiment": "neutral",
                            "notes": evidence[:500],
                            "rubrica": "",
                            "confidence": max(0.25, 1.0 - conf),
                            "chunk_start": chunk.get("start_timestamp", "0:00"),
                            "chunk_end": chunk.get("end_timestamp", "0:00"),
                            "chunk_start_seconds": cand.start_time,
                            "mention_time": cand.start_time,
                            "mention_timestamp": seconds_to_timestamp(cand.start_time),
                            "_flag_reason": "possible_locale_mention_low_confidence",
                        }
                    )
                continue

            detail_window = get_transcript_window(transcript, cand.start_time, 22.0)
            if not detail_window:
                detail_window = window

            detail = _detail_llm(llm, detail_window, name_clean, video_title) if llm else {}

            rubrica = ""
            if channel_rubriche:
                rubrica = channel_rubriche[0]
            if video_intel and video_intel.series_name:
                rubrica = video_intel.series_name

            notes_parts = [evidence] if evidence else []
            dn = (detail.get("notes") or "").strip()
            if dn:
                notes_parts.append(dn)
            notes_merged = " | ".join(notes_parts)[:1500]

            cat = detail.get("category")
            if not isinstance(cat, list) or not cat:
                cat = _label_to_category(cand.label)

            row = {
                "locale_name": name_clean,
                "address": str(detail.get("address") or "").strip(),
                "city": str(detail.get("city") or "").strip(),
                "category": cat,
                "rating": _normalize_rating(detail.get("rating")),
                "sentiment": _normalize_sentiment(detail.get("sentiment")),
                "notes": notes_merged,
                "rubrica": rubrica,
                "confidence": _visit_confidence(conf, cand.ner_score, src),
                "chunk_start": chunk.get("start_timestamp", "0:00"),
                "chunk_end": chunk.get("end_timestamp", "0:00"),
                "chunk_start_seconds": cand.start_time,
                "mention_time": cand.start_time,
                "mention_timestamp": seconds_to_timestamp(cand.start_time),
                "verified": src == "llm",
            }
            if video_intel and video_intel.city and not row["city"]:
                row["city"] = video_intel.city

            # If a chapter hint provides a precise start time, prefer it over ASR-derived time.
            chapter_ts = _hint_start_time(name_clean, video_intel)
            if chapter_ts is not None:
                row["mention_time"] = chapter_ts
                row["chunk_start_seconds"] = chapter_ts
                row["mention_timestamp"] = seconds_to_timestamp(chapter_ts)
                row["chunk_start"] = seconds_to_timestamp(chapter_ts)
                row["chunk_end"] = seconds_to_timestamp(chapter_ts + 90)

            visit_hits.append(
                {
                    "norm": _normalize_name(name_clean),
                    "chunk_index": int(chunk.get("chunk_index", 0)),
                    "row": row,
                    "protected": _is_protected_name(name_clean, video_intel),
                }
            )

    # Cross-chunk consensus: keep if protected OR seen in >=2 chunks as visit
    by_norm: dict[str, list[dict]] = defaultdict(list)
    for h in visit_hits:
        by_norm[h["norm"]].append(h)

    extractions: list[dict] = []
    for norm, group in by_norm.items():
        chunks_seen = {h["chunk_index"] for h in group}
        prot = any(h["protected"] for h in group)
        llm_ok = any(h["row"].get("verified") for h in group)
        if not prot and len(chunks_seen) < 2:
            best_try = max(group, key=lambda x: x["row"]["confidence"])
            row = best_try["row"]
            if llm_ok or float(row.get("confidence", 0)) >= CONFIDENCE_THRESHOLD:
                extractions.append(row)
                continue
            fe = dict(row)
            fe["confidence"] = min(float(fe.get("confidence", 0.5)), 0.55)
            fe["_flag_reason"] = "single_chunk_visit"
            flagged.append(fe)
            continue

        best = max(group, key=lambda x: x["row"]["confidence"])
        extractions.append(best["row"])

    _promote_venue_hints(video_intel, channel_rubriche, extractions, flagged)

    # Threshold split
    hi: list[dict] = []
    lo: list[dict] = []
    for r in extractions:
        if float(r.get("confidence", 0)) >= CONFIDENCE_THRESHOLD:
            hi.append(r)
        else:
            r = dict(r)
            r["_flag_reason"] = r.get("_flag_reason") or "low_confidence"
            lo.append(r)

    flagged.extend(lo)

    logger.info(
        f"Video {video_id}: pipeline {len(hi)} extractions, {len(flagged)} flagged"
    )
    return hi, flagged
