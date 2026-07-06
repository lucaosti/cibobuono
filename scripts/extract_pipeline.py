"""
extract_pipeline.py — Neuro-symbolic venue extraction: GLiNER candidates + rules + LLM.

Replaces monolithic chunk JSON extraction; keeps extract_from_video(...) signature
(plus optional transcript= for accurate time windows).
"""

from __future__ import annotations

__author__ = "Luca Ostinelli"


import json
import os
import re
from collections import defaultdict
from typing import TYPE_CHECKING

from thefuzz import fuzz

from scripts.batch_visit_llm import BatchEvalResult, DEFAULT_BATCH_SIZE, batch_evaluate_candidates
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
from scripts.utils import CONFIDENCE_THRESHOLD, CONF_RULE_VISIT, CONF_BATCH_VISIT, CONF_CHAPTER_HINT
from scripts.ner_candidates import Candidate, extract_all_chunks_candidates
from scripts.utils import setup_logging
from scripts.calibrate_confidence import apply_platt, load_calibration
from scripts.perceptor import get_perception
from scripts.venue_discovery import discover_venues_llm
from scripts.vote_aggregator import combine_confidence, perceptor_votes
from scripts.visit_classifier import (
    classify_candidate,
    classify_visit_rules,
    get_transcript_window,
    verify_venue_name,
    _looks_like_venue_name,
)

if TYPE_CHECKING:
    from scripts.video_intelligence import VideoIntel

logger = setup_logging("extract_pipeline")

# Confidence thresholds — canonical values live in utils.py; re-exported here
# so callers that already import from extract_pipeline keep working.
CONF_BATCH_MENTION = 0.72     # batch-LLM confirmed mention (may still be flagged if NER score low)
NER_FLAG_SCORE_MIN = 0.42     # NER score floor to flag low-confidence mentions
FUZZY_DEDUP_RATIO = 88        # thefuzz ratio threshold for near-duplicate name merge

# Confidence interpolation weights: blend classifier confidence with NER span score.
# Classifier signal dominates (0.65) since it uses full context; NER score contributes
# quality of the span itself.  Sum must equal 1.0.
_CONF_CLASSIFIER_WEIGHT = 0.65
_CONF_NER_WEIGHT = 0.35


def _batch_llm_enabled() -> bool:
    raw = os.environ.get("CIBOBUONO_BATCH_LLM", "auto").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    try:
        from scripts.hardware import get_profile

        return get_profile().has_cuda
    except Exception:
        return True

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
    # Use substring matching: GLiNER labels may include a definition suffix after ":".
    label = label.lower().split(":")[0].strip()
    if "bakery" in label or "pastry" in label or "gelateria" in label:
        return ["panificio", "forno"]
    if "bar" in label or "cafe" in label or "enoteca" in label:
        return ["bar", "caffe"]
    if "street food" in label or "rosticceria" in label or "friggitoria" in label:
        return ["street_food"]
    if "pizzeria" in label:
        return ["pizzeria"]
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
            response_format={"type": "json_object"},
            stop=["```"],
        )
        out = response["choices"][0]["message"]["content"].strip()
        data = _parse_detail_json(out)
        return data or {}
    except Exception as e:
        logger.warning(f"Detail LLM failed for '{place}': {e}")
        return {}


_calibration_cache: object = "_unloaded"


def _calibration() -> tuple[float, float] | None:
    """Lazily load the fitted Platt-scaling params (None if unfitted/missing)."""
    global _calibration_cache
    if _calibration_cache == "_unloaded":
        _calibration_cache = load_calibration()
    return _calibration_cache


def _visit_confidence(conf: float, ner_score: float, src: str) -> float:
    bonus = 0.08 if src == "rule" else 0.05 if src == "llm" else 0.0
    return min(1.0, conf * _CONF_CLASSIFIER_WEIGHT + ner_score * _CONF_NER_WEIGHT + bonus)


def _final_confidence(
    base_conf: float,
    perception_record: dict | None,
    candidate_name: str,
    candidate_time: float,
) -> float:
    """Blend the text-pipeline confidence with independent Perceptor votes
    (log-odds combination; no-op when there's no perception data), then
    apply the fitted confidence recalibration (no-op when unfitted)."""
    votes = perceptor_votes(perception_record, candidate_name, candidate_time)
    combined = combine_confidence(base_conf, votes) if votes else base_conf
    return _norm_confidence(apply_platt(combined, _calibration()))


def _promote_venue_hints(
    video_intel: VideoIntel | None,
    channel_rubriche: list[str],
    extractions: list[dict],
    flagged: list[dict],
    perception_record: dict | None = None,
) -> None:
    """Promote ONLY structured, reliable hints (title/chapter) into extractions.

    Description- and comment-derived hints come from noisy regex/crowd text, so
    they are NOT promoted blindly; they only help confirm NER candidates and
    must still be geocoded/verified downstream.
    """
    if not video_intel or not video_intel.venue_hints:
        return

    seen = {_normalize_name(e["locale_name"]) for e in extractions}
    # Structured sources with explicit venue names (not free-text comments).
    trusted_sources = {"title", "chapter", "description_timestamp"}
    conf_map = {"very_high": 0.85, "high": 0.80}

    rubrica = channel_rubriche[0] if channel_rubriche else ""
    if video_intel.series_name:
        rubrica = video_intel.series_name

    for h in video_intel.venue_hints:
        if h.get("source") not in trusted_sources:
            continue
        if h.get("confidence") not in ("very_high", "high"):
            continue
        name = _clean_locale_name(str(h.get("name") or ""))
        if not _is_valid_locale_name(name):
            continue
        norm = _normalize_name(name)
        if norm in seen:
            continue

        conf = conf_map.get(str(h.get("confidence") or ""), 0.80)
        start = h.get("start_time")
        conf = _final_confidence(conf, perception_record, name, float(start or 0.0))
        row = {
            "locale_name": name,
            "address": str(h.get("address") or "").strip(),
            "city": video_intel.city or "",
            "category": ["ristorante"],
            "rating": None,
            "sentiment": "neutral",
            "notes": f"Promoted from hint {h.get('source', '?')}",
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


# Confidence boost when two independent sources (NER-rule pipeline and
# holistic LLM discovery) agree on the same venue, instead of discarding the
# agreement signal by only keeping "whichever has higher confidence".
_SOURCE_AGREEMENT_BONUS = 0.10


def _merge_extraction_rows(
    ner_rows: list[dict],
    discovery_rows: list[dict],
) -> list[dict]:
    """Merge NER pipeline rows with holistic LLM discovery.

    Same venue found by both sources: boost confidence (agreement is itself
    evidence, per the Condorcet-jury intuition that independent agreeing
    voters are more informative than either alone) instead of only keeping
    the higher-confidence row. Venue found by one source only: keep as-is.
    """
    ner_by_norm: dict[str, dict] = {}
    for row in ner_rows:
        norm = _normalize_name(row.get("locale_name", ""))
        if norm:
            ner_by_norm[norm] = row

    disc_by_norm: dict[str, dict] = {}
    for row in discovery_rows:
        norm = _normalize_name(row.get("locale_name", ""))
        if norm:
            disc_by_norm[norm] = row

    merged: dict[str, dict] = {}
    for norm in set(ner_by_norm) | set(disc_by_norm):
        a = ner_by_norm.get(norm)
        b = disc_by_norm.get(norm)
        if a is not None and b is not None:
            best, other = (a, b) if float(a.get("confidence", 0)) >= float(b.get("confidence", 0)) else (b, a)
            row = dict(best)
            row["confidence"] = _norm_confidence(
                float(row.get("confidence", 0)) + _SOURCE_AGREEMENT_BONUS
            )
            if len(str(other.get("notes") or "")) > len(str(row.get("notes") or "")):
                row["notes"] = other.get("notes")
            row["_agreement"] = "ner+discovery"
            merged[norm] = row
        else:
            merged[norm] = a if a is not None else b

    # Fuzzy merge near-duplicates (e.g. "Da Remo" vs "Pizzeria Da Remo")
    keys = list(merged.keys())
    drop: set[str] = set()
    for i, a in enumerate(keys):
        if a in drop:
            continue
        for b in keys[i + 1 :]:
            if b in drop:
                continue
            if fuzz.ratio(a, b) >= FUZZY_DEDUP_RATIO or a in b or b in a:
                ra, rb = merged[a], merged[b]
                if float(ra.get("confidence", 0)) >= float(rb.get("confidence", 0)):
                    drop.add(b)
                else:
                    drop.add(a)
                    break
    return [merged[k] for k in merged if k not in drop]


def _build_visit_row(
    cand: Candidate,
    chunk: dict,
    *,
    evidence: str,
    conf: float,
    src: str,
    detail: dict,
    channel_rubriche: list[str],
    video_intel: VideoIntel | None,
    video_title: str,
    perception_record: dict | None = None,
) -> dict:
    name_clean = _clean_locale_name(cand.name)
    rubrica = channel_rubriche[0] if channel_rubriche else ""
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
        "confidence": _final_confidence(
            _visit_confidence(conf, cand.ner_score, src),
            perception_record,
            name_clean,
            cand.start_time,
        ),
        "chunk_start": chunk.get("start_timestamp", "0:00"),
        "chunk_end": chunk.get("end_timestamp", "0:00"),
        "chunk_start_seconds": cand.start_time,
        "mention_time": cand.start_time,
        "mention_timestamp": seconds_to_timestamp(cand.start_time),
        "verified": src == "llm",
    }
    if video_intel and video_intel.city and not row["city"]:
        row["city"] = video_intel.city

    chapter_ts = _hint_start_time(name_clean, video_intel)
    if chapter_ts is not None:
        row["mention_time"] = chapter_ts
        row["chunk_start_seconds"] = chapter_ts
        row["mention_timestamp"] = seconds_to_timestamp(chapter_ts)
        row["chunk_start"] = seconds_to_timestamp(chapter_ts)
        row["chunk_end"] = seconds_to_timestamp(chapter_ts + 90)
    return row


def _append_flagged_candidate(
    flagged: list[dict],
    cand: Candidate,
    chunk: dict,
    *,
    evidence: str,
    conf: float,
    reason: str,
) -> None:
    name_clean = _clean_locale_name(cand.name)
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
            "_flag_reason": reason,
        }
    )


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
    chapters = (youtube_extra or {}).get("chapters") if youtube_extra else None

    # Perceptor audio/video signals for this video, if the stage ran (best
    # effort: absent/errored Perceptor data means perceptor_votes() abstains).
    try:
        perception_record = get_perception(video_id)
    except Exception as e:
        logger.warning(f"Could not load perception record for {video_id}: {e}")
        perception_record = None

    # Holistic pass first: chapter/description-guided structured extraction.
    discovery_rows: list[dict] = []
    if llm and transcript:
        discovery_rows = discover_venues_llm(
            llm,
            transcript=transcript,
            video_title=video_title,
            video_description=video_description,
            video_intel=video_intel,
            chapters=chapters,
        )
        for row in discovery_rows:
            row["confidence"] = _final_confidence(
                float(row.get("confidence", 0.0)),
                perception_record,
                row.get("locale_name", ""),
                float(row.get("mention_time", 0.0) or 0.0),
            )

    # Accumulate positive visit hits per normalized name: list of dict records
    visit_hits: list[dict] = []
    flagged: list[dict] = []

    chunk_by_index = {int(c.get("chunk_index", i)): c for i, c in enumerate(chunks)}
    chunk_ner = extract_all_chunks_candidates(chunks)

    batch_items: list[dict] = []
    batch_meta: list[dict] = []

    for chunk_idx, (venues, ctx_ents) in chunk_ner.items():
        chunk = chunk_by_index.get(chunk_idx, chunks[0] if chunks else {})
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

            if not _looks_like_venue_name(name_clean):
                continue

            decision, reason = classify_visit_rules(window, cand, ctx_by_label)
            if decision == "visit":
                detail_window = get_transcript_window(transcript, cand.start_time, 22.0) or window
                detail = _detail_llm(llm, detail_window, name_clean, video_title) if llm else {}
                row = _build_visit_row(
                    cand,
                    chunk,
                    evidence=f"[rule:{reason}]",
                    conf=CONF_RULE_VISIT,
                    src="rule",
                    detail=detail,
                    channel_rubriche=channel_rubriche,
                    video_intel=video_intel,
                    video_title=video_title,
                    perception_record=perception_record,
                )
                visit_hits.append(
                    {
                        "norm": _normalize_name(name_clean),
                        "chunk_index": chunk_idx,
                        "row": row,
                        "protected": _is_protected_name(name_clean, video_intel),
                    }
                )
                continue

            if decision == "mention":
                continue

            cid = f"c{len(batch_items)}"
            batch_items.append({"id": cid, "name": name_clean, "window": window})
            batch_meta.append(
                {
                    "id": cid,
                    "cand": cand,
                    "chunk": chunk,
                    "ctx_by_label": ctx_by_label,
                    "window": window,
                    "reason": reason,
                }
            )

    batch_results: dict[str, BatchEvalResult] = {}
    if batch_meta and llm and _batch_llm_enabled():
        batch_size = int(os.environ.get("CIBOBUONO_BATCH_LLM_SIZE", str(DEFAULT_BATCH_SIZE)))
        batch_results = batch_evaluate_candidates(
            llm,
            batch_items,
            video_title=video_title,
            batch_size=batch_size,
        )
        logger.info(
            f"Batch LLM: {len(batch_items)} candidates in "
            f"{max(1, (len(batch_items) + batch_size - 1) // batch_size)} call(s)"
        )

    for meta in batch_meta:
        cand = meta["cand"]
        chunk = meta["chunk"]
        ctx_by_label = meta["ctx_by_label"]
        window = meta["window"]
        name_clean = _clean_locale_name(cand.name)
        cid = meta["id"]

        if batch_results:
            ev = batch_results.get(cid)
            if ev is None:
                continue
            if not ev.is_venue:
                continue
            is_visit = ev.is_visit
            evidence = ev.evidence or f"[batch:{meta['reason']}]"
            conf = CONF_BATCH_VISIT if is_visit else CONF_BATCH_MENTION
            src = "llm"
        else:
            if not verify_venue_name(llm, name_clean, window):
                continue
            is_visit, evidence, conf, src = classify_candidate(
                window, cand, ctx_by_label, llm
            )

        if not is_visit:
            if cand.ner_score >= NER_FLAG_SCORE_MIN:
                _append_flagged_candidate(
                    flagged,
                    cand,
                    chunk,
                    evidence=evidence,
                    conf=conf,
                    reason="possible_locale_mention_low_confidence",
                )
            continue

        detail_window = get_transcript_window(transcript, cand.start_time, 22.0) or window
        detail = _detail_llm(llm, detail_window, name_clean, video_title) if llm else {}
        row = _build_visit_row(
            cand,
            chunk,
            evidence=evidence,
            conf=conf,
            src=src,
            detail=detail,
            channel_rubriche=channel_rubriche,
            video_intel=video_intel,
            video_title=video_title,
            perception_record=perception_record,
        )
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

    _promote_venue_hints(
        video_intel, channel_rubriche, extractions, flagged, perception_record
    )

    extractions = _merge_extraction_rows(extractions, discovery_rows)

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
