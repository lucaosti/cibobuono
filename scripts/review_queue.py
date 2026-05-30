"""
review_queue.py — Manual review queue and user locale reports.

Flagged segments (incertezze) live in flagged_segments.json.
User reports of wrong locales go to locale_reports.json.
"""

from __future__ import annotations

__author__ = "Luca Ostinelli"

import hashlib
import json
from typing import Any

from scripts.utils import (
    CORRECTIONS_JSON,
    FLAGGED_SEGMENTS_JSON,
    LOCALES_JSON,
    VIDEOS_JSON,
    VISITS_JSON,
    load_json,
    save_json,
    today_str,
)

LOCALE_REPORTS_JSON = FLAGGED_SEGMENTS_JSON.parent / "locale_reports.json"

_FLAG_REASON_LABELS = {
    "possible_locale_mention_low_confidence": "Menzione incerta",
    "low_confidence": "Bassa confidenza",
    "single_chunk_visit": "Un solo segmento",
    "hint_low_confidence": "Hint debole",
    "geocoding_failed": "Geocoding fallito",
    "osm_not_found": "Non trovato su OSM",
    "rating_title_transcript_mismatch": "Voto titolo ≠ trascrizione",
    "locale_name_not_identified": "Nome mancante",
    "address_not_identified": "Indirizzo mancante",
    "ambiguous_locale_reference": "Riferimento ambiguo",
}


def youtube_timestamp_url(video_id: str, seconds: float | int) -> str:
    return f"https://youtu.be/{video_id}?t={int(seconds)}"


def _video_titles() -> dict[str, str]:
    return {v["video_id"]: v.get("title", "") for v in load_json(VIDEOS_JSON)}


def pending_reviews(*, limit: int = 100) -> list[dict]:
    """Flagged segments awaiting human review, enriched for the dashboard."""
    titles = _video_titles()
    out: list[dict] = []
    for seg in load_json(FLAGGED_SEGMENTS_JSON):
        if seg.get("reviewed_by_human"):
            continue
        reason = seg.get("reason", "")
        out.append(
            {
                **seg,
                "video_title": titles.get(seg.get("video_id", ""), ""),
                "reason_label": _FLAG_REASON_LABELS.get(reason, reason),
                "review_key": _review_key(seg),
            }
        )
        if len(out) >= limit:
            break
    return out


def _review_key(seg: dict) -> str:
    raw = f"{seg.get('video_id')}|{seg.get('timestamp_start')}|{seg.get('locale_name')}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def resolve_review(
    *,
    video_id: str,
    timestamp_start: str,
    action: str,
    locale_name: str | None = None,
    notes: str = "",
) -> tuple[bool, str]:
    """Mark a flagged segment reviewed; approve creates a verified visit."""
    if action == "approve":
        from scripts.manual_edits import promote_flagged_to_visit

        return promote_flagged_to_visit(
            video_id=video_id,
            timestamp_start=timestamp_start,
            locale_name=locale_name,
            notes=notes,
        )

    flagged = load_json(FLAGGED_SEGMENTS_JSON)
    found = False
    target_name = locale_name

    for seg in flagged:
        if seg.get("video_id") != video_id:
            continue
        if seg.get("timestamp_start") != timestamp_start:
            continue
        if locale_name and seg.get("locale_name") and seg.get("locale_name") != locale_name:
            continue
        seg["reviewed_by_human"] = True
        seg["reviewed_date"] = today_str()
        seg["review_action"] = action
        if notes:
            seg["review_notes"] = notes[:500]
        target_name = seg.get("locale_name") or locale_name
        found = True
        break

    if not found:
        return False, "Segmento non trovato"

    save_json(FLAGGED_SEGMENTS_JSON, flagged)

    if action == "reject" and target_name:
        _add_hide_correction(target_name, notes or "Rifiutato in revisione")

    return True, f"Segmento segnato come {action}"


def _add_hide_correction(locale_name: str, reason: str) -> None:
    locales = load_json(LOCALES_JSON)
    lid = None
    nl = locale_name.lower().strip()
    for loc in locales:
        if loc.get("name", "").lower().strip() == nl:
            lid = loc.get("locale_id")
            break
    if not lid:
        return
    corrections = load_json(CORRECTIONS_JSON)
    if any(c.get("locale_id") == lid and c.get("type") == "hide" for c in corrections):
        return
    corrections.append(
        {
            "locale_id": lid,
            "type": "hide",
            "reason": reason[:300],
        }
    )
    save_json(CORRECTIONS_JSON, corrections)


def list_reports(*, limit: int = 50) -> list[dict]:
    if not LOCALE_REPORTS_JSON.exists():
        return []
    try:
        data = json.loads(LOCALE_REPORTS_JSON.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    open_only = [r for r in data if r.get("status", "open") == "open"]
    return open_only[:limit]


def submit_report(
    *,
    locale_name: str,
    reason: str,
    video_id: str = "",
    visit_id: str = "",
    locale_id: str = "",
    youtube_url: str = "",
) -> dict:
    """Record a user report that a locale citation is wrong."""
    LOCALE_REPORTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    reports: list[dict] = []
    if LOCALE_REPORTS_JSON.exists():
        try:
            reports = json.loads(LOCALE_REPORTS_JSON.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            reports = []

    if not youtube_url and video_id:
        youtube_url = f"https://youtu.be/{video_id}"

    report_id = hashlib.sha256(
        f"{visit_id}|{locale_id}|{locale_name}|{today_str()}".encode()
    ).hexdigest()[:12]

    entry = {
        "report_id": f"report_{report_id}",
        "locale_name": locale_name.strip(),
        "locale_id": locale_id,
        "visit_id": visit_id,
        "video_id": video_id,
        "youtube_url": youtube_url,
        "reason": reason.strip()[:500],
        "status": "open",
        "reported_at": today_str(),
    }
    reports.append(entry)
    LOCALE_REPORTS_JSON.write_text(
        json.dumps(reports, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if locale_id:
        corrections = load_json(CORRECTIONS_JSON)
        if not any(c.get("locale_id") == locale_id and c.get("type") == "hide" for c in corrections):
            corrections.append(
                {
                    "locale_id": locale_id,
                    "type": "hide",
                    "reason": f"Segnalazione utente: {reason[:200]}",
                }
            )
            save_json(CORRECTIONS_JSON, corrections)

    return entry


def resolve_report(report_id: str, *, action: str = "resolved") -> tuple[bool, str]:
    if not LOCALE_REPORTS_JSON.exists():
        return False, "Nessuna segnalazione"
    reports = json.loads(LOCALE_REPORTS_JSON.read_text(encoding="utf-8"))
    for r in reports:
        if r.get("report_id") == report_id:
            r["status"] = action
            r["resolved_at"] = today_str()
            LOCALE_REPORTS_JSON.write_text(
                json.dumps(reports, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return True, "Segnalazione aggiornata"
    return False, "Segnalazione non trovata"


def visits_with_links(*, limit: int = 40) -> list[dict]:
    """Recent visits enriched with locale name and YouTube deep link."""
    locales = {l["locale_id"]: l for l in load_json(LOCALES_JSON)}
    titles = _video_titles()
    out: list[dict] = []
    for v in reversed(load_json(VISITS_JSON)):
        loc = locales.get(v.get("locale_id", ""), {})
        out.append(
            {
                "visit_id": v.get("visit_id"),
                "locale_id": v.get("locale_id"),
                "locale_name": loc.get("name", "?"),
                "city": loc.get("city", ""),
                "video_id": v.get("video_id"),
                "video_title": titles.get(v.get("video_id", ""), "")[:80],
                "youtube_url": v.get("youtube_url", ""),
                "timestamp_start": v.get("timestamp_start"),
                "confidence": v.get("llm_confidence"),
                "rating": v.get("rating"),
            }
        )
        if len(out) >= limit:
            break
    return out
