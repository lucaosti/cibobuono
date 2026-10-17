"""
trace_candidates.py — Per-candidate diagnostic trace of the extraction chain.

For every cached transcript (or the given video ids) this runs the same steps
as :func:`scripts.extract_pipeline.extract_from_video` but records what happened
to EACH venue candidate at EACH stage: NER span and score, structural name
gate, deterministic rule decision and reason, optional LLM arbiter verdict,
and the final outcome (published / flagged / dropped) with its confidence.

The JSONL output is the raw material for a gold set: a reviewer (or a stronger
teacher model) labels each row with the true visit/mention/not-a-venue class,
and the per-stage columns show which stage produced each false positive or
false negative.

CLI:
    python -m scripts.trace_candidates [VIDEO_ID ...] [--with-llm] [--out PATH]

Without video ids every ``cache/*_transcript.json`` is traced. Without
``--with-llm`` the LLM stages are skipped (rules-only), which is what the
pipeline itself does when no GGUF model is available.
"""

from __future__ import annotations

__author__ = "Luca Ostinelli"

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from scripts.chunk_transcription import chunk_transcription
from scripts.extract_locales import _clean_locale_name, _is_valid_locale_name
from scripts.ner_candidates import extract_all_chunks_candidates
from scripts.utils import CACHE_DIR, LOGS_DIR, setup_logging, today_str
from scripts.visit_classifier import (
    _looks_like_venue_name,
    classify_visit_rules,
    get_transcript_window,
)

logger = setup_logging("trace_candidates")

WINDOW_PAD_S = 18.0
WINDOW_MAX_CHARS = 700


def _load_transcript(video_id: str) -> dict | None:
    path = CACHE_DIR / f"{video_id}_transcript.json"
    if not path.exists():
        logger.warning(f"No cached transcript for {video_id}")
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Cannot read transcript {path}: {e}")
        return None


def trace_video(
    video_id: str,
    transcript: dict,
    *,
    llm=None,
    final_rows: tuple[list[dict], list[dict]] | None = None,
) -> list[dict]:
    """Return one trace row per venue candidate found in *transcript*.

    ``final_rows`` is the ``(extractions, flagged)`` pair returned by the real
    pipeline for this video; when given, each candidate is matched by
    normalized name so the row records where it ended up.
    """
    chunks = chunk_transcription(transcript)
    if not chunks:
        return []
    chunk_by_index = {int(c.get("chunk_index", i)): c for i, c in enumerate(chunks)}
    chunk_ner = extract_all_chunks_candidates(chunks)

    published: dict[str, float] = {}
    flagged: dict[str, tuple[float, str]] = {}
    if final_rows is not None:
        for r in final_rows[0]:
            published[_clean_locale_name(r.get("locale_name", "")).lower()] = float(
                r.get("confidence", 0)
            )
        for r in final_rows[1]:
            flagged[_clean_locale_name(r.get("locale_name", "")).lower()] = (
                float(r.get("confidence", 0)),
                str(r.get("_flag_reason") or ""),
            )

    rows: list[dict] = []
    for chunk_idx, (venues, ctx_ents) in sorted(chunk_ner.items()):
        chunk = chunk_by_index.get(chunk_idx, {})
        ctx_by_label: dict[str, set[str]] = defaultdict(set)
        for c in ctx_ents:
            ctx_by_label[c.label].add(c.name.lower().strip())

        for cand in venues:
            name_clean = _clean_locale_name(cand.name)
            window = get_transcript_window(transcript, cand.start_time, WINDOW_PAD_S)
            if not window:
                window = (chunk.get("text") or "")[:2000]

            row = {
                "video_id": video_id,
                "chunk_index": chunk_idx,
                "name": name_clean,
                "raw_span": cand.name,
                "label": cand.label.split(":")[0].strip(),
                "ner_score": round(cand.ner_score, 3),
                "start_time": round(cand.start_time, 1),
                "window": window[:WINDOW_MAX_CHARS],
                "valid_name": _is_valid_locale_name(name_clean),
                "looks_like_venue": _looks_like_venue_name(name_clean),
                "rule_decision": None,
                "rule_reason": None,
                "llm_visit": None,
                "llm_evidence": None,
                "final": "dropped",
                "final_conf": None,
                "flag_reason": None,
                "gold": None,
            }
            if row["valid_name"] and row["looks_like_venue"]:
                decision, reason = classify_visit_rules(window, cand, ctx_by_label)
                row["rule_decision"] = decision
                row["rule_reason"] = reason
                if llm is not None and decision == "unsure":
                    from scripts.visit_classifier import classify_with_llm

                    v, ev, _ = classify_with_llm(llm, window, cand)
                    row["llm_visit"] = v
                    row["llm_evidence"] = ev
            else:
                row["rule_reason"] = "name_gate"

            key = name_clean.lower()
            if key in published:
                row["final"] = "published"
                row["final_conf"] = published[key]
            elif key in flagged:
                row["final"] = "flagged"
                row["final_conf"], row["flag_reason"] = flagged[key]
            rows.append(row)
    return rows


def summarize(rows: list[dict]) -> dict:
    """Aggregate counts that show where candidates go."""
    by_final = Counter(r["final"] for r in rows)
    by_rule = Counter(f"{r['rule_decision']}:{r['rule_reason']}" for r in rows)
    return {
        "candidates": len(rows),
        "videos": len({r["video_id"] for r in rows}),
        "final": dict(by_final),
        "rule_paths": dict(by_rule.most_common()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Trace every venue candidate through the extraction stages")
    parser.add_argument("video_ids", nargs="*", help="video ids (default: every cached transcript)")
    parser.add_argument("--with-llm", action="store_true", help="also run the LLM arbiter and the full pipeline")
    parser.add_argument("--out", type=Path, default=None, help="JSONL output path")
    args = parser.parse_args()

    video_ids = args.video_ids or sorted(
        p.name.replace("_transcript.json", "") for p in CACHE_DIR.glob("*_transcript.json")
    )
    if not video_ids:
        logger.warning("No transcripts to trace")
        return 0

    llm = None
    if args.with_llm:
        from scripts.extract_locales import get_llm

        llm = get_llm()

    out_path = args.out or (LOGS_DIR / f"trace_{today_str()}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    for vid in video_ids:
        transcript = _load_transcript(vid)
        if not transcript:
            continue
        final_rows = None
        if args.with_llm:
            from scripts.extract_pipeline import extract_from_video

            chunks = chunk_transcription(transcript)
            final_rows = extract_from_video(vid, chunks, transcript=transcript)
        rows = trace_video(vid, transcript, llm=llm, final_rows=final_rows)
        all_rows.extend(rows)
        logger.info(f"{vid}: {len(rows)} candidates")

    with open(out_path, "w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = summarize(all_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"trace written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
