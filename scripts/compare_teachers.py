"""
compare_teachers.py — Cross-check two teacher label files and build the review queue.

For every video the on-site venues of the *primary* and the *secondary* teacher
(:mod:`scripts.teacher_label` outputs) are paired: same name (fuzzy), one name
the acronym of the other, same address, the same quoted evidence, or, for street
stalls without a name and for transcription variants of a name, timestamps
within ``TIME_MATCH`` seconds. Every venue then gets automatic checks:

- ``agreement``: ``both`` | ``both_variant`` (paired, names differ: review it)
  | ``primary_only`` | ``secondary_only``
- ``evidence_in_transcript``: the quoted evidence occurs (fuzzily) in the speech
- ``timestamp_ok``: the timestamp parses and falls inside the video

The output JSONL has one row per venue and is the input of the human review
page; the summary printed at the end gives the counts per check.

With ``--consensus`` the two teachers are also merged into one gold label file:
the union of the paired venues, keeping only named ones (on gold200 no unnamed
venue a reviewer checked was confirmed) and dropping videos both teachers call
``other`` (not about food). The primary teacher's record wins; the secondary's
differing name is kept as ``alt_name``.

CLI:
    python -m scripts.compare_teachers --primary labels_q38_frames.jsonl \
        --secondary labels_q72.jsonl --meta gold_meta.json --transcripts DIR \
        --out review_queue.jsonl [--consensus gold_labels.jsonl]
"""

from __future__ import annotations

__author__ = "Luca Ostinelli"

import argparse
import json
from collections import Counter
from pathlib import Path

from thefuzz import fuzz

from scripts.schemas import timestamp_to_seconds as _strict_ts_to_seconds

NAME_MATCH = 80
ADDRESS_MATCH = 85
EVIDENCE_MATCH = 85
NAME_VARIANT_MATCH = 70  # partial ratio for differently spelled names close in time
TIME_MATCH = 45  # seconds; 90% of name-matched pairs on gold200 are within 41 s
EVIDENCE_SAME = 90  # both teachers quoting the same sentence point at the same venue
SAME_MOMENT = 10  # seconds; one teacher puts two named venues this close only 3 times in ~740


def load_labels(path: Path) -> dict[str, dict]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["video_id"]] = row
    return rows


def on_site(record: dict | None) -> list[dict]:
    return [v for v in (record or {}).get("venues") or [] if v.get("on_site", True)]


def _seconds(venue: dict) -> float | None:
    try:
        return _strict_ts_to_seconds(str(venue.get("timestamp") or ""))
    except (ValueError, TypeError):
        return None


def _time_gap(a: dict, b: dict) -> float:
    ta, tb = _seconds(a), _seconds(b)
    return abs(ta - tb) if ta is not None and tb is not None else float("inf")


def _is_acronym(short: str, long: str) -> bool:
    """``FTG`` of ``Frenchie To Go``: the initials of the long name's words."""
    words = long.split()
    return len(short) >= 2 and " " not in short and len(words) >= 2 and short == "".join(w[0] for w in words)


def same_venue(a: dict, b: dict) -> bool:
    name_a, name_b = (a.get("name") or "").lower(), (b.get("name") or "").lower()
    close = _time_gap(a, b) <= TIME_MATCH
    if name_a and name_b:
        if fuzz.token_set_ratio(name_a, name_b) >= NAME_MATCH:
            return True
        if close and fuzz.partial_ratio(name_a, name_b) >= NAME_VARIANT_MATCH:
            return True
        if _is_acronym(name_a, name_b) or _is_acronym(name_b, name_a):
            return True
        if _time_gap(a, b) <= SAME_MOMENT:
            return True
    addr_a, addr_b = (a.get("address") or "").lower(), (b.get("address") or "").lower()
    if addr_a and addr_b and fuzz.token_set_ratio(addr_a, addr_b) >= ADDRESS_MATCH:
        return True
    ev_a, ev_b = (a.get("evidence") or "").strip().lower(), (b.get("evidence") or "").strip().lower()
    if ev_a and ev_b and fuzz.ratio(ev_a, ev_b) >= EVIDENCE_SAME:
        return True
    # An unnamed venue can only be told apart by when it appears.
    return close and not (name_a and name_b)


def _names_differ(a: dict, b: dict) -> bool:
    name_a, name_b = (a.get("name") or "").lower(), (b.get("name") or "").lower()
    return bool(name_a and name_b) and fuzz.token_set_ratio(name_a, name_b) < NAME_MATCH


def pair_venues(primary: list[dict], secondary: list[dict]) -> list[tuple[dict | None, dict | None]]:
    """Greedy one-to-one pairing in primary order, nearest in time among the
    matching secondary venues; unmatched venues pair with None."""
    used: set[int] = set()
    pairs: list[tuple[dict | None, dict | None]] = []
    for a in primary:
        matches = [j for j, b in enumerate(secondary) if j not in used and same_venue(a, b)]
        j = min(matches, key=lambda j: _time_gap(a, secondary[j]), default=None)
        if j is None:
            pairs.append((a, None))
        else:
            used.add(j)
            pairs.append((a, secondary[j]))
    pairs.extend((None, b) for j, b in enumerate(secondary) if j not in used)
    return pairs


def evidence_in_transcript(evidence: str, speech: str) -> bool:
    evidence = (evidence or "").strip().lower()
    return bool(evidence) and fuzz.partial_ratio(evidence, speech) >= EVIDENCE_MATCH


def timestamp_ok(timestamp: str | None, duration: float | None) -> bool:
    seconds = _seconds({"timestamp": timestamp})
    return seconds is not None and (duration is None or 0 <= seconds <= duration)


def review_rows(video_id: str, meta: dict, primary: dict | None, secondary: dict | None, speech: str) -> list[dict]:
    rows = []
    for a, b in pair_venues(on_site(primary), on_site(secondary)):
        venue = a or b
        if a and b:
            agreement = "both_variant" if _names_differ(a, b) else "both"
        else:
            agreement = "primary_only" if a else "secondary_only"
        rows.append({
            "video_id": video_id,
            "title": meta.get("title", ""),
            "channel": meta.get("channel", ""),
            "agreement": agreement,
            "primary": a,
            "secondary": b,
            "evidence_in_transcript": evidence_in_transcript(venue.get("evidence", ""), speech),
            "timestamp_ok": timestamp_ok(venue.get("timestamp"), meta.get("duration")),
        })
    return rows


def consensus(primary: dict | None, secondary: dict | None) -> list[dict]:
    """Merged named venues of one video; ``sources`` says which teacher found each."""
    types = {(primary or {}).get("video_type"), (secondary or {}).get("video_type")}
    if types == {"other"}:
        return []
    merged = []
    for a, b in pair_venues(on_site(primary), on_site(secondary)):
        venue = dict(a or b)
        if not venue.get("name"):
            venue["name"] = (b or {}).get("name") or ""
        if not venue["name"]:
            continue
        for field in ("address", "city"):
            if not venue.get(field) and b:
                venue[field] = b.get(field) or ""
        if a and b and _names_differ(a, b):
            venue["alt_name"] = b["name"]
        venue["sources"] = [s for s, v in (("primary", a), ("secondary", b)) if v]
        merged.append(venue)
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description="Cross-check two teacher label files")
    parser.add_argument("--primary", required=True, type=Path)
    parser.add_argument("--secondary", required=True, type=Path)
    parser.add_argument("--meta", required=True, type=Path)
    parser.add_argument("--transcripts", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--consensus", type=Path, default=None, help="merged gold labels JSONL")
    args = parser.parse_args()

    meta = {m["video_id"]: m for m in json.loads(args.meta.read_text(encoding="utf-8"))}
    primary, secondary = load_labels(args.primary), load_labels(args.secondary)
    rows: list[dict] = []
    for video_id, m in meta.items():
        path = args.transcripts / f"{video_id}_transcript.json"
        speech = ""
        if path.exists():
            segments = json.loads(path.read_text(encoding="utf-8")).get("segments") or []
            speech = " ".join(s.get("text", "") for s in segments).lower()
        rows.extend(review_rows(video_id, m, primary.get(video_id), secondary.get(video_id), speech))

    args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"videos {len(meta)}, labelled primary {len(primary)}, secondary {len(secondary)}, venues {len(rows)}")
    print("agreement:", dict(Counter(r["agreement"] for r in rows)))
    print("evidence in transcript:", dict(Counter(r["evidence_in_transcript"] for r in rows)))
    print("timestamp ok:", dict(Counter(r["timestamp_ok"] for r in rows)))

    if args.consensus:
        gold = [
            {"video_id": vid, "title": m.get("title", ""), "venues": consensus(primary.get(vid), secondary.get(vid))}
            for vid, m in meta.items()
        ]
        args.consensus.write_text("".join(json.dumps(g, ensure_ascii=False) + "\n" for g in gold), encoding="utf-8")
        venues = [v for g in gold for v in g["venues"]]
        print(f"consensus: {len(venues)} named venues in {sum(1 for g in gold if g['venues'])} videos,",
              "found by:", dict(Counter("+".join(v["sources"]) for v in venues)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
