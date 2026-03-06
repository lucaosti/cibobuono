"""
video_intelligence.py — Extract structural cues from video title and description.

Before sending anything to the LLM, we can infer a LOT from the title alone:
- Video type (multi-venue tour, single venue review, non-review, …)
- City being visited
- Venue name (for single-venue formats like "Hit di Franchino: X City")
- Whether the video should be skipped entirely

These signals are passed to the LLM as HIGH-PRIORITY hints so it focuses on
confirming and enriching rather than discovering from garbled ASR.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from scripts.utils import DATA_DIR, setup_logging

logger = setup_logging("intelligence")

GROUND_TRUTH_FILE = DATA_DIR / "ground_truth.json"


@dataclass
class VideoIntel:
    """Structured intelligence derived from title + description."""
    video_type: str = "unknown"       # multi_venue_tour | single_venue | non_review | unknown
    city: str = ""
    venue_hints: list[dict] = field(default_factory=list)  # [{"name": ..., "address": ...}]
    skip_reason: str = ""             # non-empty → video should be skipped
    series_name: str = ""             # e.g. "Hit di Franchino", "Forni criminali"
    title_rating: str | None = None   # rating extracted from title (e.g., "10" from "da DIECI")


# ── Title patterns (channel-specific) ────────────────────────────────────

# "Hit di Franchino: VENUE City" → single venue review
_HIT_PATTERN = re.compile(
    r"Hit\s+di\s+Franchino\s*:\s*(.+?)\s+(Roma|Milano|Torino|Napoli|Firenze|Bologna|"
    r"Genova|Palermo|Catania|Bari|Verona|Padova|Brescia|Bergamo|Parma|Modena|"
    r"Viterbo|Frosinone|Latina|Rieti|Cassino|[A-Z][a-zà-ú]+)$",
    re.IGNORECASE,
)

# "{CITY} criminale" → multi-venue food tour
_CITY_CRIMINALE = re.compile(
    r"^([A-ZÀ-Ú][A-ZÀ-Úa-zà-ú\s]+?)\s+criminale$",
    re.IGNORECASE,
)

# "Forni criminali {CITY}" → bakery tour
_FORNI_CRIMINALI = re.compile(
    r"Forni\s+criminali\s+([A-ZÀ-Ú][A-ZÀ-Úa-zà-ú\s]+?)(?:\s+da\s+\w+)?$",
    re.IGNORECASE,
)

# "Cosa mangia {NAME}" → eating-with-someone format (still has venues)
_COSA_MANGIA = re.compile(r"Cosa\s+mangia\s+(.+)", re.IGNORECASE)

# Italian number words → digit mapping for title ratings
_ITALIAN_NUMBERS = {
    "uno": "1", "due": "2", "tre": "3", "quattro": "4", "cinque": "5",
    "sei": "6", "sette": "7", "otto": "8", "nove": "9", "dieci": "10",
}

# "da DIECI", "da 8", "da OTTO" etc. in the title = overall rating
_TITLE_RATING_PATTERN = re.compile(
    r"\bda\s+(" + "|".join(_ITALIAN_NUMBERS.keys()) + r"|\d+(?:[+\-]{1,2})?)\b",
    re.IGNORECASE,
)

# Non-review patterns
_NON_REVIEW_PATTERNS = [
    re.compile(r"^Cena\s+a\s+\d+\s+mani", re.IGNORECASE),
    re.compile(r"^Salviamo\s+", re.IGNORECASE),
    re.compile(r"(?:sfida|challenge|versus|vs)\s+", re.IGNORECASE),
    re.compile(r"^Franchino\s+contro\s+", re.IGNORECASE),
    re.compile(r"^Sono\s+stato\s+umiliato", re.IGNORECASE),
]


def _extract_title_rating(title: str) -> str | None:
    """Extract rating from title patterns like 'da DIECI', 'da 8', 'da OTTO'."""
    m = _TITLE_RATING_PATTERN.search(title)
    if not m:
        return None
    raw = m.group(1).strip().lower()
    if raw in _ITALIAN_NUMBERS:
        return _ITALIAN_NUMBERS[raw]
    # Numeric with optional modifiers
    if raw[0].isdigit():
        return raw
    return None


def analyze_title(title: str) -> VideoIntel:
    """Extract structural intelligence from the video title."""
    intel = VideoIntel()

    # Rating from title (applies to all types)
    rating = _extract_title_rating(title)
    if rating:
        intel.title_rating = rating
        logger.info(f"Title → rating: {rating}")

    # Non-review detection
    for pat in _NON_REVIEW_PATTERNS:
        if pat.search(title):
            intel.video_type = "non_review"
            intel.skip_reason = f"Title matches non-review pattern: {pat.pattern}"
            return intel

    # "Hit di Franchino: VENUE City"
    m = _HIT_PATTERN.match(title)
    if m:
        venue_name = m.group(1).strip()
        city = m.group(2).strip()
        intel.video_type = "single_venue"
        intel.series_name = "Hit di Franchino"
        intel.city = city
        intel.venue_hints = [{"name": venue_name, "source": "title", "confidence": "high"}]
        logger.info(f"Title → single venue: '{venue_name}' in {city} (Hit di Franchino)")
        return intel

    # "Forni criminali CITY"
    m = _FORNI_CRIMINALI.search(title)
    if m:
        city = m.group(1).strip()
        intel.video_type = "multi_venue_tour"
        intel.series_name = "Forni criminali"
        intel.city = city
        logger.info(f"Title → bakery tour of {city} (Forni criminali)")
        return intel

    # "CITY criminale"
    m = _CITY_CRIMINALE.match(title)
    if m:
        city = m.group(1).strip()
        intel.video_type = "multi_venue_tour"
        intel.city = city
        logger.info(f"Title → food tour of {city}")
        return intel

    # "Cosa mangia X"
    m = _COSA_MANGIA.match(title)
    if m:
        intel.video_type = "multi_venue_tour"
        logger.info(f"Title → eating-with format: {m.group(1)}")
        return intel

    return intel


def analyze_description(description: str, intel: VideoIntel) -> VideoIntel:
    """Enrich VideoIntel with venue hints from the description."""
    if not description:
        return intel

    # "scopriamo X a CITY" / "vi porto a X" / "andiamo a X"
    for pattern in [
        r"(?:scopriamo|vi porto a|andiamo a|portiamo a|provare)\s+([A-ZÀ-Ú][a-zà-ú']+(?:\s+[A-ZÀ-Úa-zà-ú']+){0,3})\s+a\s+([A-ZÀ-Ú][a-zà-ú]+)",
        r"(?:scopriamo|vi porto in|andiamo in)\s+([A-ZÀ-Ú][a-zà-ú']+(?:\s+[A-ZÀ-Úa-zà-ú']+){0,3})",
    ]:
        for m in re.finditer(pattern, description, re.IGNORECASE):
            name = m.group(1).strip()
            if len(name) >= 3 and name.lower() not in ("roma", "milano", "torino", "napoli",
                                                         "viterbo", "rieti", "frosinone",
                                                         "una", "questo", "quello"):
                existing_names = {h["name"].lower() for h in intel.venue_hints}
                if name.lower() not in existing_names:
                    intel.venue_hints.append({"name": name, "source": "description", "confidence": "medium"})
                    logger.info(f"Description → venue hint: '{name}'")

    # Address patterns: "Via/Piazza/Corso ... N, CITY"
    for m in re.finditer(
        r"((?:Via|Piazza|Corso|Viale|Largo|Vicolo)\s+[^,]+,\s*\d+[^,]*,\s*\d{5}\s+[A-ZÀ-Ú][a-zà-ú]+)",
        description,
    ):
        addr = m.group(1).strip()
        for hint in intel.venue_hints:
            if not hint.get("address"):
                hint["address"] = addr
                break

    return intel


def load_ground_truth() -> dict[str, dict]:
    """Load ground truth indexed by video_id."""
    if not GROUND_TRUTH_FILE.exists():
        return {}
    try:
        with open(GROUND_TRUTH_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {item["video_id"]: item for item in data}
    except Exception:
        return {}


def get_ground_truth_for_video(video_id: str) -> dict | None:
    """Get ground truth data for a specific video, if available."""
    gt = load_ground_truth()
    return gt.get(video_id)


def build_few_shot_examples() -> str:
    """Build few-shot examples from ground truth for the LLM prompt."""
    gt = load_ground_truth()
    if not gt:
        return ""

    examples = []
    for vid_id, gt_data in list(gt.items())[:3]:
        if gt_data.get("video_type") == "non_review":
            continue
        venues = gt_data.get("venues", [])
        if not venues:
            continue
        title = gt_data.get("title", "")
        venue_strs = [f'"{v["name"]}"' for v in venues]
        fp_strs = [f'"{fp}"' for fp in gt_data.get("false_positives", [])[:2]]

        example = f'TITLE: "{title}" → CORRECT: {", ".join(venue_strs)}'
        if fp_strs:
            example += f' | FALSE POSITIVES (do NOT extract): {", ".join(fp_strs)}'
        examples.append(example)

    if not examples:
        return ""

    return (
        "\nFEW-SHOT EXAMPLES from this channel:\n"
        + "\n".join(f"  • {e}" for e in examples)
        + "\n"
    )
