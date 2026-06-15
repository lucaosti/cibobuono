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

__author__ = "Luca Ostinelli"


import re
from dataclasses import dataclass, field

from scripts.utils import setup_logging

logger = setup_logging("intelligence")


@dataclass
class VideoIntel:
    """Structured intelligence derived from title + description."""
    video_type: str = "unknown"       # multi_venue_tour | single_venue | non_review | unknown
    city: str = ""
    venue_hints: list[dict] = field(default_factory=list)  # [{"name": ..., "address": ..., "start_time": ...}]
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

# "Forni criminali [nuova HIT] a CITY" → bakery tour
# Handles both "Forni criminali Roma" and "Forni criminali nuova HIT a Centocelle"
_FORNI_CRIMINALI = re.compile(
    r"Forni\s+criminali(?:\s+\S+)*?\s+a\s+([A-ZÀ-Úa-zà-ú][A-ZÀ-Úa-zà-ú\s]+?)(?:\s+da\s+\w+)?$",
    re.IGNORECASE,
)
# Fallback: plain "Forni criminali CITY" (no preposition "a")
_FORNI_CRIMINALI_PLAIN = re.compile(
    r"Forni\s+criminali\s+([A-ZÀ-Ú][a-zà-ú]+(?:\s+[A-ZÀ-Ú][a-zà-ú]+)*)$",
    re.IGNORECASE,
)

# "Pizza a taglio criminale VENUE" → single-venue episode (no city in title)
_PIZZA_TAGLIO_CRIMINALE = re.compile(
    r"Pizza\s+a\s+taglio\s+criminale\s+(.+)$",
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
    if not raw:
        return None
    if raw in _ITALIAN_NUMBERS:
        return _ITALIAN_NUMBERS[raw]
    if raw[0].isdigit():
        return raw
    return None


def analyze_title(title: str) -> VideoIntel:
    """Extract structural intelligence from the video title."""
    intel = VideoIntel()

    rating = _extract_title_rating(title)
    if rating:
        intel.title_rating = rating
        logger.info(f"Title → rating: {rating}")

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

    # "Forni criminali [nuova HIT] a CITY"
    m = _FORNI_CRIMINALI.search(title) or _FORNI_CRIMINALI_PLAIN.search(title)
    if m:
        city = m.group(1).strip()
        intel.video_type = "multi_venue_tour"
        intel.series_name = "Forni criminali"
        intel.city = city
        logger.info(f"Title → bakery tour of {city} (Forni criminali)")
        return intel

    # "Pizza a taglio criminale VENUE" — single venue, city unknown from title
    m = _PIZZA_TAGLIO_CRIMINALE.match(title)
    if m:
        venue_name = m.group(1).strip()
        intel.video_type = "single_venue"
        intel.series_name = "Pizza a taglio criminale"
        intel.venue_hints = [{"name": venue_name, "source": "title", "confidence": "high"}]
        logger.info(f"Title → pizza al taglio venue: '{venue_name}'")
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


# ── Food-video signals (rules before LLM) ────────────────────────────────

_FOOD_TITLE_SIGNALS = [
    re.compile(r"\bcriminale\b", re.IGNORECASE),
    re.compile(r"forni\s+criminali", re.IGNORECASE),
    re.compile(r"hit\s+di\s+franchino", re.IGNORECASE),
    re.compile(r"cosa\s+mangia", re.IGNORECASE),
    re.compile(r"\bmangiamo\b|\bassaggiamo\b|\bassaggio\b|\bdegust", re.IGNORECASE),
    re.compile(r"street\s*food|food\s*tour", re.IGNORECASE),
    re.compile(
        r"\b(?:pizzeria|trattoria|ristorante|osteria|forno|panificio|gelateria|"
        r"rosticceria|friggitoria)\b",
        re.IGNORECASE,
    ),
    re.compile(r"miglior(?:e|i)?\s+(?:pizza|pasta|gelato|carbonara|suppl)", re.IGNORECASE),
    re.compile(r"\bproviamo\b|\bandiamo\s+(?:da|in|a)\b", re.IGNORECASE),
    re.compile(r"giro\s+(?:di|dei|del|delle)\b", re.IGNORECASE),
]

# Broader fallback for LLM false negatives (title-only, checked after LLM says NO).
_FOOD_TITLE_LOOSE = re.compile(
    r"mang|assagg|degust|food|street.?food|pizz|past[aà]|ristor|trattor|osteri|"
    r"forn|gelat|rostic|frittor|criminale|mercato\s+dei|gastro",
    re.IGNORECASE,
)


def title_suggests_food(title: str) -> bool:
    """Loose title-only food signal for overriding LLM false negatives."""
    if not title:
        return False
    if any(pat.search(title) for pat in _FOOD_TITLE_SIGNALS):
        return True
    return bool(_FOOD_TITLE_LOOSE.search(title))


def food_video_confidence(
    title: str,
    intel: VideoIntel | None = None,
    description: str = "",
) -> tuple[bool | None, str]:
    """Rule-based food signal. True=food, False=not food, None=uncertain.

    Used to skip LLM food-check and to override LLM false negatives when the
    title/description clearly indicate a food-venue video.
    """
    intel = intel or VideoIntel()
    title = (title or "").strip()

    if intel.video_type in ("single_venue", "multi_venue_tour"):
        return True, f"title intel: {intel.video_type}"

    if intel.series_name:
        return True, f"known food series: {intel.series_name}"

    if intel.venue_hints:
        return True, f"venue hints from metadata ({len(intel.venue_hints)})"

    for pat in _FOOD_TITLE_SIGNALS:
        if pat.search(title):
            return True, f"food title pattern ({pat.pattern[:48]})"

    if description and parse_description_timestamps(description):
        return True, "description lists venue timestamps"

    if description and len(description) >= 120:
        desc_lower = description.lower()
        food_desc_markers = (
            "pizzeria", "trattoria", "ristorante", "osteria", "forno",
            "via ", "piazza ", "corso ", "indirizzo", "google maps",
        )
        if sum(1 for m in food_desc_markers if m in desc_lower) >= 2:
            return True, "description has multiple venue/address markers"

    return None, ""


_DESC_TS_LINE = re.compile(
    r"^(\d{1,2}:\d{2}(?::\d{2})?)\s+(.+)$",
)


def parse_description_timestamps(description: str) -> list[dict]:
    """
    Parse timestamp lines often found in YouTube descriptions, e.g. '1:23 Venue name'.
    Returns list of {timestamp, label}.
    """
    if not description:
        return []
    out: list[dict] = []
    for line in description.splitlines():
        line = line.strip()
        m = _DESC_TS_LINE.match(line)
        if m:
            out.append(
                {
                    "timestamp": m.group(1),
                    "label": m.group(2).strip()[:200],
                }
            )
    return out


def analyze_description(description: str, intel: VideoIntel) -> VideoIntel:
    """Enrich VideoIntel with venue hints from the full video description."""
    if not description:
        return intel

    existing_names = {h["name"].lower() for h in intel.venue_hints}

    def _add(name: str, confidence: str = "medium") -> None:
        n = name.strip()
        if len(n) < 3 or n.lower() in existing_names:
            return
        if n.lower() in ("roma", "milano", "torino", "napoli", "viterbo", "rieti",
                         "frosinone", "una", "questo", "quello", "link", "social"):
            return
        intel.venue_hints.append({"name": n, "source": "description", "confidence": confidence})
        existing_names.add(n.lower())
        logger.info(f"Description → venue hint: '{n}'")

    # "scopriamo X a CITY" / "vi porto a X" / "andiamo a X"
    for pattern in [
        r"(?:scopriamo|vi porto a|andiamo a|portiamo a|provare|mangeremo da|mangiamo da)\s+"
        r"([A-ZÀ-Ú][a-zà-ú']+(?:\s+[A-ZÀ-Úa-zà-ú']+){0,4})",
        r"(?:scopriamo|vi porto in|andiamo in)\s+([A-ZÀ-Ú][a-zà-ú']+(?:\s+[A-ZÀ-Úa-zà-ú']+){0,4})",
    ]:
        for m in re.finditer(pattern, description, re.IGNORECASE):
            _add(m.group(1).strip())

    # Venue-type prefix anywhere in description: "Pizzeria Da Remo", "Forno Roscioli"
    for m in re.finditer(
        r"(?i)\b(pizzeria|trattoria|ristorante|osteria|forno|panificio|pasticceria|"
        r"gelateria|friggitoria|rosticceria|enoteca|hosteria|locanda)\s+"
        r"([A-ZÀ-Ú][\wà-ú'']+(?:\s+[A-ZÀ-Ú][\wà-ú'']+){0,2})",
        description,
    ):
        _add(f"{m.group(1).capitalize()} {m.group(2).strip()}", "high")

    # "Da X" / "Al X" as standalone venue references
    for m in re.finditer(
        r"(?i)\b(?:da|al|alla|allo|dal|dalla|presso)\s+"
        r"([A-ZÀ-Ú][\wà-ú'']+(?:\s+[A-ZÀ-Ú][\wà-ú'']+){0,2})",
        description,
    ):
        _add(m.group(1).strip())

    # Lines that look like venue listings (bullet / numbered)
    for line in description.splitlines():
        line = line.strip()
        m = re.match(
            r"^(?:[-•*]|\d+[.)])\s*"
            r"((?:[Pp]izzeria|[Tt]rattoria|[Rr]istorante|[Ff]orno|[Oo]steria)\s+.+)$",
            line,
        )
        if m:
            _add(m.group(1).strip()[:80], "high")
        # Plain capitalized venue name on its own line (common in Italian descriptions)
        m2 = re.match(r"^([A-ZÀ-Ú][\wà-ú'']+(?:\s+[A-ZÀ-Ú][\wà-ú'']+){0,2})$", line)
        if m2 and 4 <= len(line) <= 60 and not re.search(r"http|www|@|#", line):
            _add(m2.group(1).strip(), "medium")

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


# ── Chapter titles as high-priority venue hints ───────────────────────────

# Chapter titles that are navigation/intro noise, not venue names
_NON_VENUE_CHAPTER_WORDS = frozenset({
    "intro", "introduzione", "conclusione", "outro", "fine", "inizio",
    "sponsor", "like", "iscriviti", "subscribe", "commenti", "link",
    "instagram", "tiktok", "facebook", "social", "recap", "riassunto",
    "credits", "crediti", "pausa", "break",
})


def _is_chapter_venue_name(title: str) -> bool:
    """Return True if a chapter title looks like a venue name, not navigation noise."""
    t = title.strip()
    if len(t) < 3 or len(t) > 80:
        return False
    tl = t.lower()
    if tl in _NON_VENUE_CHAPTER_WORDS:
        return False
    if re.fullmatch(r"[\d:#\s]+", t):
        return False
    if re.fullmatch(r"\d+", t):
        return False
    return True


def analyze_description_timestamps(
    dts: list[dict],
    intel: VideoIntel,
) -> VideoIntel:
    """
    Enrich VideoIntel with venue hints from description timestamp lines.

    Italian food creators often list visited venues as timestamped chapters in the
    video description (e.g. "1:30 Da Remo Roma").  These are high-signal hints —
    confidence is set to "high" (one step below chapters' "very_high" because the
    format is less structured and creators sometimes use generic labels).

    This processes the output of parse_description_timestamps() and should be called
    BEFORE analyze_chapters so that manually-set chapters can upgrade confidence.
    """
    if not dts:
        return intel

    from scripts.schemas import timestamp_to_seconds

    existing_names = {h["name"].lower() for h in intel.venue_hints}
    added = 0
    for entry in dts:
        raw = (entry.get("label") or "").strip()
        if not raw or not _is_chapter_venue_name(raw):
            continue
        if raw.lower() in existing_names:
            continue

        ts_str = entry.get("timestamp", "")
        try:
            start_seconds = float(timestamp_to_seconds(ts_str))
        except (ValueError, TypeError):
            start_seconds = None

        hint: dict = {
            "name": raw,
            "source": "description_timestamp",
            "confidence": "high",
        }
        if start_seconds is not None:
            hint["start_time"] = start_seconds

        intel.venue_hints.append(hint)
        existing_names.add(raw.lower())
        added += 1
        logger.info(f"Description timestamp → venue hint: '{raw}' (t={ts_str})")

    if added:
        if intel.video_type == "unknown":
            intel.video_type = "multi_venue_tour"
        logger.info(f"Description timestamps: {added} venue hints added")

    return intel


def analyze_chapters(chapters: list[dict], intel: VideoIntel) -> VideoIntel:
    """
    Enrich VideoIntel with venue hints from YouTube chapter titles.

    Chapters are the strongest possible hint source — YouTubers in multi-venue
    formats (e.g. "Roma criminale") name each chapter after the locale they visit.
    These get confidence="very_high" so they bypass the cross-chunk filter.
    """
    if not chapters:
        return intel

    existing_names = {h["name"].lower() for h in intel.venue_hints}
    added = 0
    for ch in chapters:
        raw = (ch.get("title") or "").strip()
        if not raw or not _is_chapter_venue_name(raw):
            continue
        if raw.lower() in existing_names:
            continue

        start_time = ch.get("start_time")
        hint: dict = {
            "name": raw,
            "source": "chapter",
            "confidence": "very_high",
        }
        if start_time is not None:
            hint["start_time"] = float(start_time)

        intel.venue_hints.append(hint)
        existing_names.add(raw.lower())
        added += 1
        logger.info(f"Chapter → venue hint: '{raw}' (t={start_time}s)")

    if added:
        if intel.video_type == "unknown":
            intel.video_type = "multi_venue_tour"
        logger.info(f"Chapters: {added} venue hints added")

    return intel


# Patterns in comments that often name venues
_COMMENT_VENUE_PATTERNS = [
    re.compile(
        r"(?:siamo stati|andate|consiglio|top|il miglior(?:e)?)\s+(?:da|al|alla|in)\s+"
        r"([A-ZÀ-Ú][\w\s'&.-]{2,40})",
        re.IGNORECASE,
    ),
    re.compile(
        r"@([A-ZÀ-Ú][\w\s'&.-]{2,35})",
    ),
    re.compile(
        r"(?:pizzeria|trattoria|forno|osteria|bar)\s+([A-ZÀ-Ú][\w\s'&.-]{2,35})",
        re.IGNORECASE,
    ),
]


def analyze_comments(comments: list[dict], intel: VideoIntel) -> VideoIntel:
    """Extract venue hints from YouTube comments (crowd-sourced names)."""
    if not comments:
        return intel

    existing = {h["name"].lower() for h in intel.venue_hints}
    mention_counts: dict[str, int] = {}
    first_display: dict[str, str] = {}  # key -> original-casing from first occurrence

    for c in comments:
        text = c.get("text") or ""
        if len(text) < 8:
            continue
        for pat in _COMMENT_VENUE_PATTERNS:
            for m in pat.finditer(text):
                name = m.group(1).strip(" .,!?:;")
                if len(name) < 3 or name.lower() in existing:
                    continue
                key = name.lower()
                if key not in first_display:
                    first_display[key] = name
                mention_counts[key] = mention_counts.get(key, 0) + 1 + (1 if c.get("like_count", 0) >= 5 else 0)

    added = 0
    for key, count in sorted(mention_counts.items(), key=lambda x: -x[1]):
        if count < 2:
            continue
        display = first_display.get(key, key.title())
        if display.lower() in existing:
            continue
        intel.venue_hints.append(
            {
                "name": display,
                "source": "comment",
                "confidence": "high" if count >= 4 else "medium",
            }
        )
        existing.add(display.lower())
        added += 1
        if added >= 8:
            break

    if added:
        logger.info(f"Comments: {added} venue hints from {len(comments)} comments")

    return intel
