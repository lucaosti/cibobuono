"""
visit_classifier.py — Deterministic visit-vs-mention rules + LLM arbiter for ambiguous cases.
"""

from __future__ import annotations

__author__ = "Luca Ostinelli"


import json
import re
from typing import TYPE_CHECKING

from scripts.utils import LLM_MAX_TOKENS, LLM_TEMPERATURE, setup_logging

if TYPE_CHECKING:
    from scripts.ner_candidates import Candidate

logger = setup_logging("visit_classify")

# Minimal Italian food lexicon (expandable; keeps false "country/person" drops in check)
FOOD_LEXICON = frozenset({
    "pizza", "pasta", "maritozzo", "cornetto", "gelato", "tiramisu", "carbonara",
    "kebab", "panino", "tramezzino", "suppli", "arancino", "sushi", "ramen",
    "hamburger", "burger", "bistecca", "pesce", "fritto", "fritti", "dolce",
    "caffe", "caffè", "vino", "birra", "antipasto", "primo", "secondo", "contorno",
    "piatto", "porzione", "menu", "rosticceria", "forno", "pane", "pasticcino",
    "trapizzino", "panzerotto", "porchetta", "montanara", "parmigiana", "amatriciana",
    "cacio", "pepe", "gricia", "gnocco", "pinsa", "focaccia", "crescentina",
    "tiramisù", "cannolo", "arancina", "calzone", "frittatina", "zeppola",
    "tartare", "crudo", "salame", "prosciutto", "mozzarella", "burrata",
})

MENTION_PATTERNS = [
    re.compile(r"\bcome\s+(da|quello|il)\s+", re.I),
    re.compile(r"\btipo\s+", re.I),
    re.compile(r"\bnon\s+è\s+mica\b", re.I),
    re.compile(r"\bprossima\s+volta\b", re.I),
    re.compile(r"\bdopo\s+(andiamo|andremo|passiamo)\b", re.I),
    re.compile(r"\bun\s+giorno\s+", re.I),
    re.compile(r"\bho\s+sentito\s+che\b", re.I),
    re.compile(r"\bsembra\s+(il|da)\b", re.I),
]

VISIT_PATTERNS = [
    re.compile(r"\bqui\s+(da|siamo|siamo\s+da)\b", re.I),
    re.compile(r"\bsiamo\s+(da|a|al|allo|alla)\b", re.I),
    re.compile(r"\boggi\s+(sono|siamo)\s+(a|da|al)\b", re.I),
    re.compile(r"\b(ho|abbiamo|ci\s+hanno)\s+(preso|ordinato|servito)\b", re.I),
    re.compile(r"\b(assaggiamo|assaggiato|proviamo|provato)\b", re.I),
    re.compile(r"\borderiamo\b", re.I),
    re.compile(r"\b(mangiamo|mangio)\s+(qui|qua|da|al)\b", re.I),
    re.compile(r"\b(ci\s+siamo\s+fermati|entriamo|entriamo\s+in)\b", re.I),
    re.compile(r"\bil\s+loro\s+", re.I),
    re.compile(r"\bquesto\s+(posto|locale|ristorante)\b", re.I),
]

PRICE_RATING_PAT = re.compile(
    r"\d+[.,]?\d*\s*€|\b(?:voto|darei|do\s+un|metto|dai)\s+(?:\d|dieci|otto|nove)\b",
    re.I,
)

_VISIT_SYSTEM = (
    "You decide if an Italian food vlog excerpt shows the host PHYSICALLY at the named place "
    "(eating, ordering, on location). Answer ONLY JSON."
)

_VISIT_USER_TEMPLATE = (
    'PLACE NAME: "{name}"\n\n'
    'EXCERPT:\n"{window}"\n\n'
    "Return exactly one JSON object, no markdown:\n"
    '{{"visit": true or false, "evidence": "short quote copied from excerpt (max 25 words)"}}\n\n'
    "Rules:\n"
    "- visit=true only if they are clearly on-site at that business in this excerpt.\n"
    "- visit=false for comparisons, jokes, future plans, or name-dropping only.\n"
    "- evidence must be a substring of EXCERPT (or empty if visit=false)."
)


def get_transcript_window(
    transcript: dict | None,
    center_s: float,
    pad: float = 18.0,
) -> str:
    """Concatenate segment texts overlapping [center_s - pad, center_s + pad]."""
    if not transcript:
        return ""
    parts: list[str] = []
    for seg in transcript.get("segments", []) or []:
        s = float(seg.get("start", 0))
        e = float(seg.get("end", s))
        if e >= center_s - pad and s <= center_s + pad:
            t = (seg.get("text") or "").strip()
            if t:
                parts.append(t)
    return " ".join(parts).strip()


def _window_has_food_noun(window_lower: str) -> bool:
    for w in FOOD_LEXICON:
        if re.search(rf"\b{re.escape(w)}\b", window_lower):
            return True
    return False


def _name_near_pattern(window: str, pat: re.Pattern[str], name: str) -> bool:
    """Rough check: pattern match within ~120 chars of first name occurrence."""
    nl = name.lower()
    idx = window.lower().find(nl)
    if idx < 0:
        return bool(pat.search(window))
    lo = max(0, idx - 80)
    hi = min(len(window), idx + len(name) + 80)
    return bool(pat.search(window[lo:hi]))


def classify_visit_rules(
    window_text: str,
    candidate: "Candidate",
    context_by_label: dict[str, set[str]],
) -> tuple[str, str]:
    """
    Returns:
        decision: 'visit' | 'mention' | 'unsure'
        reason: short debug string
    """
    w = window_text.strip()
    if not w:
        return "unsure", "empty_window"

    wl = w.lower()
    name_l = candidate.name.lower().strip()

    # Blacklist: same string as geographic/person context entity
    geo_person = set()
    for lab in ("city", "neighborhood", "country", "person"):
        geo_person |= context_by_label.get(lab, set())
    if name_l in geo_person or any(name_l in g or g in name_l for g in geo_person if len(g) >= 4):
        if not _window_has_food_noun(wl) and not PRICE_RATING_PAT.search(w):
            return "mention", "context_entity_no_food"

    mention_hit = any(_name_near_pattern(w, p, candidate.name) for p in MENTION_PATTERNS)
    visit_hit = any(_name_near_pattern(w, p, candidate.name) for p in VISIT_PATTERNS)
    food_ok = _window_has_food_noun(wl)
    price_ok = bool(PRICE_RATING_PAT.search(w))

    if mention_hit and not visit_hit:
        return "mention", "mention_pattern"
    if visit_hit and (food_ok or price_ok or candidate.ner_score >= 0.55):
        return "visit", "visit_pattern"
    if visit_hit:
        return "unsure", "visit_pattern_weak_food"
    if mention_hit and visit_hit:
        return "unsure", "conflict_patterns"
    if food_ok and name_l in wl and price_ok:
        return "visit", "food_and_price_near_name"

    if name_l in wl and (food_ok or price_ok):
        return "unsure", "name_with_food_or_price"

    return "unsure", "no_clear_signal"


def _parse_llm_visit_json(text: str) -> dict | None:
    text = text.strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "visit" in data:
            return data
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[^{}]*\"visit\"[^{}]*\}", text, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(0))
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
    return None


def classify_with_llm(
    llm,
    window_text: str,
    candidate: "Candidate",
) -> tuple[bool, str, float]:
    """LLM arbiter. Returns (visit, evidence, confidence)."""
    user_msg = _VISIT_USER_TEMPLATE.format(
        name=candidate.name.replace('"', "'")[:120],
        window=(window_text[:2500]).replace('"', "'"),
    )
    try:
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": _VISIT_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=min(180, LLM_MAX_TOKENS),
            temperature=LLM_TEMPERATURE,
            stop=["```", "\n\n\n"],
        )
        out = response["choices"][0]["message"]["content"].strip()
        data = _parse_llm_visit_json(out)
        if data is None:
            return False, "", 0.45
        visit = bool(data.get("visit"))
        ev = str(data.get("evidence", "")).strip()[:500]
        conf = 0.85 if visit else 0.72
        return visit, ev, conf
    except Exception as e:
        logger.warning(f"LLM visit classify failed: {e}")
        return False, "", 0.4


# Words that are NEVER a venue proper name (dishes, ingredients, generic terms).
# Used to reject false positives before the (expensive) visit classification.
NON_VENUE_TERMS = frozenset({
    # dishes / food
    "pizza", "pasta", "carbonara", "amatriciana", "gricia", "cacio", "pepe",
    "maritozzo", "cornetto", "gelato", "tiramisu", "tiramisù", "supplì", "suppli",
    "arancino", "arancina", "trapizzino", "panzerotto", "porchetta", "montanara",
    "parmigiana", "lasagna", "lasagne", "gnocco", "pinsa", "focaccia", "calzone",
    "cannolo", "cannoli", "panino", "hamburger", "burger", "kebab", "ramen",
    "sushi", "bistecca", "tartare", "crudo", "mozzarella", "burrata", "prosciutto",
    "salame", "vino", "birra", "caffè", "caffe", "espresso", "cappuccino",
    "menu", "antipasto", "primo", "secondo", "contorno", "dolce", "dessert",
    # generic / filler
    "buono", "buonissimo", "ottimo", "delizioso", "fantastico", "spettacolare",
    "ragazzi", "amici", "video", "canale", "puntata", "oggi", "qui", "qua",
    "italia", "italiano", "italiana", "roma", "milano", "napoli", "torino",
})


def _looks_like_venue_name(name: str) -> bool:
    """Cheap structural gate: real venue names are capitalized proper nouns."""
    n = name.strip()
    if len(n) < 3:
        return False
    tokens = n.split()
    meaningful = [t for t in tokens if t.lower() not in NON_VENUE_TERMS]
    if not meaningful:
        return False
    # at least one capitalized token that is not a known non-venue term
    has_proper = any(
        t[:1].isupper() and t.lower() not in NON_VENUE_TERMS for t in tokens
    )
    return has_proper


_VENUE_VERIFY_SYSTEM = (
    "You verify whether a phrase is the PROPER NAME of a specific food business "
    "(restaurant, pizzeria, bakery, bar, street-food stall). Answer ONLY JSON."
)

_VENUE_VERIFY_TEMPLATE = (
    'PHRASE: "{name}"\n\n'
    'CONTEXT:\n"{window}"\n\n'
    "Return exactly one JSON object, no markdown:\n"
    '{{"is_venue": true or false}}\n\n'
    "Rules:\n"
    "- is_venue=true ONLY if the phrase names a specific food business.\n"
    "- is_venue=false for dishes, ingredients, cities, neighborhoods, people, "
    "product brands, or generic words.\n"
    "- If unsure, answer false."
)


def verify_venue_name(llm, name: str, window_text: str) -> bool:
    """Two-stage verification: reject non-venue candidates (VerifiNER-style).

    Without an LLM, fall back to the structural gate only.
    """
    if not _looks_like_venue_name(name):
        return False
    if llm is None:
        return True  # structural gate already passed
    user_msg = _VENUE_VERIFY_TEMPLATE.format(
        name=name.replace('"', "'")[:120],
        window=(window_text[:1200]).replace('"', "'"),
    )
    try:
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": _VENUE_VERIFY_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=30,
            temperature=0.0,
            stop=["```", "\n\n"],
        )
        out = response["choices"][0]["message"]["content"].strip()
        data = _parse_llm_visit_json(out.replace("is_venue", "visit")) or {}
        return bool(data.get("visit"))
    except Exception as e:
        logger.warning(f"Venue verification failed for '{name}': {e}")
        return True  # do not lose candidates on transient LLM errors


def classify_candidate(
    window_text: str,
    candidate: "Candidate",
    context_by_label: dict[str, set[str]],
    llm,
) -> tuple[bool, str, float, str]:
    """
    Full pipeline: rules then LLM if unsure.

    Returns:
        (is_visit, evidence, confidence, source) — source is 'rule' or 'llm'.
    """
    decision, reason = classify_visit_rules(window_text, candidate, context_by_label)
    if decision == "visit":
        return True, f"[rule:{reason}]", 0.82, "rule"
    if decision == "mention":
        return False, f"[rule:{reason}]", 0.75, "rule"

    if llm is None:
        return False, f"[unsure:{reason},no_llm]", 0.35, "rule"

    v, ev, c = classify_with_llm(llm, window_text, candidate)
    tag = ev or f"[llm:{reason}]"
    return v, tag, c, "llm"
