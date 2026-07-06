"""
vote_aggregator.py — Combine independent visit-detection signals via weighted log-odds.

The text pipeline (rules + LLM arbiter + holistic discovery) already produces a
single blended confidence per candidate. This module adds genuinely independent,
different-modality votes on top of it — currently the Perceptor audio/video
signals (VLM caption OCR, speaker diarization) — and combines them the way a
simplified Snorkel-style label model would: each voter's evidence is converted
to log-odds and averaged by a reliability weight, not just "keep the max".

Perceptor voters default to a lower weight than the text pipeline because there
is not yet enough corrected-outcome history to estimate their true reliability
(cold start); see scripts/calibrate_confidence.py for the same reasoning
applied to the overall confidence formula.
"""

from __future__ import annotations

__author__ = "Luca Ostinelli"

import math
from dataclasses import dataclass

from thefuzz import fuzz

from scripts.utils import setup_logging

logger = setup_logging("vote_aggregator")

# Reliability weights: how much each voter's log-odds contributes relative to
# the base text-pipeline confidence (weight 1.0). Perceptor voters are
# independent modalities but currently unvalidated against real outcomes.
WEIGHT_BASE = 1.0
WEIGHT_PERCEPTOR_OCR = 0.35
WEIGHT_PERCEPTOR_SPEAKER = 0.30

# Fuzzy match ratio (thefuzz.partial_ratio) above which a caption is considered
# to mention the candidate venue name.
OCR_MATCH_RATIO = 82

# Time window (seconds) around a candidate's mention time to search for a
# matching VLM caption or diarized speaker segment.
_TIME_WINDOW_S = 20.0

_EPS = 1e-4


@dataclass
class Vote:
    source: str
    decision: str  # "visit" | "mention"
    confidence: float  # this voter's own confidence in its decision, 0-1
    weight: float  # reliability weight relative to WEIGHT_BASE


def _logit(p: float) -> float:
    p = min(max(p, _EPS), 1.0 - _EPS)
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def perceptor_votes(
    perception_record: dict | None,
    candidate_name: str,
    candidate_time: float,
    *,
    window_s: float = _TIME_WINDOW_S,
) -> list[Vote]:
    """Independent-modality votes derived from Perceptor audio/video signals.

    Returns an empty list (abstain) when there's no perception record, no
    captions/diarization near the candidate's time, or nothing to say.
    """
    votes: list[Vote] = []
    if not perception_record:
        return votes

    name_l = candidate_name.lower().strip()
    if not name_l:
        return votes

    # OCR/caption vote: does a nearby VLM caption mention the venue name or
    # matching signage/menu text?
    video_info = perception_record.get("video") or {}
    best_ratio = 0
    for cap in video_info.get("captions") or []:
        t = cap.get("t")
        if t is None or abs(float(t) - candidate_time) > window_s:
            continue
        text = (cap.get("caption") or "").lower()
        if not text:
            continue
        best_ratio = max(best_ratio, fuzz.partial_ratio(name_l, text))
    if best_ratio >= OCR_MATCH_RATIO:
        votes.append(
            Vote(
                source="perceptor_ocr",
                decision="visit",
                confidence=min(0.95, best_ratio / 100.0),
                weight=WEIGHT_PERCEPTOR_OCR,
            )
        )

    # Speaker vote: is the person talking at this timestamp the channel's
    # dominant/registered voice (heuristic proxy for "the host"), or someone
    # else (guest/bystander) — catches "a guest describes their own visit"
    # being misread as the host's visit.
    audio_info = perception_record.get("audio") or {}
    for seg in audio_info.get("segment_speakers") or []:
        s, e = seg.get("start", 0.0), seg.get("end", 0.0)
        if s <= candidate_time <= e:
            speaker = seg.get("speaker")
            if speaker == "S?":
                break
            if speaker == "S0":
                votes.append(
                    Vote(
                        source="perceptor_speaker",
                        decision="visit",
                        confidence=0.65,
                        weight=WEIGHT_PERCEPTOR_SPEAKER,
                    )
                )
            else:
                votes.append(
                    Vote(
                        source="perceptor_speaker",
                        decision="mention",
                        confidence=0.60,
                        weight=WEIGHT_PERCEPTOR_SPEAKER,
                    )
                )
            break

    return votes


def combine_confidence(base_confidence: float, votes: list[Vote]) -> float:
    """Weighted log-odds combination of the base (text-pipeline) confidence
    with additional independent votes. Falls back to base_confidence unchanged
    when there are no votes (the common case today, with Perceptor disabled).
    """
    if not votes:
        return base_confidence

    logit_total = WEIGHT_BASE * _logit(base_confidence)
    weight_total = WEIGHT_BASE
    for v in votes:
        p = v.confidence if v.decision == "visit" else 1.0 - v.confidence
        logit_total += v.weight * _logit(p)
        weight_total += v.weight

    return _sigmoid(logit_total / weight_total)
