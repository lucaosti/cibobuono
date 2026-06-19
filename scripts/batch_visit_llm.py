"""
batch_visit_llm.py — Batch LLM evaluation of venue candidates (fewer GPU round-trips).

Replaces per-candidate verify_venue_name + classify_with_llm with one structured call
per batch (up to N candidates), dramatically reducing LLM idle/token overhead.
"""

from __future__ import annotations

__author__ = "Luca Ostinelli"

import json
import re
from dataclasses import dataclass

from scripts.utils import LLM_MAX_TOKENS, LLM_TEMPERATURE, setup_logging

logger = setup_logging("batch_visit_llm")

DEFAULT_BATCH_SIZE = 10

_BATCH_SYSTEM = (
    "You evaluate Italian food-vlog venue candidates. "
    "For each place decide if it is a real food-business name and if the host visits on-site. "
    "Return ONLY valid JSON."
)

_BATCH_USER = (
    "VIDEO TITLE: \"{title}\"\n\n"
    "CANDIDATES (evaluate each independently):\n"
    "{blocks}\n\n"
    "Return one JSON object:\n"
    '{{"results": [\n'
    '  {{"id": "c0", "is_venue": true/false, "is_visit": true/false, '
    '"evidence": "short quote or empty"}}\n'
    "]}}\n\n"
    "Rules:\n"
    "- is_venue=false for dishes, cities, people, brands, generic words.\n"
    "- is_visit=true only if they eat/order/enter that business in the excerpt.\n"
    "- evidence must come from the excerpt (max 20 words) or empty if is_visit=false."
)


@dataclass
class BatchEvalResult:
    is_venue: bool
    is_visit: bool
    evidence: str


def _parse_batch_json(text: str) -> list[dict]:
    text = (text or "").strip()
    payload: dict | None = None
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            payload = data
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\"results\"[\s\S]*\}", text)
        if m:
            try:
                payload = json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    if not payload:
        return []
    raw = payload.get("results")
    return raw if isinstance(raw, list) else []


def batch_evaluate_candidates(
    llm,
    items: list[dict],
    *,
    video_title: str = "",
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, BatchEvalResult]:
    """
    Evaluate many candidates in batched LLM calls.

    Each item: {id, name, window}
    Returns map id -> BatchEvalResult
    """
    if llm is None or not items:
        return {}

    out: dict[str, BatchEvalResult] = {}
    title = (video_title or "")[:200].replace('"', "'")

    for start in range(0, len(items), batch_size):
        batch = items[start : start + batch_size]
        blocks: list[str] = []
        for it in batch:
            cid = str(it["id"])
            name = str(it["name"]).replace('"', "'")[:120]
            window = str(it.get("window") or "")[:800].replace('"', "'")
            blocks.append(f'[{cid}] PLACE: "{name}"\nEXCERPT: "{window}"\n')

        user_msg = _BATCH_USER.format(title=title, blocks="\n".join(blocks))
        try:
            response = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": _BATCH_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=min(900, LLM_MAX_TOKENS),
                # Binary classification: greedy decoding reduces hallucination vs
                # sampling (Wang et al., 2023 — Self-Consistency).
                temperature=0.0,
                response_format={"type": "json_object"},
                stop=["```"],
            )
            raw = response["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.warning(f"Batch LLM evaluate failed: {e}")
            continue

        by_id = {str(r.get("id", "")): r for r in _parse_batch_json(raw) if isinstance(r, dict)}
        for it in batch:
            cid = str(it["id"])
            row = by_id.get(cid, {})
            is_visit = bool(row.get("is_visit"))
            evidence = str(row.get("evidence") or "").strip()[:500]
            # Hallucination guard: a genuine on-site visit must have citeable evidence.
            # Empty evidence with is_visit=true is a model artefact; demote to False.
            if is_visit and len(evidence) < 5:
                logger.debug("Batch LLM: demoting is_visit=true (no evidence) for id=%s", cid)
                is_visit = False
            out[cid] = BatchEvalResult(
                is_venue=bool(row.get("is_venue")),
                is_visit=is_visit,
                evidence=evidence,
            )

    return out
