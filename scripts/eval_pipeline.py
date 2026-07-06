"""
eval_pipeline.py — Precision/recall/F1 of the visit-vs-mention classifier
against a hand-labeled gold set, so each redundancy/calibration change can be
attributed a measured improvement instead of judged by feel.

data/eval_set.json is a plain JSON list (repo convention — see
scripts.utils.load_json) of examples:
    {
        "candidate_name": "Roscioli",
        "window_text": "Siamo da Roscioli e assaggiamo la pizza bianca...",
        "start_time": 12.0,        # optional, defaults to 0.0
        "ner_score": 0.8,           # optional, defaults to 0.6
        "gold_label": "visit"       # "visit" | "mention"
    }

Starts empty: this repo has not published any real extractions yet
(data/visits.json is empty), so there is no reviewed data to seed it from.
Populate it by hand-labeling real (window_text, gold_label) pairs as
corrections.json/visits.json accumulate — see scripts.calibrate_confidence
for the companion mechanism that reuses corrections.json for confidence
recalibration once there's enough of it.

CLI:
    python -m scripts.eval_pipeline [--with-llm]
"""

from __future__ import annotations

__author__ = "Luca Ostinelli"

from scripts.ner_candidates import Candidate
from scripts.utils import DATA_DIR, load_json, setup_logging
from scripts.visit_classifier import classify_visit_rules, classify_with_llm

logger = setup_logging("eval_pipeline")

EVAL_SET_JSON = DATA_DIR / "eval_set.json"


def load_eval_set(path=EVAL_SET_JSON) -> list[dict]:
    examples = load_json(path)
    valid = [e for e in examples if e.get("candidate_name") and e.get("gold_label") in ("visit", "mention")]
    dropped = len(examples) - len(valid)
    if dropped:
        logger.warning(f"Dropped {dropped} malformed eval_set.json entries")
    return valid


def _candidate_for(example: dict) -> Candidate:
    name = str(example["candidate_name"])
    return Candidate(
        name=name,
        label=str(example.get("label", "restaurant")),
        start_char=0,
        end_char=len(name),
        start_time=float(example.get("start_time", 0.0)),
        chunk_index=0,
        ner_score=float(example.get("ner_score", 0.6)),
    )


def evaluate(eval_set: list[dict], llm=None) -> dict:
    """Run the rule engine (+ LLM arbiter if llm is given) against the gold
    set and return precision/recall/F1 for the "visit" class.

    No silent capping: every example in eval_set is scored, and the returned
    dict's "n" always matches len(eval_set) so a shrunk eval set is visible.
    """
    tp = fp = fn = tn = 0
    for example in eval_set:
        window = str(example.get("window_text", ""))
        cand = _candidate_for(example)
        decision, reason = classify_visit_rules(window, cand, {})
        if decision == "unsure":
            if llm is not None:
                visit, _, _ = classify_with_llm(llm, window, cand)
            else:
                # Mirrors classify_candidate's own no-LLM fallback: reject.
                visit = False
        else:
            visit = decision == "visit"

        gold_visit = example["gold_label"] == "visit"
        if visit and gold_visit:
            tp += 1
        elif visit and not gold_visit:
            fp += 1
        elif not visit and gold_visit:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and (precision + recall) > 0
        else None
    )
    return {
        "n": len(eval_set),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main() -> int:
    import argparse

    from scripts.pipeline_metrics import record_eval_metrics

    parser = argparse.ArgumentParser(description="Evaluate visit classifier against data/eval_set.json")
    parser.add_argument("--with-llm", action="store_true", help="also resolve 'unsure' rule outcomes with the LLM arbiter")
    args = parser.parse_args()

    eval_set = load_eval_set()
    if not eval_set:
        logger.warning(
            "data/eval_set.json is empty — nothing to evaluate. "
            "Add labeled (candidate_name, window_text, gold_label) examples first."
        )
        return 0

    llm = None
    if args.with_llm:
        from scripts.extract_locales import get_llm

        llm = get_llm()

    metrics = evaluate(eval_set, llm=llm)
    logger.info(
        f"eval_set n={metrics['n']} precision={metrics['precision']} "
        f"recall={metrics['recall']} f1={metrics['f1']}"
    )
    record_eval_metrics(metrics, with_llm=args.with_llm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
