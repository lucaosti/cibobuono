"""
pipeline_metrics.py — Calibration stats recorded at the end of each pipeline run.

Appends one JSON record per run to logs/pipeline_metrics.json so trends in
geocoding success, OSM match rates, confidence distributions, and city-coherence
can be tracked over time.
"""

from __future__ import annotations

__author__ = "Luca Ostinelli"

import json
import math
import os
import tempfile
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from scripts.utils import LOGS_DIR, ensure_dirs, setup_logging

if TYPE_CHECKING:
    from scripts.pipeline_executor import FinalizeResult

logger = setup_logging("pipeline_metrics")

METRICS_FILE = LOGS_DIR / "pipeline_metrics.json"
EVAL_METRICS_FILE = LOGS_DIR / "eval_metrics.json"


def compute_run_metrics(results: list["FinalizeResult"]) -> dict:
    """Aggregate per-video FinalizeResult counters into a single run record."""
    processed = [r for r in results if r.outcome == "processed"]
    errored = [r for r in results if r.outcome == "errored"]

    attempted = sum(r.extractions_attempted for r in processed)
    geocoded = sum(r.geocoded for r in processed)
    osm_verified = sum(r.osm_verified for r in processed)
    published = sum(r.published for r in processed)
    city_mismatches = sum(r.city_mismatches for r in processed)
    all_confidences = [c for r in processed for c in r.confidences]
    visits_created = sum(r.visits_created for r in processed)
    flagged = sum(r.flagged_segments for r in processed)

    def _rate(num: int, den: int) -> float | None:
        return round(num / den, 4) if den else None

    def _mean(xs: list[float]) -> float | None:
        return round(sum(xs) / len(xs), 4) if xs else None

    def _stdev(xs: list[float]) -> float | None:
        if len(xs) < 2:
            return None
        m = sum(xs) / len(xs)
        return round(math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs)), 4)

    return {
        "videos_processed": len(processed),
        "videos_errored": len(errored),
        "extractions_attempted": attempted,
        "geocode_rate": _rate(geocoded, attempted),
        "osm_rate": _rate(osm_verified, geocoded),
        "publish_rate": _rate(published, osm_verified),
        "city_mismatch_rate": _rate(city_mismatches, geocoded),
        "visits_created": visits_created,
        "flagged_total": flagged,
        "confidence_mean": _mean(all_confidences),
        "confidence_stdev": _stdev(all_confidences),
        "confidence_n": len(all_confidences),
    }


def record_run_metrics(
    results: list["FinalizeResult"],
    *,
    run_id: str = "",
    extra: dict | None = None,
) -> dict:
    """Compute metrics, append to METRICS_FILE, and return the record."""
    ensure_dirs()
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        **compute_run_metrics(results),
        **(extra or {}),
    }

    existing: list[dict] = []
    if METRICS_FILE.exists():
        try:
            with open(METRICS_FILE, encoding="utf-8") as f:
                existing = json.load(f)
            if not isinstance(existing, list):
                existing = []
        except (json.JSONDecodeError, OSError):
            existing = []

    existing.append(record)

    fd, tmp = tempfile.mkstemp(
        suffix=".json.tmp", prefix=".pipeline_metrics.", dir=str(LOGS_DIR), text=True
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
        os.replace(tmp, METRICS_FILE)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise

    logger.info(
        "Run metrics recorded: %d processed, %d visits, geocode_rate=%.2f, osm_rate=%.2f",
        record["videos_processed"],
        record["visits_created"],
        record["geocode_rate"] or 0,
        record["osm_rate"] or 0,
    )
    return record


def record_eval_metrics(metrics: dict, **extra) -> dict:
    """Append a scripts.eval_pipeline.evaluate() result to EVAL_METRICS_FILE.

    Trend-tracks classifier precision/recall/F1 over time so each redundancy
    or calibration change (self-consistency ensemble, Perceptor fusion, Platt
    scaling) can be attributed a before/after delta rather than judged by feel.
    """
    ensure_dirs()
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        **metrics,
        **extra,
    }

    existing: list[dict] = []
    if EVAL_METRICS_FILE.exists():
        try:
            with open(EVAL_METRICS_FILE, encoding="utf-8") as f:
                existing = json.load(f)
            if not isinstance(existing, list):
                existing = []
        except (json.JSONDecodeError, OSError):
            existing = []

    existing.append(record)

    fd, tmp = tempfile.mkstemp(
        suffix=".json.tmp", prefix=".eval_metrics.", dir=str(LOGS_DIR), text=True
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
        os.replace(tmp, EVAL_METRICS_FILE)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise

    logger.info(
        "Eval metrics recorded: n=%d precision=%s recall=%s f1=%s",
        record.get("n", 0),
        record.get("precision"),
        record.get("recall"),
        record.get("f1"),
    )
    return record
