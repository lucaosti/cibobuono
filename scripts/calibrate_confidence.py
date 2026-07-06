"""
calibrate_confidence.py — Platt-scaling recalibration of visit confidence.

The hand-tuned linear blend in extract_pipeline._visit_confidence has never
been checked against real outcomes. data/corrections.json ("hide"/"edit" on
published visits) is a weak-supervision ground truth already collected by
review_queue.py but otherwise unused for this purpose.

Fits a 2-parameter logistic (Platt) recalibration rather than isotonic
regression: isotonic needs many more points than corrections.json is likely
to hold for a while, and a 2-parameter fit degrades gracefully on small data.
Below MIN_SAMPLES labeled pairs, calibration is left unfitted and
apply_platt() is a no-op — callers keep using the original linear formula.

Run standalone to refit after new corrections accumulate:
    python -m scripts.calibrate_confidence
"""

from __future__ import annotations

__author__ = "Luca Ostinelli"

import json
import math
import os
import tempfile

from scripts.utils import DATA_DIR, load_json, setup_logging

logger = setup_logging("calibrate_confidence")

CORRECTIONS_JSON = DATA_DIR / "corrections.json"
VISITS_JSON = DATA_DIR / "visits.json"
CALIBRATION_JSON = DATA_DIR / "calibration.json"

# Minimum labeled (confidence, outcome) pairs before trusting a fitted map
# over the original linear formula.
MIN_SAMPLES = 30

_EPS = 1e-4


def _logit(p: float) -> float:
    p = min(max(p, _EPS), 1.0 - _EPS)
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _build_training_pairs() -> list[tuple[float, int]]:
    """(predicted_confidence, label) pairs.

    label=0 when a visit's locale was later "hidden" (confirmed false
    positive); label=1 otherwise. Weak-supervision caveat: absence of a
    correction is treated as an implicit positive, which also includes visits
    nobody has reviewed yet — not just genuinely correct ones. This noise is
    exactly why MIN_SAMPLES gates the fit and why Platt (2 parameters) is used
    instead of a higher-capacity model that would overfit the noise.
    """
    visits = load_json(VISITS_JSON)
    corrections = load_json(CORRECTIONS_JSON)
    hidden_ids = {c.get("locale_id") for c in corrections if c.get("type") == "hide"}

    pairs: list[tuple[float, int]] = []
    for v in visits:
        conf = v.get("llm_confidence")
        if conf is None:
            continue
        label = 0 if v.get("locale_id") in hidden_ids else 1
        pairs.append((float(conf), label))
    return pairs


def fit_platt(
    pairs: list[tuple[float, int]],
    *,
    lr: float = 0.1,
    epochs: int = 500,
) -> tuple[float, float] | None:
    """Fit calibrated_p = sigmoid(a * logit(p) + b) via gradient descent.

    Returns None (unfitted) when there are fewer than MIN_SAMPLES pairs.
    """
    if len(pairs) < MIN_SAMPLES:
        return None

    a, b = 1.0, 0.0
    n = len(pairs)
    xs = [_logit(p) for p, _ in pairs]
    ys = [y for _, y in pairs]
    for _ in range(epochs):
        grad_a = grad_b = 0.0
        for x, y in zip(xs, ys):
            z = a * x + b
            pred = _sigmoid(z)
            err = pred - y
            grad_a += err * x
            grad_b += err
        a -= lr * grad_a / n
        b -= lr * grad_b / n
    return a, b


def apply_platt(conf: float, params: tuple[float, float] | None) -> float:
    """Recalibrate conf with a fitted (a, b); no-op if params is None."""
    if params is None:
        return conf
    a, b = params
    return _sigmoid(a * _logit(conf) + b)


def load_calibration() -> tuple[float, float] | None:
    """Load fitted (a, b) from CALIBRATION_JSON, or None if unfitted/missing."""
    if not CALIBRATION_JSON.exists():
        return None
    try:
        with open(CALIBRATION_JSON, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Cannot read {CALIBRATION_JSON}: {e}")
        return None
    if not isinstance(data, dict) or not data.get("fitted"):
        return None
    try:
        return float(data["a"]), float(data["b"])
    except (KeyError, TypeError, ValueError):
        return None


def save_calibration(params: tuple[float, float] | None, *, n_samples: int) -> None:
    """Persist the fit result (or the fact that there wasn't enough data)."""
    CALIBRATION_JSON.parent.mkdir(parents=True, exist_ok=True)
    if params is None:
        payload = {"fitted": False, "n_samples": n_samples}
    else:
        a, b = params
        payload = {"fitted": True, "a": a, "b": b, "n_samples": n_samples}

    fd, tmp = tempfile.mkstemp(
        suffix=".json.tmp", prefix=".calibration.", dir=str(CALIBRATION_JSON.parent), text=True
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp, CALIBRATION_JSON)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def main() -> int:
    pairs = _build_training_pairs()
    params = fit_platt(pairs)
    save_calibration(params, n_samples=len(pairs))
    if params is None:
        logger.info(
            f"Only {len(pairs)} labeled pairs (< {MIN_SAMPLES}); "
            "calibration left unfitted, linear formula stays in effect."
        )
    else:
        a, b = params
        logger.info(f"Fitted Platt scaling on {len(pairs)} pairs: a={a:.3f} b={b:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
