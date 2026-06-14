"""Tests for scripts.pipeline_metrics — calibration stats accumulation."""

__author__ = "Luca Ostinelli"

import json

import pytest

from scripts.pipeline_executor import FinalizeResult
from scripts.pipeline_metrics import METRICS_FILE, compute_run_metrics, record_run_metrics


def _result(**kw) -> FinalizeResult:
    defaults = dict(
        video_id="v1",
        outcome="processed",
        visits_created=0,
        flagged_segments=0,
        extractions_attempted=0,
        geocoded=0,
        osm_verified=0,
        published=0,
        city_mismatches=0,
        confidences=[],
    )
    defaults.update(kw)
    return FinalizeResult(**defaults)


# ---------------------------------------------------------------------------
# compute_run_metrics
# ---------------------------------------------------------------------------


class TestComputeRunMetrics:
    def test_empty_results(self):
        m = compute_run_metrics([])
        assert m["videos_processed"] == 0
        assert m["videos_errored"] == 0
        assert m["extractions_attempted"] == 0
        assert m["geocode_rate"] is None
        assert m["confidence_mean"] is None

    def test_single_perfect_video(self):
        r = _result(
            extractions_attempted=3,
            geocoded=3,
            osm_verified=3,
            published=3,
            city_mismatches=0,
            visits_created=3,
            confidences=[0.9, 0.85, 0.95],
        )
        m = compute_run_metrics([r])
        assert m["videos_processed"] == 1
        assert m["geocode_rate"] == 1.0
        assert m["osm_rate"] == 1.0
        assert m["publish_rate"] == 1.0
        assert m["city_mismatch_rate"] == 0.0
        assert m["confidence_mean"] == pytest.approx(0.9, abs=0.01)
        assert m["confidence_n"] == 3

    def test_partial_geocode(self):
        r = _result(extractions_attempted=4, geocoded=2, osm_verified=2, published=2, confidences=[0.8, 0.7])
        m = compute_run_metrics([r])
        assert m["geocode_rate"] == pytest.approx(0.5, abs=0.01)
        assert m["osm_rate"] == 1.0

    def test_errored_videos_excluded_from_rates(self):
        good = _result(extractions_attempted=2, geocoded=2, osm_verified=2, published=2, confidences=[0.9, 0.9])
        bad = _result(video_id="v2", outcome="errored")
        m = compute_run_metrics([good, bad])
        assert m["videos_processed"] == 1
        assert m["videos_errored"] == 1
        assert m["extractions_attempted"] == 2

    def test_city_mismatch_rate(self):
        r = _result(extractions_attempted=4, geocoded=4, osm_verified=3, published=3, city_mismatches=1)
        m = compute_run_metrics([r])
        assert m["city_mismatch_rate"] == pytest.approx(0.25, abs=0.01)

    def test_confidence_stdev_single_value(self):
        r = _result(confidences=[0.9])
        m = compute_run_metrics([r])
        assert m["confidence_stdev"] is None

    def test_confidence_stdev_multiple(self):
        r = _result(confidences=[0.8, 1.0])
        m = compute_run_metrics([r])
        assert m["confidence_stdev"] is not None
        assert m["confidence_stdev"] > 0

    def test_aggregates_across_multiple_videos(self):
        r1 = _result(video_id="v1", extractions_attempted=2, geocoded=2, osm_verified=2, published=2, visits_created=2, confidences=[0.9, 0.8])
        r2 = _result(video_id="v2", extractions_attempted=3, geocoded=3, osm_verified=3, published=3, visits_created=3, confidences=[0.7, 0.85, 0.95])
        m = compute_run_metrics([r1, r2])
        assert m["visits_created"] == 5
        assert m["extractions_attempted"] == 5
        assert m["confidence_n"] == 5


# ---------------------------------------------------------------------------
# record_run_metrics
# ---------------------------------------------------------------------------


class TestRecordRunMetrics:
    def test_writes_to_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr("scripts.pipeline_metrics.METRICS_FILE", tmp_path / "metrics.json")
        monkeypatch.setattr("scripts.pipeline_metrics.LOGS_DIR", tmp_path)
        r = _result(extractions_attempted=1, geocoded=1, osm_verified=1, published=1, visits_created=1, confidences=[0.9])
        record = record_run_metrics([r], run_id="test-run")
        assert record["run_id"] == "test-run"
        assert record["videos_processed"] == 1

        saved = json.loads((tmp_path / "metrics.json").read_text())
        assert isinstance(saved, list)
        assert len(saved) == 1
        assert saved[0]["run_id"] == "test-run"

    def test_appends_to_existing(self, tmp_path, monkeypatch):
        mf = tmp_path / "metrics.json"
        mf.write_text(json.dumps([{"run_id": "old", "ts": "2026-01-01"}]))
        monkeypatch.setattr("scripts.pipeline_metrics.METRICS_FILE", mf)
        monkeypatch.setattr("scripts.pipeline_metrics.LOGS_DIR", tmp_path)
        record_run_metrics([], run_id="new")
        saved = json.loads(mf.read_text())
        assert len(saved) == 2
        assert saved[0]["run_id"] == "old"
        assert saved[1]["run_id"] == "new"

    def test_recovers_from_corrupted_file(self, tmp_path, monkeypatch):
        mf = tmp_path / "metrics.json"
        mf.write_text("{not valid json")
        monkeypatch.setattr("scripts.pipeline_metrics.METRICS_FILE", mf)
        monkeypatch.setattr("scripts.pipeline_metrics.LOGS_DIR", tmp_path)
        record_run_metrics([], run_id="after-corruption")
        saved = json.loads(mf.read_text())
        assert len(saved) == 1

    def test_includes_extra_fields(self, tmp_path, monkeypatch):
        mf = tmp_path / "metrics.json"
        monkeypatch.setattr("scripts.pipeline_metrics.METRICS_FILE", mf)
        monkeypatch.setattr("scripts.pipeline_metrics.LOGS_DIR", tmp_path)
        record = record_run_metrics([], run_id="x", extra={"channel": "ch1"})
        assert record["channel"] == "ch1"

    def test_empty_results_does_not_raise(self, tmp_path, monkeypatch):
        monkeypatch.setattr("scripts.pipeline_metrics.METRICS_FILE", tmp_path / "m.json")
        monkeypatch.setattr("scripts.pipeline_metrics.LOGS_DIR", tmp_path)
        record = record_run_metrics([])
        assert record["videos_processed"] == 0
        assert record["geocode_rate"] is None
