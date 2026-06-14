"""Tests for pipeline_executor background finalize."""

__author__ = "Luca Ostinelli"

import pytest
from unittest.mock import patch

from scripts.pipeline_executor import (
    FinalizeJob,
    FinalizeResult,
    PipelineExecutor,
    _check_city_coherence,
    _is_trusted,
    _publishable_extraction,
    finalize_video,
    parallel_postprocess_enabled,
)


# ---------------------------------------------------------------------------
# parallel_postprocess_enabled
# ---------------------------------------------------------------------------


def test_parallel_enabled_on_cuda_by_default():
    assert parallel_postprocess_enabled(True) is True
    assert parallel_postprocess_enabled(False) is False


def test_parallel_forced_via_env(monkeypatch):
    monkeypatch.setenv("CIBOBUONO_PARALLEL_POSTPROCESS", "1")
    assert parallel_postprocess_enabled(False) is True


def test_parallel_disabled_via_env(monkeypatch):
    monkeypatch.setenv("CIBOBUONO_PARALLEL_POSTPROCESS", "off")
    assert parallel_postprocess_enabled(True) is False


# ---------------------------------------------------------------------------
# _check_city_coherence
# ---------------------------------------------------------------------------


class TestCityCoherence:
    def test_both_empty_is_coherent(self):
        assert _check_city_coherence({}, "") is True

    def test_intel_empty_is_coherent(self):
        assert _check_city_coherence({"geocoded_city": "Roma"}, "") is True

    def test_geocoded_empty_is_coherent(self):
        assert _check_city_coherence({}, "Roma") is True

    def test_exact_match(self):
        assert _check_city_coherence({"geocoded_city": "Roma"}, "Roma") is True

    def test_case_insensitive(self):
        assert _check_city_coherence({"geocoded_city": "ROMA"}, "roma") is True

    def test_partial_match_above_threshold(self):
        # "Città di Roma" should still match "Roma" with partial_ratio
        assert _check_city_coherence({"geocoded_city": "Città di Roma"}, "Roma") is True

    def test_mismatch(self):
        assert _check_city_coherence({"geocoded_city": "Milano"}, "Roma") is False

    def test_completely_different(self):
        assert _check_city_coherence({"geocoded_city": "Napoli"}, "Torino") is False


# ---------------------------------------------------------------------------
# _is_trusted
# ---------------------------------------------------------------------------


class TestIsTrusted:
    def test_exact_match(self):
        assert _is_trusted({"locale_name": "Da Remo"}, {"da remo"}) is True

    def test_substring_match(self):
        assert _is_trusted({"locale_name": "Osteria da Remo"}, {"da remo"}) is True

    def test_not_trusted(self):
        assert _is_trusted({"locale_name": "Pizzeria Napoli"}, {"da remo"}) is False

    def test_empty_trusted(self):
        assert _is_trusted({"locale_name": "Anything"}, set()) is False


# ---------------------------------------------------------------------------
# _publishable_extraction
# ---------------------------------------------------------------------------


class TestPublishableExtraction:
    def test_osm_verified_with_coords(self):
        ext = {"osm_verified": True, "osm_match_score": 85, "lat": 41.9, "lon": 12.5}
        ok, why = _publishable_extraction(ext, set())
        assert ok is True
        assert why == ""

    def test_osm_verified_weak_score_rejected(self):
        ext = {"osm_verified": True, "osm_match_score": 30, "lat": 41.9, "lon": 12.5}
        ok, why = _publishable_extraction(ext, set())
        assert ok is False
        assert "osm_weak_match" in why

    def test_not_osm_verified_rejected(self):
        ext = {"osm_verified": False, "lat": 41.9, "lon": 12.5}
        ok, why = _publishable_extraction(ext, set())
        assert ok is False
        assert why == "osm_not_found"

    def test_missing_coords_rejected(self):
        ext = {"osm_verified": True, "osm_match_score": 90, "lat": None, "lon": 12.5}
        ok, why = _publishable_extraction(ext, set())
        assert ok is False
        assert why == "missing_coords"

    def test_trusted_with_coords(self):
        ext = {"locale_name": "da remo", "lat": 41.9, "lon": 12.5}
        ok, why = _publishable_extraction(ext, {"da remo"})
        assert ok is True

    def test_trusted_no_coords_and_no_osm(self):
        ext = {"locale_name": "da remo", "lat": None, "lon": None, "osm_verified": False}
        ok, why = _publishable_extraction(ext, {"da remo"})
        assert ok is False
        assert why == "trusted_no_coords"


# ---------------------------------------------------------------------------
# finalize_video — happy path and city mismatch
# ---------------------------------------------------------------------------


def _make_job(**kw) -> FinalizeJob:
    defaults = dict(
        video_id="v1",
        channel_id="ch1",
        publish_date="2026-06-14",
        extractions=[],
        flagged_extractions=[],
        intel_city="",
    )
    defaults.update(kw)
    return FinalizeJob(**defaults)


def _patch_finalize(ext_list, verified_list, mapping, visits):
    return (
        patch("scripts.geocode_locales.geocode_extractions", return_value=(ext_list, [])),
        patch("scripts.verify_locales.verify_extractions", return_value=(verified_list, [])),
        patch("scripts.deduplicate_locales.deduplicate_locales", return_value=([], mapping)),
        patch("scripts.populate_json.populate_visits", return_value=visits),
        patch("scripts.populate_json.populate_flagged"),
        patch("scripts.fetch_videos.update_video_status"),
        patch("scripts.populate_json.update_processed_videos"),
    )


def test_finalize_happy_path():
    ext = {"locale_name": "Da Remo", "city": "Roma", "geocoded_city": "Roma",
           "osm_verified": True, "osm_match_score": 85, "lat": 41.9, "lon": 12.5, "confidence": 0.9}
    job = _make_job(extractions=[ext], intel_city="Roma")
    mapping = [{"locale_id": "l1", "locale_name": "Da Remo"}]
    patches = _patch_finalize([ext], [ext], mapping, [{"visit_id": "x"}])
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6]:
        result = finalize_video(job)
    assert result.outcome == "processed"
    assert result.visits_created == 1
    assert result.extractions_attempted == 1
    assert result.geocoded == 1
    assert result.osm_verified == 1
    assert result.published == 1
    assert result.city_mismatches == 0
    assert result.confidences == [0.9]


def test_finalize_city_mismatch_flags_extraction():
    ext = {"locale_name": "Luini", "city": "Milano", "geocoded_city": "Napoli",
           "osm_verified": True, "osm_match_score": 85, "lat": 45.5, "lon": 9.2, "confidence": 0.8}
    job = _make_job(extractions=[ext], intel_city="Milano")
    patches = _patch_finalize([ext], [], [], [])
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6]:
        result = finalize_video(job)
    assert result.city_mismatches == 1
    assert result.flagged_segments >= 1
    assert result.published == 0


def test_finalize_geocode_failure_non_trusted():
    job = _make_job(extractions=[{"locale_name": "Unknown Place"}])
    with patch("scripts.geocode_locales.geocode_extractions", return_value=([], [{"locale_name": "Unknown Place"}])):
        with patch("scripts.verify_locales.verify_extractions", return_value=([], [])):
            with patch("scripts.deduplicate_locales.deduplicate_locales", return_value=([], [])):
                with patch("scripts.populate_json.populate_visits", return_value=[]):
                    with patch("scripts.populate_json.populate_flagged"):
                        with patch("scripts.fetch_videos.update_video_status"):
                            with patch("scripts.populate_json.update_processed_videos"):
                                result = finalize_video(job)
    assert result.flagged_segments == 1
    assert result.geocoded == 0


def test_finalize_exception_returns_errored():
    job = _make_job(extractions=[{"locale_name": "X"}])
    with patch("scripts.geocode_locales.geocode_extractions", side_effect=RuntimeError("boom")):
        result = finalize_video(job)
    assert result.outcome == "errored"
    assert "boom" in result.error


# ---------------------------------------------------------------------------
# PipelineExecutor
# ---------------------------------------------------------------------------


def test_executor_submits_background_finalize():
    ex = PipelineExecutor(parallel_postprocess=True, max_pending_finalize=2)
    job = _make_job(video_id="v2")
    with patch(
        "scripts.pipeline_executor.finalize_video",
        return_value=FinalizeResult(video_id="v2"),
    ) as mock_fin:
        ex.submit_finalize(job)
        ex.drain_finalize()
        mock_fin.assert_called_once()
    ex.shutdown()


def test_executor_sync_finalize_when_disabled():
    ex = PipelineExecutor(parallel_postprocess=False)
    job = _make_job(video_id="v3")
    with patch(
        "scripts.pipeline_executor.finalize_video",
        return_value=FinalizeResult(video_id="v3", outcome="processed"),
    ) as mock_fin:
        result = ex.submit_finalize(job)
        mock_fin.assert_called_once()
    assert isinstance(result, FinalizeResult)
    assert result.outcome == "processed"
    ex.shutdown()


def test_executor_poll_completed_drains_done_futures():
    ex = PipelineExecutor(parallel_postprocess=True, max_pending_finalize=4)
    job = _make_job(video_id="v4")
    with patch(
        "scripts.pipeline_executor.finalize_video",
        return_value=FinalizeResult(video_id="v4", visits_created=2),
    ):
        ex.submit_finalize(job)
        results = ex.drain_finalize()
    assert any(r.visits_created == 2 for r in results)
    ex.shutdown()


def test_intel_prep_scheduled_and_taken():
    ex = PipelineExecutor(parallel_postprocess=False, io_workers=2)
    fake = __import__(
        "scripts.pipeline_executor", fromlist=["IntelPrepResult"]
    ).IntelPrepResult(
        video_id="v5",
        video_description="desc",
        youtube_extra={"chapters": []},
    )
    with patch("scripts.pipeline_executor._prepare_video_intel", return_value=fake):
        ex.schedule_intel_prep("v5", "Title")
        result = ex.take_intel_prep("v5", "Title")
    assert result.video_id == "v5"
    assert result.video_description == "desc"
    ex.shutdown()


def test_intel_prep_fallback_when_not_scheduled():
    ex = PipelineExecutor(parallel_postprocess=False, io_workers=2)
    fake = __import__(
        "scripts.pipeline_executor", fromlist=["IntelPrepResult"]
    ).IntelPrepResult(video_id="v6")
    with patch("scripts.pipeline_executor._prepare_video_intel", return_value=fake):
        result = ex.take_intel_prep("v6", "Title")
    assert result.video_id == "v6"
    ex.shutdown()
