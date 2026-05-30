"""Tests for pipeline_executor background finalize."""

__author__ = "Luca Ostinelli"

from unittest.mock import patch

from scripts.pipeline_executor import (
    FinalizeJob,
    FinalizeResult,
    PipelineExecutor,
    finalize_video,
    parallel_postprocess_enabled,
)


def test_parallel_enabled_on_cuda_by_default():
    assert parallel_postprocess_enabled(True) is True
    assert parallel_postprocess_enabled(False) is False


def test_finalize_job_runs_geocode_chain():
    job = FinalizeJob(
        video_id="v1",
        channel_id="ch1",
        publish_date="2026-05-30",
        extractions=[{"locale_name": "Da Remo", "city": "Roma"}],
        flagged_extractions=[],
    )
    mapping = [{"locale_id": "l1", "locale_name": "Da Remo"}]
    with patch("scripts.geocode_locales.geocode_extractions", return_value=(job.extractions, [])):
        with patch("scripts.verify_locales.verify_extractions", return_value=(job.extractions, [])):
            with patch("scripts.deduplicate_locales.deduplicate_locales", return_value=([], mapping)):
                with patch("scripts.populate_json.populate_visits", return_value=[{"visit_id": "x"}]):
                    with patch("scripts.populate_json.populate_flagged"):
                        with patch("scripts.fetch_videos.update_video_status"):
                            with patch("scripts.populate_json.update_processed_videos"):
                                result = finalize_video(job)
    assert result.outcome == "processed"
    assert result.visits_created == 1


def test_executor_submits_background_finalize():
    ex = PipelineExecutor(parallel_postprocess=True, max_pending_finalize=2)
    job = FinalizeJob("v2", "ch", "2026-05-30", [], [])
    with patch(
        "scripts.pipeline_executor.finalize_video",
        return_value=FinalizeResult(video_id="v2"),
    ) as mock_fin:
        ex.submit_finalize(job)
        ex.drain_finalize()
        mock_fin.assert_called_once()
    ex.shutdown()


def test_intel_prep_scheduled_and_taken():
    ex = PipelineExecutor(parallel_postprocess=False, io_workers=2)
    fake = __import__(
        "scripts.pipeline_executor", fromlist=["IntelPrepResult"]
    ).IntelPrepResult(
        video_id="v3",
        video_description="desc",
        youtube_extra={"chapters": []},
    )
    with patch("scripts.pipeline_executor._prepare_video_intel", return_value=fake):
        ex.schedule_intel_prep("v3", "Title")
        result = ex.take_intel_prep("v3", "Title")
    assert result.video_id == "v3"
    assert result.video_description == "desc"
    ex.shutdown()
