"""
Tests for the continuous (--watch) pipeline mode.

The watch loop is a thin shell around run_pipeline(): catalog + process + push +
interruptible sleep, repeated until SIGINT/SIGTERM. We don't run the real
pipeline here — we patch run_pipeline so the test stays fast and hermetic.
"""

from __future__ import annotations

__author__ = "Luca Ostinelli"

import time
from unittest.mock import patch

import pytest

import scripts.run_pipeline as rp


@pytest.fixture(autouse=True)
def _reset_shutdown_flag():
    """Each test starts with a clean shutdown flag."""
    rp._pipeline_shutdown["graceful"] = False
    yield
    rp._pipeline_shutdown["graceful"] = False


class TestInterruptibleSleep:
    def test_sleeps_full_duration_when_not_interrupted(self):
        start = time.monotonic()
        rp._interruptible_sleep(0.2, slice_seconds=0.05)
        elapsed = time.monotonic() - start
        assert 0.15 <= elapsed <= 0.6

    def test_returns_early_when_shutdown_flag_set(self):
        """Setting the shutdown flag mid-sleep should unblock within one slice."""
        rp._pipeline_shutdown["graceful"] = True
        start = time.monotonic()
        rp._interruptible_sleep(10.0, slice_seconds=0.05)
        elapsed = time.monotonic() - start
        assert elapsed < 0.1, (
            f"Sleep should exit immediately when shutdown is already set, "
            f"but took {elapsed:.3f}s"
        )


class TestGitPushIfChanged:
    def test_no_changes_skips_push(self):
        """No uncommitted changes under data/ => _git_push() must NOT run."""
        with patch.object(rp, "_data_dir_has_uncommitted_changes", return_value=False), \
             patch.object(rp, "_git_push") as mock_push:
            assert rp._git_push_if_changed() is False
            mock_push.assert_not_called()

    def test_changes_trigger_push(self):
        with patch.object(rp, "_data_dir_has_uncommitted_changes", return_value=True), \
             patch.object(rp, "_git_push") as mock_push:
            assert rp._git_push_if_changed() is True
            mock_push.assert_called_once()


class TestRunPipelineWatch:
    def test_stops_when_shutdown_requested(self):
        """run_pipeline_watch must exit once the graceful flag is set, even
        if poll_interval is very large."""
        calls = {"n": 0}

        def fake_run_pipeline(**kwargs):
            calls["n"] += 1
            assert kwargs["_external_setup"] is True, (
                "Watch loop must call run_pipeline with _external_setup=True "
                "so it doesn't reset the shutdown flag mid-loop."
            )
            assert kwargs["no_dashboard"] is True, (
                "Watch mode is log-only by design."
            )
            # Simulate user pressing Ctrl+C after the first cycle.
            rp._pipeline_shutdown["graceful"] = True

        with patch.object(rp, "run_pipeline", side_effect=fake_run_pipeline):
            rp.run_pipeline_watch(poll_interval=3600)

        assert calls["n"] == 1

    def test_min_poll_interval_is_enforced(self):
        """A user-supplied poll_interval below WATCH_MIN_INTERVAL_SECONDS
        must be clamped, so a pathological --poll-interval 0 doesn't melt
        YouTube with requests."""
        from scripts.utils import WATCH_MIN_INTERVAL_SECONDS

        observed_sleep_arg: list[float] = []

        def fake_run_pipeline(**kwargs):
            # Trigger shutdown so we only do one iteration.
            rp._pipeline_shutdown["graceful"] = True

        def fake_sleep(total_seconds, slice_seconds=5.0):
            observed_sleep_arg.append(total_seconds)

        with patch.object(rp, "run_pipeline", side_effect=fake_run_pipeline), \
             patch.object(rp, "_interruptible_sleep", side_effect=fake_sleep):
            rp.run_pipeline_watch(poll_interval=0)

        # The loop terminates before _interruptible_sleep is called because
        # the shutdown flag is set inside the first run_pipeline call. So we
        # just assert that clamping happened by checking that calling with a
        # tiny interval did not crash — and verify the floor was applied by
        # running a second variant that lets the sleep happen.
        rp._pipeline_shutdown["graceful"] = False
        observed_sleep_arg.clear()

        cycle_count = {"n": 0}

        def fake_run_pipeline_two_cycles(**kwargs):
            cycle_count["n"] += 1
            if cycle_count["n"] >= 2:
                rp._pipeline_shutdown["graceful"] = True

        with patch.object(rp, "run_pipeline", side_effect=fake_run_pipeline_two_cycles), \
             patch.object(rp, "_interruptible_sleep", side_effect=fake_sleep):
            rp.run_pipeline_watch(poll_interval=1)

        assert observed_sleep_arg, "expected at least one sleep between cycles"
        assert all(s >= WATCH_MIN_INTERVAL_SECONDS for s in observed_sleep_arg), (
            f"poll_interval should be clamped to >= {WATCH_MIN_INTERVAL_SECONDS}, "
            f"got {observed_sleep_arg}"
        )

    def test_crash_in_cycle_does_not_stop_loop(self):
        """A single cycle exception is logged but the watch loop keeps going.
        Daemons must be resilient to transient yt-dlp / network errors."""
        cycle_count = {"n": 0}

        def fake_run_pipeline(**kwargs):
            cycle_count["n"] += 1
            if cycle_count["n"] == 1:
                raise RuntimeError("simulated yt-dlp blip")
            rp._pipeline_shutdown["graceful"] = True

        with patch.object(rp, "run_pipeline", side_effect=fake_run_pipeline), \
             patch.object(rp, "_interruptible_sleep"):
            rp.run_pipeline_watch(poll_interval=60)

        assert cycle_count["n"] == 2, (
            "Loop should have continued past the first cycle's exception"
        )

    def test_systemexit_propagates(self):
        """SystemExit (e.g. missing GGUF model) MUST kill the daemon — it's a
        configuration error that won't fix itself by retrying."""
        def fake_run_pipeline(**kwargs):
            raise SystemExit(1)

        with patch.object(rp, "run_pipeline", side_effect=fake_run_pipeline), \
             patch.object(rp, "_interruptible_sleep"):
            with pytest.raises(SystemExit):
                rp.run_pipeline_watch(poll_interval=60)
