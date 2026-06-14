"""Tests for validate_data: schema validation gate."""

from __future__ import annotations

__author__ = "Luca Ostinelli"

from unittest.mock import patch, MagicMock


class TestValidateDataMain:
    def _run_main(self):
        from scripts.validate_data import main
        return main()

    def test_returns_zero_when_all_valid(self):
        empty_list = []
        with patch("scripts.validate_data.load_json", return_value=empty_list), \
             patch("scripts.validate_data.validate_channels"), \
             patch("scripts.validate_data.validate_videos"), \
             patch("scripts.validate_data.validate_locales"), \
             patch("scripts.validate_data.validate_visits"), \
             patch("scripts.validate_data.validate_flagged_segments"), \
             patch("scripts.validate_data.validate_processed_videos"), \
             patch("scripts.validate_data.validate_skipped_videos"), \
             patch("scripts.validate_data.validate_corrections"):
            result = self._run_main()
        assert result == 0

    def test_returns_one_on_validation_error(self, capsys):
        def _raise(*_a, **_kw):
            raise ValueError("Schema mismatch")

        with patch("scripts.validate_data.load_json", return_value=[]), \
             patch("scripts.validate_data.validate_channels", side_effect=_raise), \
             patch("scripts.validate_data.validate_videos"), \
             patch("scripts.validate_data.validate_locales"), \
             patch("scripts.validate_data.validate_visits"), \
             patch("scripts.validate_data.validate_flagged_segments"), \
             patch("scripts.validate_data.validate_processed_videos"), \
             patch("scripts.validate_data.validate_skipped_videos"), \
             patch("scripts.validate_data.validate_corrections"):
            result = self._run_main()
        assert result == 1
        captured = capsys.readouterr()
        assert "channels.json" in captured.err

    def test_all_errors_reported(self, capsys):
        """Each failing file shows up in stderr, not just the first."""
        def _raise(*_a, **_kw):
            raise ValueError("bad")

        with patch("scripts.validate_data.load_json", return_value=[]), \
             patch("scripts.validate_data.validate_channels", side_effect=_raise), \
             patch("scripts.validate_data.validate_videos", side_effect=_raise), \
             patch("scripts.validate_data.validate_locales"), \
             patch("scripts.validate_data.validate_visits"), \
             patch("scripts.validate_data.validate_flagged_segments"), \
             patch("scripts.validate_data.validate_processed_videos"), \
             patch("scripts.validate_data.validate_skipped_videos"), \
             patch("scripts.validate_data.validate_corrections"):
            result = self._run_main()
        captured = capsys.readouterr()
        assert result == 1
        assert "channels.json" in captured.err
        assert "videos.json" in captured.err

    def test_load_json_os_error_is_caught(self, capsys):
        """An OSError reading a file counts as a validation error."""
        def _raise(*_a, **_kw):
            raise OSError("file gone")

        with patch("scripts.validate_data.load_json", side_effect=_raise):
            result = self._run_main()
        assert result == 1

    def test_returns_zero_on_empty_data_files(self):
        """Empty lists/dicts are valid (no items to validate)."""
        with patch("scripts.validate_data.load_json", return_value=[]), \
             patch("scripts.validate_data.validate_channels"), \
             patch("scripts.validate_data.validate_videos"), \
             patch("scripts.validate_data.validate_locales"), \
             patch("scripts.validate_data.validate_visits"), \
             patch("scripts.validate_data.validate_flagged_segments"), \
             patch("scripts.validate_data.validate_processed_videos"), \
             patch("scripts.validate_data.validate_skipped_videos"), \
             patch("scripts.validate_data.validate_corrections"):
            assert self._run_main() == 0
